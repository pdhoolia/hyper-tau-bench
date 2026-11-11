"""Two commands for whoever starts builds as pods from outside (a runner, an operator).

    python -m tau2.hyper.sandbox.kube_tools probe --target HOST:PORT [--target …] [--harness NAME …]
    python -m tau2.hyper.sandbox.kube_tools sweep --selector KEY=VALUE[,KEY=VALUE]

``probe`` answers one question about a cluster before a build runs there: is a
sandbox pod isolated? It starts a sandbox pod behind a build's own network
rules and a control pod with no rule, and tries the same addresses from both.
A cluster whose network plugin does not enforce NetworkPolicy lets the sandbox
reach what the control reaches, and the answer is no. It also says under which
user id the sandbox runs and whether each named harness starts there.

``sweep`` removes what builds left behind: everything that carries the given
labels (``TAU2_KUBE_LABELS`` puts a caller's labels on all a build makes). A
build that was killed, or whose runner died, is removed this way.

Both read the same environment as a build (``TAU2_KUBE_*``) and print one JSON
object on standard output.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from typing import Optional

from tau2.hyper.sandbox import kube_runtime
from tau2.hyper.sandbox.kube_runtime import (
    LABEL_BUILD,
    LABEL_ROLE,
    REQUESTS,
    ROLE_GATEWAY,
    KubeSettings,
    _container_security,
    _create,
    _jsonpath,
    _pod_base,
    _wait_ready,
    delete_resources,
    gateway_service_manifest,
    network_policy_manifests,
    sandbox_pod_manifest,
)

ROLE_CONTROL = "probe-control"

# The command each harness's driver starts, by the harness's name.
HARNESS_COMMANDS = {
    "claude-code": "claude",
    "codex": "codex",
    "opencode": "opencode",
    "bob": "bob",
}

# One connection attempt per address. A name is looked up here, in the pod that
# has a name service; the sandbox has none and is given the numbers.
_CONNECT = r"""
import json, socket, sys
out = []
for target in sys.argv[1:]:
    host, _, port = target.rpartition(":")
    entry = {"target": target, "address": None, "port": int(port), "reached": False}
    try:
        entry["address"] = socket.getaddrinfo(host, int(port), socket.AF_INET, socket.SOCK_STREAM)[0][4][0]
        with socket.create_connection((entry["address"], int(port)), timeout=3):
            entry["reached"] = True
    except Exception as exc:
        entry["error"] = type(exc).__name__ + ": " + str(exc)
    out.append(entry)
print(json.dumps(out))
"""

# Who the sandbox runs as, and whether each harness's command starts there.
_STARTS = r"""
import json, os, subprocess, sys
home = os.environ.get("HOME", "/")
out = {"uid": os.getuid(), "gid": os.getgid(), "home": home, "home_writable": os.access(home, os.W_OK), "harnesses": {}}
for item in sys.argv[1:]:
    name, _, command = item.partition("=")
    try:
        ran = subprocess.run([command, "--version"], capture_output=True, text=True, timeout=60)
        said = (ran.stdout or ran.stderr).strip().splitlines()
        out["harnesses"][name] = {"started": ran.returncode == 0, "version": said[-1] if said else ""}
    except FileNotFoundError:
        out["harnesses"][name] = {"started": False, "detail": "the image has no `%s` command" % command}
    except subprocess.TimeoutExpired:
        out["harnesses"][name] = {"started": False, "detail": "`%s --version` did not end in 60 seconds" % command}
print(json.dumps(out))
"""


def control_pod_manifest(*, name: str, build: str, image: str) -> dict:
    """A pod of the probe that no network rule selects: what the cluster reaches without one."""
    return _pod_base(
        name,
        {LABEL_BUILD: build, LABEL_ROLE: ROLE_CONTROL},
        {
            "containers": [
                {
                    "name": "control",
                    "image": image,
                    "command": ["sleep", "infinity"],
                    "resources": {"requests": dict(REQUESTS[ROLE_GATEWAY])},
                    "securityContext": _container_security(read_only_root=True),
                }
            ]
        },
    )


def _run_json(settings: KubeSettings, pod: str, script: str, args: list[str], *, timeout: float):
    result = kube_runtime.kube(
        settings, ["exec", pod, "--", "python", "-I", "-c", script, *args], timeout=timeout
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"The probe could not run in pod {pod}: {result.stderr.strip()[-500:]}"
        )
    return json.loads(result.stdout.strip().splitlines()[-1])


def isolation(control: list[dict], sandbox: list[dict]) -> tuple[Optional[bool], str]:
    """The answer, and why: no when the sandbox reached an address, unknown when nothing could be tested."""
    through = [entry["target"] for entry in sandbox if entry["reached"]]
    if through:
        return False, (
            f"the sandbox reached {', '.join(through)}: "
            "this cluster does not enforce the sandbox's network rule"
        )
    tested = [entry["target"] for entry in control if entry["reached"]]
    if not tested:
        return None, (
            "no address answered a pod with no rule, so the sandbox's rule could not be tested"
        )
    return True, (
        f"a pod with no rule reached {', '.join(tested)}, and the sandbox reached none of them"
    )


def probe(
    settings: KubeSettings,
    *,
    image: str,
    targets: list[str],
    harnesses: list[str],
    rules: bool = True,
) -> dict:
    """Start a sandbox behind a build's rules and a control beside it, and try the addresses from both."""
    name = f"tau2-probe-{uuid.uuid4().hex[:12]}"
    control = f"{name}-control"
    service = f"{name}-model-gateway"
    pods: list[str] = []
    others: list[str] = []

    def make(manifest: dict) -> None:
        # Listed before it is made, so that a half-made one is removed too.
        resource = f"{manifest['kind'].lower()}/{manifest['metadata']['name']}"
        (pods if manifest["kind"] == "Pod" else others).append(resource)
        _create(settings, manifest)

    try:
        if rules:
            for policy in network_policy_manifests(build=name):
                make(policy)
        make(gateway_service_manifest(name=service, build=name))
        gateway_ip = _jsonpath(settings, f"service/{service}", "{.spec.clusterIP}")
        if not gateway_ip or gateway_ip == "None":
            raise RuntimeError("The probe's Service has no cluster address")
        make(sandbox_pod_manifest(name=name, image=image, env={}, gateway_ip=gateway_ip))
        make(control_pod_manifest(name=control, build=name, image=image))
        _wait_ready(settings, name)
        _wait_ready(settings, control)

        scc = _jsonpath(
            settings, f"pod/{name}", r"{.metadata.annotations.openshift\.io/scc}"
        )
        from_control = _run_json(
            settings, control, _CONNECT, targets, timeout=30 + 5 * len(targets)
        )
        numbers = [
            f"{entry['address']}:{entry['port']}" for entry in from_control if entry["reached"]
        ]
        from_sandbox = (
            _run_json(settings, name, _CONNECT, numbers, timeout=30 + 5 * len(numbers))
            if numbers
            else []
        )
        # The sandbox's entries carry the names the control resolved.
        names = {
            f"{entry['address']}:{entry['port']}": entry["target"] for entry in from_control
        }
        for entry in from_sandbox:
            entry["target"] = names.get(entry["target"], entry["target"])
        starts = _run_json(
            settings,
            name,
            _STARTS,
            [f"{harness}={HARNESS_COMMANDS[harness]}" for harness in harnesses],
            timeout=60 + 70 * len(harnesses),
        )
        isolated, detail = isolation(from_control, from_sandbox)
        return {
            "isolated": isolated,
            "detail": detail,
            "rules": rules,
            "namespace": settings.namespace,
            "image": image,
            "security_context_constraint": scc or None,
            "control": from_control,
            "sandbox": from_sandbox,
            **starts,
        }
    finally:
        # The pods first: a sandbox never stands without its rule.
        delete_resources(settings, pods)
        delete_resources(settings, others)


def sweep(settings: KubeSettings, selector: str) -> list[str]:
    """Remove everything that carries the selector's labels; returns what was asked to go.

    A build's Secret is found by its name, not by a list: the path then needs
    no right to list Secrets, which would be a right to read every Secret of
    the namespace. Each build's Service stands from before its Secret is made
    until after it is removed, so the Services name every build.
    """
    if not selector or "=" not in selector:
        raise ValueError("sweep needs a selector of labels, key=value: it never removes everything")
    listed = kube_runtime.kube(
        settings,
        ["get", "pod,service,networkpolicy", "-l", selector, "-o", "json"],
        timeout=60,
    )
    if listed.returncode != 0:
        raise RuntimeError(f"Could not list what {selector} selects: {listed.stderr.strip()[-500:]}")
    pods: list[str] = []
    others: list[str] = []
    builds: set[str] = set()
    for item in json.loads(listed.stdout or "{}").get("items", []):
        metadata = item.get("metadata", {})
        resource = f"{item.get('kind', '').lower()}/{metadata.get('name')}"
        (pods if item.get("kind") == "Pod" else others).append(resource)
        build = metadata.get("labels", {}).get(LABEL_BUILD)
        if build:
            builds.add(build)
    secrets = [f"secret/{build}-model-gateway" for build in sorted(builds)]
    # The pods first: a sandbox never stands without its rule.
    delete_resources(settings, pods)
    delete_resources(settings, secrets + others)
    return pods + secrets + others


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="kube_tools", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    probing = commands.add_parser("probe", help="is a sandbox pod isolated on this cluster?")
    probing.add_argument(
        "--target",
        action="append",
        default=[],
        metavar="HOST:PORT",
        help="an address the sandbox must not reach, that answers a pod with no rule (repeat)",
    )
    probing.add_argument(
        "--harness",
        action="append",
        default=[],
        choices=sorted(HARNESS_COMMANDS),
        help="a harness whose command is started in the sandbox (repeat)",
    )
    probing.add_argument("--image", help="the construction image; default TAU2_KUBE_IMAGE")
    probing.add_argument(
        "--no-rules",
        action="store_true",
        help="leave the build's network rules out: the answer of a cluster that enforces none",
    )
    sweeping = commands.add_parser("sweep", help="remove what builds with these labels left")
    sweeping.add_argument("--selector", required=True, metavar="KEY=VALUE[,KEY=VALUE]")
    args = parser.parse_args(argv)

    settings = KubeSettings.from_environment()
    try:
        if args.command == "sweep":
            print(json.dumps({"removed": sweep(settings, args.selector)}))
            return 0
        image = args.image or settings.image
        if not image:
            parser.error("probe needs the construction image: --image or TAU2_KUBE_IMAGE")
        if not args.target:
            parser.error("probe needs at least one --target HOST:PORT")
        for target in args.target:
            host, _, port = target.rpartition(":")
            if not host or not port.isdigit():
                parser.error(f"a target is HOST:PORT, not {target!r}")
        print(
            json.dumps(
                probe(
                    settings,
                    image=image,
                    targets=args.target,
                    harnesses=args.harness,
                    rules=not args.no_rules,
                )
            )
        )
        return 0
    except (RuntimeError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
