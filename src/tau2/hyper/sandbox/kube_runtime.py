"""A second way to start a build's containers: pods, beside Docker.

Docker stays the default. ``TAU2_SANDBOX_BACKEND=kubernetes`` picks this path,
which talks to the cluster through its command line (``kubectl``, or ``oc`` on
OpenShift) the way the Docker path talks to ``docker``:

- the sandbox is a pod that sleeps, and the Developer's commands run in it by
  ``exec``;
- the model gateway is a pod beside it, behind a Service of its own;
- the scoring is a pod with no network, fed over the standard input of an
  ``exec``;
- the kit is copied into a volume of the sandbox pod and copied out again,
  where Docker binds a host folder;
- the internal network is a pair of NetworkPolicies: the sandbox reaches the
  model gateway and nothing else, and nothing reaches the sandbox.

Every pod asks for nothing privileged and names no user id, so a cluster that
assigns one (OpenShift's ``restricted-v2``) admits it. The rights it needs are
those of a project's admin: pods, ``pods/exec``, ``pods/log``, services,
secrets and network policies in one namespace.

Only what a Claude Code build needs is here. A harness that keeps a record
below its runtime home can read it out (``read_runtime_tree``), but no other
harness has been run on this path.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Optional

from loguru import logger

from tau2.hyper.sandbox.model_gateway import (
    MODEL_GATEWAY_HOST,
    MODEL_GATEWAY_PORT,
    ModelGatewaySpec,
)
from tau2.hyper.sandbox.native_runtime import (
    NATIVE_NETWORK_PROFILE,
    NativeMount,
    NativeProcessEvent,
    NativeProcessResult,
    NativeRuntimeConfig,
    NativeSandboxRuntime,
    run_supervised_process,
)

BACKEND_ENV = "TAU2_SANDBOX_BACKEND"
DOCKER_BACKEND = "docker"
KUBE_BACKEND = "kubernetes"

LABEL_BUILD = "tau2.hyper/build"
LABEL_ROLE = "tau2.hyper/role"
ROLE_SANDBOX = "sandbox"
ROLE_GATEWAY = "model-gateway"
ROLE_SCORING = "scoring"
SCORING_POLICY_NAME = "tau2-hyper-scoring"

WORKSPACE = "/workspace"
RUNTIME_HOME = "/runtime-home"
# The host keeps the kit's Git metadata (the baseline commit the scoring reads
# the Developer's delta against); the pod gets a copy at the start and its own
# later commits stay in the pod.
_HOST_ONLY = (".git",)

# Run a command under an environment read from a file below the runtime home,
# so that a token is in no argument list on the host or in the pod.
_ENV_EXEC = (
    "import json,os,sys;"
    "os.environ.update(json.load(open(sys.argv[1])));"
    "os.execvp(sys.argv[2],sys.argv[2:])"
)
_WRITE_PRIVATE_FILE = (
    "import os,sys;"
    "p=sys.argv[1];"
    "os.makedirs(os.path.dirname(p),mode=0o700,exist_ok=True);"
    "fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600);"
    "os.write(fd,sys.stdin.buffer.read());"
    "os.close(fd)"
)
# The pod's side of the callback relay. The callback directory is a folder both
# sides see under Docker; here it is a volume of the pod, and this loop carries
# each request file out on its standard output and each response back in from
# its standard input. A request is renamed, not removed, so that a relay that
# starts again after a dropped stream sends what was never answered.
_RELAY = r"""
import json, os, sys, threading, time
d = sys.argv[1]
def answers():
    for line in sys.stdin:
        try:
            m = json.loads(line)
            tmp = os.path.join(d, ".response-%s.tmp" % m["id"])
            with open(tmp, "w") as f:
                f.write(m["body"])
            os.replace(tmp, os.path.join(d, "response-%s.json" % m["id"]))
            try:
                os.unlink(os.path.join(d, "relayed-%s.json" % m["id"]))
            except OSError:
                pass
        except Exception:
            pass
    os._exit(0)
threading.Thread(target=answers, daemon=True).start()
def send(name):
    rid = name[len("relayed-"):-len(".json")]
    try:
        with open(os.path.join(d, name)) as f:
            body = f.read()
    except OSError:
        return
    sys.stdout.write(json.dumps({"id": rid, "body": body}) + "\n")
    sys.stdout.flush()
for name in sorted(os.listdir(d)):
    if name.startswith("relayed-") and name.endswith(".json"):
        send(name)
beat = time.monotonic()
while True:
    for name in sorted(os.listdir(d)):
        if name.startswith("request-") and name.endswith(".json"):
            relayed = "relayed-" + name[len("request-"):]
            try:
                os.replace(os.path.join(d, name), os.path.join(d, relayed))
            except OSError:
                continue
            send(relayed)
    if time.monotonic() - beat > 15:
        sys.stdout.write("{}\n")
        sys.stdout.flush()
        beat = time.monotonic()
    time.sleep(0.05)
"""
_USAGE_PROBE = "cat /sys/fs/cgroup/memory.peak 2>/dev/null || echo; cat /sys/fs/cgroup/cpu.stat"


def sandbox_backend() -> str:
    """Which way a build's containers start: ``docker`` (the default) or ``kubernetes``."""
    value = (os.environ.get(BACKEND_ENV) or DOCKER_BACKEND).strip().lower()
    if value not in (DOCKER_BACKEND, KUBE_BACKEND):
        raise ValueError(
            f"{BACKEND_ENV} must be {DOCKER_BACKEND!r} or {KUBE_BACKEND!r}, not {value!r}"
        )
    return value


@dataclass(frozen=True)
class KubeSettings:
    """Where the pods go, read from the environment once per runtime."""

    cli: str = "kubectl"
    namespace: Optional[str] = None
    context: Optional[str] = None
    # The construction image as the cluster pulls it. A Docker image name is a
    # name in the machine's daemon; the cluster needs a registry's.
    image: Optional[str] = None
    ready_timeout_seconds: int = 300
    # A file that gets one JSON line per pod when it ends: its peak memory and
    # the CPU seconds it used, from the pod's own counters.
    usage_log: Optional[str] = None

    @classmethod
    def from_environment(cls) -> "KubeSettings":
        return cls(
            cli=os.environ.get("TAU2_KUBE_CLI") or "kubectl",
            namespace=os.environ.get("TAU2_KUBE_NAMESPACE") or None,
            context=os.environ.get("TAU2_KUBE_CONTEXT") or None,
            image=os.environ.get("TAU2_KUBE_IMAGE") or None,
            ready_timeout_seconds=int(
                os.environ.get("TAU2_KUBE_READY_TIMEOUT_SECONDS") or 300
            ),
            usage_log=os.environ.get("TAU2_KUBE_USAGE_LOG") or None,
        )

    def base(self) -> list[str]:
        """The command line up to the verb."""
        command = [self.cli]
        if self.context:
            command.extend(["--context", self.context])
        if self.namespace:
            command.extend(["-n", self.namespace])
        return command

    def exec_argv(
        self, pod: str, command: list[str], *, stdin: bool = False
    ) -> list[str]:
        """``exec`` of a command in a pod's one container."""
        return [
            *self.base(),
            "exec",
            *(["-i"] if stdin else []),
            pod,
            "--",
            *command,
        ]


def kube(
    settings: KubeSettings,
    args: list[str],
    *,
    input: Optional[str] = None,
    timeout: float = 60,
) -> subprocess.CompletedProcess:
    """Run one control command against the cluster with normalized failures."""
    command = [*settings.base(), *args]
    try:
        return subprocess.run(
            command,
            input=input,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"{settings.cli} not found for the Kubernetes sandbox backend"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Timed out running {settings.cli} {args[0] if args else ''}"
        ) from exc


def _create(settings: KubeSettings, manifest: dict, *, apply: bool = False) -> None:
    result = kube(
        settings,
        ["apply" if apply else "create", "-f", "-"],
        input=json.dumps(manifest),
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        kind = manifest.get("kind")
        name = manifest.get("metadata", {}).get("name")
        raise RuntimeError(f"Failed to create {kind} {name}: {detail}")


def _wait_ready(settings: KubeSettings, pod: str) -> None:
    result = kube(
        settings,
        [
            "wait",
            "--for=condition=Ready",
            f"pod/{pod}",
            f"--timeout={settings.ready_timeout_seconds}s",
        ],
        timeout=settings.ready_timeout_seconds + 30,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        described = kube(settings, ["describe", f"pod/{pod}"], timeout=30)
        raise RuntimeError(
            f"Pod {pod} did not become ready: {detail}\n{described.stdout[-2000:]}"
        )


def _jsonpath(settings: KubeSettings, resource: str, path: str) -> str:
    result = kube(settings, ["get", resource, "-o", f"jsonpath={path}"], timeout=30)
    return result.stdout.strip() if result.returncode == 0 else ""


def _memory_quantity(value: Optional[str]) -> Optional[str]:
    """A Docker memory size (``4g``, ``512m``) as a Kubernetes quantity."""
    if not value:
        return None
    text = str(value).strip()
    units = {"k": "Ki", "m": "Mi", "g": "Gi"}
    lowered = text.lower().removesuffix("b")
    if lowered and lowered[-1] in units and lowered[:-1].replace(".", "", 1).isdigit():
        return lowered[:-1] + units[lowered[-1]]
    return text


# What a pod asks the scheduler for. Docker's `--memory` and `--cpus` are ceilings:
# eight scoring containers capped at 2 CPUs each run side by side on a laptop. As
# requests they would each reserve that much, and a node (or a project's quota)
# takes two of them. So the task's Docker settings become limits only, and the
# request is what a pod of that kind was measured to use (one build of a retail
# task on OpenShift Local, 2026-11-11), never more than its limit.
REQUESTS = {
    ROLE_SANDBOX: {"cpu": "250m", "memory": "768Mi"},
    ROLE_GATEWAY: {"cpu": "50m", "memory": "64Mi"},
    ROLE_SCORING: {"cpu": "100m", "memory": "256Mi"},
}


def _resources(role: str, memory: Optional[str] = None, cpus: Optional[str] = None) -> dict:
    """Requests by the pod's kind; limits from the task's Docker settings, when it has any."""
    resources: dict[str, Any] = {"requests": dict(REQUESTS[role])}
    limits = {}
    if memory:
        limits["memory"] = _memory_quantity(memory)
    if cpus:
        limits["cpu"] = str(cpus)
    if limits:
        resources["limits"] = limits
    return resources


def _container_security(*, read_only_root: bool) -> dict:
    return {
        "allowPrivilegeEscalation": False,
        "capabilities": {"drop": ["ALL"]},
        "readOnlyRootFilesystem": read_only_root,
    }


# A pod whose network rule lets no name service through gets a resolver that
# answers at once. The rule drops a lookup without a word, and a library that
# looks a name up at import (a price sheet, a version check) would wait out the
# resolver's whole timeout, about twenty seconds, where Docker's `--network
# none` fails at once. Nothing listens on the pod's own port 53, so the lookup
# is refused there. The sandbox finds the model gateway by its hosts line.
_NO_NAME_SERVICE = {
    "dnsPolicy": "None",
    "dnsConfig": {
        "nameservers": ["127.0.0.1"],
        "options": [{"name": "timeout", "value": "1"}, {"name": "attempts", "value": "1"}],
    },
}


def _pod_base(name: str, labels: dict, spec: dict) -> dict:
    """What every pod of this path shares: nothing privileged, no user id named."""
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": name, "labels": labels},
        "spec": {
            "restartPolicy": "Never",
            "automountServiceAccountToken": False,
            "enableServiceLinks": False,
            "terminationGracePeriodSeconds": 1,
            "securityContext": {
                "runAsNonRoot": True,
                "seccompProfile": {"type": "RuntimeDefault"},
            },
            **spec,
        },
    }


def sandbox_pod_manifest(
    *,
    name: str,
    image: str,
    env: dict[str, str],
    gateway_ip: str,
    mount_targets: tuple[str, ...] = (),
    memory: Optional[str] = None,
    cpus: Optional[str] = None,
) -> dict:
    """The sandbox: a pod that sleeps, with the kit and the runtime home as volumes."""
    environment = {
        "HOME": RUNTIME_HOME,
        "XDG_CONFIG_HOME": f"{RUNTIME_HOME}/.config",
        "XDG_CACHE_HOME": f"{RUNTIME_HOME}/.cache",
        "PYTHONPATH": f"{WORKSPACE}:/opt/tau2/src",
        **env,
    }
    volumes = [
        {"name": "workspace", "emptyDir": {}},
        {"name": "runtime-home", "emptyDir": {}},
    ]
    mounts = [
        {"name": "workspace", "mountPath": WORKSPACE},
        {"name": "runtime-home", "mountPath": RUNTIME_HOME},
    ]
    for index, target in enumerate(mount_targets):
        volumes.append({"name": f"mount-{index}", "emptyDir": {}})
        mounts.append({"name": f"mount-{index}", "mountPath": target})
    container: dict[str, Any] = {
        "name": "sandbox",
        "image": image,
        "command": ["sleep", "infinity"],
        "workingDir": WORKSPACE,
        "env": [{"name": k, "value": v} for k, v in sorted(environment.items())],
        "securityContext": _container_security(read_only_root=False),
        "volumeMounts": mounts,
    }
    container["resources"] = _resources(ROLE_SANDBOX, memory, cpus)
    return _pod_base(
        name,
        {LABEL_BUILD: name, LABEL_ROLE: ROLE_SANDBOX},
        {
            # The sandbox resolves the model gateway by this one line and has no
            # way to the cluster's name service.
            "hostAliases": [{"ip": gateway_ip, "hostnames": [MODEL_GATEWAY_HOST]}],
            **_NO_NAME_SERVICE,
            "containers": [container],
            "volumes": volumes,
        },
    )


def gateway_pod_manifest(
    *,
    name: str,
    build: str,
    image: str,
    secret_name: str,
) -> dict:
    """The model gateway: the only pod of a build with a way out of the cluster."""
    return _pod_base(
        name,
        {LABEL_BUILD: build, LABEL_ROLE: ROLE_GATEWAY},
        {
            "containers": [
                {
                    "name": "model-gateway",
                    "image": image,
                    "command": ["python", "-m", "tau2.hyper.sandbox.model_gateway"],
                    "envFrom": [{"secretRef": {"name": secret_name}}],
                    "ports": [{"containerPort": MODEL_GATEWAY_PORT}],
                    "resources": _resources(ROLE_GATEWAY),
                    "securityContext": _container_security(read_only_root=True),
                    "volumeMounts": [{"name": "tmp", "mountPath": "/tmp"}],
                }
            ],
            "volumes": [{"name": "tmp", "emptyDir": {}}],
        },
    )


def gateway_service_manifest(*, name: str, build: str) -> dict:
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": name, "labels": {LABEL_BUILD: build}},
        "spec": {
            "selector": {LABEL_BUILD: build, LABEL_ROLE: ROLE_GATEWAY},
            "ports": [{"port": MODEL_GATEWAY_PORT, "targetPort": MODEL_GATEWAY_PORT}],
        },
    }


def network_policy_manifests(*, build: str) -> list[dict]:
    """The internal network of one build: sandbox to model gateway, and no more."""
    port = [{"protocol": "TCP", "port": MODEL_GATEWAY_PORT}]
    sandbox = {LABEL_BUILD: build, LABEL_ROLE: ROLE_SANDBOX}
    gateway = {LABEL_BUILD: build, LABEL_ROLE: ROLE_GATEWAY}
    return [
        {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {"name": f"{build}-sandbox", "labels": {LABEL_BUILD: build}},
            "spec": {
                "podSelector": {"matchLabels": sandbox},
                "policyTypes": ["Ingress", "Egress"],
                "ingress": [],
                "egress": [
                    {"to": [{"podSelector": {"matchLabels": gateway}}], "ports": port}
                ],
            },
        },
        {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {
                "name": f"{build}-model-gateway",
                "labels": {LABEL_BUILD: build},
            },
            "spec": {
                "podSelector": {"matchLabels": gateway},
                "policyTypes": ["Ingress"],
                "ingress": [
                    {"from": [{"podSelector": {"matchLabels": sandbox}}], "ports": port}
                ],
            },
        },
    ]


def scoring_policy_manifest() -> dict:
    """No way in and no way out for every scoring pod of the namespace."""
    return {
        "apiVersion": "networking.k8s.io/v1",
        "kind": "NetworkPolicy",
        "metadata": {"name": SCORING_POLICY_NAME},
        "spec": {
            "podSelector": {"matchLabels": {LABEL_ROLE: ROLE_SCORING}},
            "policyTypes": ["Ingress", "Egress"],
            "ingress": [],
            "egress": [],
        },
    }


def scoring_pod_manifest(
    *,
    name: str,
    image: str,
    memory: Optional[str] = None,
    cpus: Optional[str] = None,
) -> dict:
    """The scoring: a pod that sleeps with a read-only root; the candidate runs in it by exec."""
    container: dict[str, Any] = {
        "name": "candidate",
        "image": image,
        "command": ["sleep", "infinity"],
        "workingDir": WORKSPACE,
        "env": [{"name": "HOME", "value": "/tmp"}],
        "securityContext": _container_security(read_only_root=True),
        "volumeMounts": [
            {"name": "workspace", "mountPath": WORKSPACE},
            {"name": "tmp", "mountPath": "/tmp"},
        ],
    }
    container["resources"] = _resources(ROLE_SCORING, memory, cpus)
    return _pod_base(
        name,
        {LABEL_ROLE: ROLE_SCORING},
        {
            **_NO_NAME_SERVICE,
            "containers": [container],
            "volumes": [
                {"name": "workspace", "emptyDir": {}},
                {"name": "tmp", "emptyDir": {"sizeLimit": "512Mi"}},
            ],
        },
    )


# --- the kit, copied in and out ------------------------------------------------


def _tree_state(root: Path) -> dict[str, tuple[int, int]]:
    """Every file below ``root`` (but the host's own) by its time and size."""
    state: dict[str, tuple[int, int]] = {}
    for current, dirs, names in os.walk(root, followlinks=False):
        here = Path(current)
        if here == root:
            dirs[:] = [d for d in dirs if d not in _HOST_ONLY]
        for name in names:
            path = here / name
            try:
                stat = path.lstat()
            except OSError:
                continue
            state[path.relative_to(root).as_posix()] = (stat.st_mtime_ns, stat.st_size)
    return state


def _tar_of(root: Path, paths: Optional[list[str]] = None) -> bytes:
    """A tar of ``root`` (or of the named files below it), paths relative to it."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        if paths is None:
            for entry in sorted(root.iterdir()):
                archive.add(entry, arcname=entry.name)
        else:
            for relative in paths:
                archive.add(root / relative, arcname=relative, recursive=False)
    return buffer.getvalue()


def _safe_member(member: tarfile.TarInfo) -> bool:
    """A file, a folder, or a link that stays where it is: nothing else leaves the pod.

    The tar comes out of the sandbox, which the Developer controls. A name or
    a link target that is absolute or climbs with ``..`` is left out, and so
    is a hard link, a device or a pipe.
    """
    name = PurePosixPath(member.name)
    if name.is_absolute() or ".." in name.parts:
        return False
    if member.isfile() or member.isdir():
        return True
    if member.issym():
        target = PurePosixPath(member.linkname)
        return not target.is_absolute() and ".." not in target.parts
    return False


def _mirror_tar_into(data: bytes, destination: Path) -> None:
    """Make ``destination`` what the tar holds, keeping the host's own entries.

    Each member passes ``_safe_member`` and then Python's ``data`` filter, one
    at a time, so one refused member does not stop the copy. (The filter alone
    is not enough: with ``errorlevel = 0`` Python 3.12.9 unpacks the member it
    just refused.)
    """
    with tempfile.TemporaryDirectory(
        dir=destination.parent, prefix=".kit-pull-"
    ) as staging_name:
        staging = Path(staging_name)
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            for member in archive:
                if not _safe_member(member):
                    logger.warning(f"Left in the sandbox, not copied out: {member.name!r}")
                    continue
                try:
                    archive.extract(member, staging, filter="data")
                except (tarfile.TarError, OSError) as exc:
                    logger.warning(f"Not copied out of the sandbox: {member.name!r}: {exc}")
        for entry in destination.iterdir():
            if entry.name in _HOST_ONLY:
                continue
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink()
        for entry in staging.iterdir():
            if entry.name in _HOST_ONLY:
                continue
            shutil.move(str(entry), destination / entry.name)


def _run_bytes(
    command: list[str], *, input: Optional[bytes] = None, timeout: float = 600
) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            command, input=input, capture_output=True, timeout=timeout, check=False
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"{command[0]} not found for the Kubernetes backend") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Timed out copying a kit ({command[0]} exec)") from exc


def copy_into_pod(
    settings: KubeSettings, pod: str, data: bytes, *, then: str = ""
) -> None:
    """Unpack a tar below the pod's ``/workspace``; ``then`` is a shell line run after it."""
    script = f"tar xf - -C {WORKSPACE}" + (f" && {then}" if then else "")
    result = _run_bytes(
        settings.exec_argv(pod, ["sh", "-c", script], stdin=True), input=data
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"Failed to copy the kit into pod {pod}: {detail}")


def read_pod_usage(settings: KubeSettings, pod: str) -> Optional[dict]:
    """The pod's own counters: peak memory, and the CPU seconds it used."""
    try:
        result = kube(
            settings,
            ["exec", pod, "--", "sh", "-c", _USAGE_PROBE],
            timeout=20,
        )
    except RuntimeError:
        return None
    if result.returncode != 0:
        return None
    lines = result.stdout.splitlines()
    usage: dict[str, Any] = {}
    if lines and lines[0].strip().isdigit():
        usage["memory_peak_bytes"] = int(lines[0].strip())
    for line in lines[1:]:
        parts = line.split()
        if len(parts) == 2 and parts[0] == "usage_usec" and parts[1].isdigit():
            usage["cpu_seconds"] = int(parts[1]) / 1_000_000
    return usage or None


def log_pod_usage(
    settings: KubeSettings, *, pod: str, role: str, usage: Optional[dict], seconds: float
) -> None:
    if not settings.usage_log or usage is None:
        return
    line = json.dumps(
        {"pod": pod, "role": role, "lifetime_seconds": round(seconds, 1), **usage}
    )
    try:
        with open(settings.usage_log, "a") as handle:
            handle.write(line + "\n")
    except OSError as exc:
        logger.warning(f"Could not write the pod usage log: {exc}")


def delete_resources(settings: KubeSettings, resources: list[str]) -> None:
    """Remove what a build made; a thing that is already gone is not an error."""
    if not resources:
        return
    try:
        result = kube(
            settings,
            [
                "delete",
                *resources,
                "--ignore-not-found",
                "--grace-period=0",
                "--force",
                "--wait=false",
            ],
            timeout=60,
        )
    except RuntimeError as exc:
        logger.warning(f"Failed to remove {', '.join(resources)}: {exc}")
        return
    if result.returncode != 0:
        logger.warning(
            f"Failed to remove {', '.join(resources)}: {result.stderr.strip()[-500:]}"
        )


# --- the scoring pod -------------------------------------------------------------

_scoring_policy_lock = threading.Lock()
_scoring_policy_applied: set[tuple] = set()


def _ensure_scoring_policy(settings: KubeSettings) -> None:
    key = (settings.cli, settings.context, settings.namespace)
    with _scoring_policy_lock:
        if key in _scoring_policy_applied:
            return
        _create(settings, scoring_policy_manifest(), apply=True)
        _scoring_policy_applied.add(key)


@dataclass
class CandidatePod:
    """One scoring pod, and the command whose standard input and output are its pipe."""

    settings: KubeSettings
    name: str
    started_at: float

    def exec_argv(self, command: list[str]) -> list[str]:
        return self.settings.exec_argv(self.name, command, stdin=True)

    def delete(self) -> None:
        if self.settings.usage_log:
            log_pod_usage(
                self.settings,
                pod=self.name,
                role=ROLE_SCORING,
                usage=read_pod_usage(self.settings, self.name),
                seconds=time.monotonic() - self.started_at,
            )
        delete_resources(self.settings, [f"pod/{self.name}"])


def start_candidate_pod(
    *,
    image: str,
    kit_path: Path,
    memory: Optional[str] = None,
    cpus: Optional[str] = None,
    settings: Optional[KubeSettings] = None,
) -> CandidatePod:
    """Start a scoring pod and give it the kit, read-only.

    The network rule exists before the pod does. The kit is copied without its
    Git metadata and then loses its write bits, so a candidate that writes
    below ``/workspace`` fails as it does on Docker's read-only mount.
    """
    settings = settings or KubeSettings.from_environment()
    _ensure_scoring_policy(settings)
    name = f"tau2-candidate-{uuid.uuid4().hex[:12]}"
    pod = CandidatePod(settings=settings, name=name, started_at=time.monotonic())
    try:
        _create(
            settings,
            scoring_pod_manifest(
                name=name, image=settings.image or image, memory=memory, cpus=cpus
            ),
        )
        _wait_ready(settings, name)
        kit_path = Path(kit_path)
        entries = [
            entry.name for entry in sorted(kit_path.iterdir()) if entry.name not in _HOST_ONLY
        ]
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            for entry in entries:
                archive.add(kit_path / entry, arcname=entry)
        copy_into_pod(
            settings,
            name,
            buffer.getvalue(),
            # The volume's own folder belongs to the cluster; everything below it loses its write bits.
            then=f"find {WORKSPACE} -mindepth 1 -exec chmod a-w {{}} +",
        )
    except BaseException:
        delete_resources(settings, [f"pod/{name}"])
        raise
    return pod


# --- the sandbox and its model gateway --------------------------------------------


class KubeSandboxRuntime(NativeSandboxRuntime):
    """The construction sandbox of one build as pods."""

    def __init__(
        self,
        kit_dir: Path,
        *,
        config: Optional[NativeRuntimeConfig] = None,
        env: Optional[dict[str, str]] = None,
        extra_mounts: tuple[NativeMount, ...] = (),
        settings: Optional[KubeSettings] = None,
    ):
        super().__init__(kit_dir, config=config, env=env, extra_mounts=extra_mounts)
        self.settings = settings or KubeSettings.from_environment()
        self.image = self.settings.image or self.config.image
        self._sync_lock = threading.RLock()
        self._close_lock = threading.Lock()
        self._pushed_state: dict[str, tuple[int, int]] = {}
        self._relays: list[_CallbackRelay] = []
        self._exec_env_count = 0
        self._created: list[str] = []
        self._scc: dict[str, str] = {}
        self._usage: dict[str, Any] = {}
        self._started_at = 0.0

    # -- small helpers --

    def _exec(
        self,
        command: list[str],
        *,
        pod: Optional[str] = None,
        input: Optional[str] = None,
        timeout: float = 60,
    ) -> subprocess.CompletedProcess:
        return kube(
            self.settings,
            ["exec", *(["-i"] if input is not None else []), pod or self.container_name, "--", *command],
            input=input,
            timeout=timeout,
        )

    def _exec_probe(self, command: list[str], *, timeout: int = 15):
        return self._exec(command, timeout=timeout)

    def _make(self, manifest: dict) -> None:
        _create(self.settings, manifest)
        self._created.append(
            f"{manifest['kind'].lower()}/{manifest['metadata']['name']}"
        )

    def _write_exec_env(self, env: dict[str, str]) -> str:
        """Put an environment below the runtime home; the path names it for ``_ENV_EXEC``."""
        self._exec_env_count += 1
        path = f"{RUNTIME_HOME}/.tau2/exec-env-{self._exec_env_count}.json"
        result = self._exec(
            ["python", "-I", "-c", _WRITE_PRIVATE_FILE, path],
            input=json.dumps(env),
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to write an exec environment: {result.stderr}")
        return path

    def _env_exec_argv(self, env_path: str, command: list[str]) -> list[str]:
        return self.settings.exec_argv(
            self.container_name,
            ["python", "-I", "-c", _ENV_EXEC, env_path, *command],
            stdin=True,
        )

    # -- the kit --

    def _push_workspace(self, *, changed_only: bool) -> None:
        """Copy the host's kit into the pod: all of it, or what the host wrote since the last pull."""
        with self._sync_lock:
            if changed_only:
                now = _tree_state(self.kit_dir)
                paths = sorted(
                    path for path, state in now.items() if self._pushed_state.get(path) != state
                )
                if not paths:
                    return
                data = _tar_of(self.kit_dir, paths)
            else:
                data = _tar_of(self.kit_dir)
            copy_into_pod(self.settings, self.container_name, data)
            self._pushed_state = _tree_state(self.kit_dir)

    def _pull_workspace(self) -> None:
        """Make the host's kit what the pod's ``/workspace`` is now."""
        with self._sync_lock:
            result = _run_bytes(
                self.settings.exec_argv(
                    self.container_name,
                    ["tar", "cf", "-", "-C", WORKSPACE, "--exclude=./.git", "."],
                )
            )
            # GNU tar ends with 1 when a file changed while it was read; the
            # archive is whole and the next pull has the newer bytes.
            if result.returncode not in (0, 1) or not result.stdout:
                detail = result.stderr.decode("utf-8", "replace").strip()
                raise RuntimeError(f"Failed to copy the kit out of the sandbox: {detail}")
            _mirror_tar_into(result.stdout, self.kit_dir)
            self._pushed_state = _tree_state(self.kit_dir)

    # -- lifecycle --

    def start(self) -> None:
        """Start the sandbox pod behind its network rules, and copy the kit in."""
        if not self.kit_dir.is_dir():
            raise ValueError(f"Kit directory does not exist: {self.kit_dir}")
        try:
            # The rules and the Service exist before any pod of the build does.
            for policy in network_policy_manifests(build=self.container_name):
                self._make(policy)
            self._make(
                gateway_service_manifest(
                    name=self.gateway_container_name, build=self.container_name
                )
            )
            self._network_started = True
            gateway_ip = _jsonpath(
                self.settings,
                f"service/{self.gateway_container_name}",
                "{.spec.clusterIP}",
            )
            if not gateway_ip or gateway_ip == "None":
                raise RuntimeError("The model gateway's Service has no cluster address")
            self._make(
                sandbox_pod_manifest(
                    name=self.container_name,
                    image=self.image,
                    env=self.env,
                    gateway_ip=gateway_ip,
                    mount_targets=tuple(mount.target for mount in self.extra_mounts),
                    memory=self.config.memory,
                    cpus=self.config.cpus,
                )
            )
            self._started = True
            self._started_at = time.monotonic()
            _wait_ready(self.settings, self.container_name)
            self._verify_runtime_contract()
            image_id = _jsonpath(
                self.settings,
                f"pod/{self.container_name}",
                "{.status.containerStatuses[0].imageID}",
            )
            if "sha256:" not in image_id:
                raise RuntimeError(
                    "Could not resolve the sandbox pod's immutable image ID: "
                    f"{image_id or 'no output'}"
                )
            self._image_id = image_id
            self._record_scc(self.container_name)
            self._push_workspace(changed_only=False)
            for mount in self.extra_mounts:
                if mount.read_only:
                    continue
                relay = _CallbackRelay(self, Path(mount.source), mount.target)
                relay.start()
                self._relays.append(relay)
        except BaseException:
            self.close(force=True)
            raise

    def _record_scc(self, pod: str) -> None:
        scc = _jsonpath(
            self.settings, f"pod/{pod}", r"{.metadata.annotations.openshift\.io/scc}"
        )
        if scc:
            self._scc[pod] = scc

    def start_model_gateway(self, spec: ModelGatewaySpec) -> None:
        """Start the only pod of the build with a way out, and wait for its health."""
        if not self._started:
            raise RuntimeError("Native sandbox must start before its model gateway")
        if self._gateway_started:
            raise RuntimeError("Native model gateway is already running")
        try:
            self._make(
                {
                    "apiVersion": "v1",
                    "kind": "Secret",
                    "metadata": {
                        "name": self.gateway_container_name,
                        "labels": {LABEL_BUILD: self.container_name},
                    },
                    "stringData": spec.sidecar_environment(),
                }
            )
            self._make(
                gateway_pod_manifest(
                    name=self.gateway_container_name,
                    build=self.container_name,
                    image=self.image,
                    secret_name=self.gateway_container_name,
                )
            )
            self._gateway_started = True
            _wait_ready(self.settings, self.gateway_container_name)
            self._record_scc(self.gateway_container_name)

            env_path = self._write_exec_env({"TAU2_MODEL_GATEWAY_TOKEN": spec.token})
            health_script = (
                "import os,urllib.request; "
                "r=urllib.request.Request("
                f"'http://{MODEL_GATEWAY_HOST}:{MODEL_GATEWAY_PORT}/health',"
                "headers={'Authorization':'Bearer '+"
                "os.environ['TAU2_MODEL_GATEWAY_TOKEN']}); "
                "urllib.request.urlopen(r,timeout=2).read()"
            )
            last_detail = "gateway did not become healthy"
            for _ in range(50):
                health = self._exec(
                    ["python", "-I", "-c", _ENV_EXEC, env_path, "python", "-c", health_script],
                    timeout=15,
                )
                if health.returncode == 0:
                    self._gateway_metadata = spec.metadata()
                    self._gateway_metadata.update(
                        {
                            "container_name": self.gateway_container_name,
                            "internal_host": MODEL_GATEWAY_HOST,
                            "internal_port": MODEL_GATEWAY_PORT,
                        }
                    )
                    return
                last_detail = (
                    health.stderr.strip() or health.stdout.strip() or last_detail
                )
                time.sleep(0.2)
            logs = kube(
                self.settings,
                ["logs", "--tail", "50", f"pod/{self.gateway_container_name}"],
                timeout=30,
            )
            log_detail = logs.stderr.strip() or logs.stdout.strip()
            raise RuntimeError(
                "Native model gateway failed its health check: "
                f"{last_detail[-1000:]}; logs={log_detail[-2000:]}"
            )
        except BaseException:
            self._remove_gateway(force=True)
            raise

    def run(
        self,
        command: list[str],
        *,
        timeout_seconds: float,
        stdin_text: Optional[str] = None,
        on_event: Optional[Callable[[NativeProcessEvent], None]] = None,
        cancel_event: Optional[threading.Event] = None,
        env: Optional[dict[str, str]] = None,
        cancel_grace_seconds: float = 0.0,
        before_close: Optional[Callable[[str], None]] = None,
    ) -> NativeProcessResult:
        """Run the harness by exec; a cancel or the timeout removes the pods."""
        if not self._started:
            self.start()
        env_path = self._write_exec_env(dict(env or {}))
        return run_supervised_process(
            self._env_exec_argv(env_path, command),
            timeout_seconds=timeout_seconds,
            stdin_text=stdin_text,
            on_event=on_event,
            on_timeout=self.close,
            cancel_event=cancel_event,
            on_cancel=self.close,
            cancel_grace_seconds=cancel_grace_seconds,
            before_stop=before_close,
        )

    def write_runtime_file(self, path: str, contents: str) -> None:
        """Write a harness config below `/runtime-home` without touching kit."""
        if not path.startswith(f"{RUNTIME_HOME}/") or ".." in Path(path).parts:
            raise ValueError("Runtime files must stay below /runtime-home")
        result = self._exec(
            [
                "bash",
                "-c",
                'umask 077; mkdir -p -- "$1"; cat > "$2"',
                "tau2-write-runtime-file",
                str(Path(path).parent),
                path,
            ],
            input=contents,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to write native runtime file: {result.stderr}")

    def read_runtime_tree(self, path: str, *, timeout: int = 120) -> bytes:
        """A directory below `/runtime-home`, as a tar stream (what a harness kept)."""
        if not path.startswith(f"{RUNTIME_HOME}/") or ".." in Path(path).parts:
            raise ValueError("Runtime trees must stay below /runtime-home")
        result = _run_bytes(
            self.settings.exec_argv(
                self.container_name, ["tar", "cf", "-", "-C", path, "."]
            ),
            timeout=timeout,
        )
        if result.returncode != 0:
            detail = result.stderr.decode("utf-8", "replace").strip()
            raise RuntimeError(f"Failed to read {path} out of the native sandbox: {detail}")
        return result.stdout

    def runtime_metadata(self) -> dict:
        """Return non-secret, serializable runtime identity."""
        return {
            "backend": "kubernetes-native",
            "image": self.image,
            "runtime_contract_version": self._runtime_contract_version,
            "runtime_source_revision": self._runtime_source_revision,
            "container_name": self.container_name,
            "namespace": self.settings.namespace,
            "kit_mount": str(self.kit_dir),
            "kit_transport": "copied-in-and-out",
            "workdir": WORKSPACE,
            "runtime_home": RUNTIME_HOME,
            "network_profile": NATIVE_NETWORK_PROFILE,
            "network_policies": [
                f"{self.container_name}-sandbox",
                f"{self.container_name}-model-gateway",
            ],
            "image_id": self._image_id,
            "memory": self.config.memory,
            "cpus": self.config.cpus,
            "env_override_keys": sorted(self.env),
            "extra_mount_targets": [mount.target for mount in self.extra_mounts],
            "model_gateway": self._gateway_metadata,
            # Filled as the pods are admitted and as they end: the same objects
            # the build's record holds.
            "security_context_constraints": self._scc,
            "usage": self._usage,
        }

    def _remove_gateway(self, *, force: bool = False) -> None:
        if not force and not self._gateway_started:
            return
        delete_resources(
            self.settings,
            [f"pod/{self.gateway_container_name}", f"secret/{self.gateway_container_name}"],
        )
        self._gateway_started = False

    def close(self, *, force: bool = False) -> None:
        """Stop every process in the sandbox, copy the kit out, and remove the build's pods."""
        with self._close_lock:
            for relay in self._relays:
                relay.stop()
            self._relays = []
            if self._started:
                # Stop the Developer's processes first, so that the kit that is
                # copied out is the one the pod ends with. The pod's first
                # process (the sleep) is not reached by the signal.
                try:
                    self._exec(["sh", "-c", "kill -KILL -1"], timeout=15)
                except RuntimeError:
                    pass
                try:
                    self._pull_workspace()
                except Exception as exc:  # noqa: BLE001 - the last pull stands
                    logger.warning(
                        f"The kit was not copied out at the end; the copy of the "
                        f"last callback stands: {exc}"
                    )
                lifetime = time.monotonic() - self._started_at
                for role, pod in (
                    (ROLE_SANDBOX, self.container_name),
                    (ROLE_GATEWAY, self.gateway_container_name),
                ):
                    if role == ROLE_GATEWAY and not self._gateway_started:
                        continue
                    usage = read_pod_usage(self.settings, pod)
                    if usage is not None:
                        self._usage[role] = usage
                        log_pod_usage(
                            self.settings, pod=pod, role=role, usage=usage, seconds=lifetime
                        )
            self._remove_gateway(force=force)
            if force or self._started or self._network_started:
                delete_resources(
                    self.settings,
                    [
                        f"pod/{self.container_name}",
                        f"service/{self.gateway_container_name}",
                        f"networkpolicy/{self.container_name}-sandbox",
                        f"networkpolicy/{self.container_name}-model-gateway",
                    ],
                )
            self._started = False
            self._network_started = False


class _CallbackRelay:
    """Carry the callback files between the pod's volume and the host's folder.

    The broker on the host and the callback server in the pod keep their file
    protocol. Around each request the kit is copied out of the pod (a local
    test reads the workspace as it is) and what the host wrote is copied back
    in (the test's artifact, a report).
    """

    def __init__(self, runtime: KubeSandboxRuntime, host_dir: Path, pod_dir: str):
        self.runtime = runtime
        self.host_dir = host_dir
        self.pod_dir = pod_dir
        self._stop = threading.Event()
        self._process: Optional[subprocess.Popen] = None
        self._answers: dict[str, str] = {}
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()

    def _serve(self) -> None:
        argv = self.runtime.settings.exec_argv(
            self.runtime.container_name,
            ["python", "-u", "-I", "-c", _RELAY, self.pod_dir],
            stdin=True,
        )
        while not self._stop.is_set():
            try:
                process = subprocess.Popen(
                    argv,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    bufsize=1,
                )
            except OSError as exc:
                logger.warning(f"The callback relay did not start: {exc}")
                return
            self._process = process
            assert process.stdout is not None
            for line in process.stdout:
                if self._stop.is_set():
                    break
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue
                request_id = message.get("id")
                if not isinstance(request_id, str):
                    continue  # the relay's heartbeat
                if not request_id.isalnum() or len(request_id) > 64:
                    continue
                answer = self._answers.get(request_id)
                if answer is None:
                    answer = self._dispatch(request_id, str(message.get("body", "")))
                    if answer is None:
                        break
                    self._answers[request_id] = answer
                try:
                    assert process.stdin is not None
                    process.stdin.write(
                        json.dumps({"id": request_id, "body": answer}) + "\n"
                    )
                    process.stdin.flush()
                except (OSError, ValueError):
                    break
            if process.poll() is None:
                process.terminate()
            process.wait()
            if not self._stop.is_set():
                # The exec stream dropped: start again. The pod's side sends
                # what was not answered, and an answer made before is kept.
                time.sleep(0.5)

    def _dispatch(self, request_id: str, body: str) -> Optional[str]:
        """Hand one request to the host's broker and wait for its response."""
        try:
            self.runtime._pull_workspace()
        except Exception as exc:  # noqa: BLE001 - the broker answers on what the host has
            logger.warning(f"The kit was not copied out before a callback: {exc}")
        temporary = self.host_dir / f".request-{request_id}.tmp"
        temporary.write_text(body)
        temporary.replace(self.host_dir / f"request-{request_id}.json")
        response_path = self.host_dir / f"response-{request_id}.json"
        while not response_path.exists():
            if self._stop.wait(0.05):
                return None
        answer = response_path.read_text()
        response_path.unlink(missing_ok=True)
        try:
            self.runtime._push_workspace(changed_only=True)
        except Exception as exc:  # noqa: BLE001 - the response still goes back
            logger.warning(f"What the host wrote was not copied into the sandbox: {exc}")
        return answer

