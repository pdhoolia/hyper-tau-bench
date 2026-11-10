"""The pod path of the construction sandbox, checked with no cluster."""

import io
import json
import sys
import tarfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tau2.hyper.runtime_contract import CONSTRUCTION_RUNTIME_CONTRACT_VERSION
from tau2.hyper.sandbox import kube_runtime
from tau2.hyper.sandbox.kube_runtime import (
    KUBE_BACKEND,
    LABEL_BUILD,
    LABEL_ROLE,
    KubeSandboxRuntime,
    KubeSettings,
    _CallbackRelay,
    _memory_quantity,
    _mirror_tar_into,
    _tree_state,
    gateway_pod_manifest,
    gateway_service_manifest,
    network_policy_manifests,
    sandbox_backend,
    sandbox_pod_manifest,
    scoring_pod_manifest,
    scoring_policy_manifest,
)
from tau2.hyper.sandbox.model_gateway import (
    MODEL_GATEWAY_HOST,
    MODEL_GATEWAY_PORT,
    ModelGatewaySpec,
)
from tau2.hyper.sandbox.native_runtime import NativeSandboxRuntime
from tau2.hyper.sandbox.sealed_runner import CandidateProcess, SealedRunnerConfig

IMAGE = "registry.example/project/tau2-construction-runtime@sha256:abc"


def _walk(value):
    """Every dict below a manifest."""
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _assert_restricted(manifest: dict) -> None:
    """What OpenShift's restricted-v2 and the Kubernetes restricted profile ask of a pod."""
    spec = manifest["spec"]
    assert spec["securityContext"]["runAsNonRoot"] is True
    assert spec["securityContext"]["seccompProfile"] == {"type": "RuntimeDefault"}
    assert spec["automountServiceAccountToken"] is False
    for node in _walk(manifest):
        # No user id of the pod's own choosing, and nothing privileged.
        assert "runAsUser" not in node
        assert "runAsGroup" not in node
        assert "fsGroup" not in node
        assert "privileged" not in node
        assert "hostNetwork" not in node
        assert "hostPath" not in node
    for container in spec["containers"]:
        security = container["securityContext"]
        assert security["allowPrivilegeEscalation"] is False
        assert security["capabilities"] == {"drop": ["ALL"]}


def test_docker_is_the_default_and_the_switch_is_strict(monkeypatch):
    monkeypatch.delenv("TAU2_SANDBOX_BACKEND", raising=False)
    assert sandbox_backend() == "docker"

    monkeypatch.setenv("TAU2_SANDBOX_BACKEND", "kubernetes")
    assert sandbox_backend() == KUBE_BACKEND
    assert issubclass(KubeSandboxRuntime, NativeSandboxRuntime)

    monkeypatch.setenv("TAU2_SANDBOX_BACKEND", "podman")
    with pytest.raises(ValueError, match="TAU2_SANDBOX_BACKEND"):
        sandbox_backend()


def test_settings_read_the_environment(monkeypatch):
    monkeypatch.setenv("TAU2_KUBE_CLI", "oc")
    monkeypatch.setenv("TAU2_KUBE_NAMESPACE", "hyper-trial")
    monkeypatch.setenv("TAU2_KUBE_IMAGE", IMAGE)
    monkeypatch.delenv("TAU2_KUBE_CONTEXT", raising=False)
    settings = KubeSettings.from_environment()
    assert settings.base() == ["oc", "-n", "hyper-trial"]
    assert settings.image == IMAGE
    assert settings.exec_argv("pod-a", ["true"], stdin=True) == [
        "oc", "-n", "hyper-trial", "exec", "-i", "pod-a", "--", "true",
    ]


def test_sandbox_pod_is_restricted_and_resolves_only_the_gateway():
    manifest = sandbox_pod_manifest(
        name="tau2-claude-code-abc",
        image=IMAGE,
        env={"TAU2_CALLBACK_DIR": "/run/tau2-callback"},
        gateway_ip="10.217.4.20",
        mount_targets=("/run/tau2-callback",),
        memory="4g",
        cpus="2",
    )
    _assert_restricted(manifest)
    spec = manifest["spec"]
    assert manifest["metadata"]["labels"] == {
        LABEL_BUILD: "tau2-claude-code-abc",
        LABEL_ROLE: "sandbox",
    }
    assert spec["hostAliases"] == [
        {"ip": "10.217.4.20", "hostnames": [MODEL_GATEWAY_HOST]}
    ]
    # No name service is reachable, so a lookup is refused at once, not dropped.
    assert spec["dnsPolicy"] == "None"
    assert spec["dnsConfig"]["nameservers"] == ["127.0.0.1"]
    container = spec["containers"][0]
    assert container["command"] == ["sleep", "infinity"]
    assert container["workingDir"] == "/workspace"
    env = {item["name"]: item["value"] for item in container["env"]}
    assert env["HOME"] == "/runtime-home"
    assert env["PYTHONPATH"] == "/workspace:/opt/tau2/src"
    assert env["TAU2_CALLBACK_DIR"] == "/run/tau2-callback"
    assert not any("API_KEY" in name for name in env)
    mounts = {item["mountPath"] for item in container["volumeMounts"]}
    assert mounts == {"/workspace", "/runtime-home", "/run/tau2-callback"}
    assert all("emptyDir" in volume for volume in spec["volumes"])
    assert container["resources"] == {
        "limits": {"memory": "4Gi", "cpu": "2"},
        "requests": {"memory": "4Gi", "cpu": "2"},
    }


def test_gateway_pod_takes_its_keys_from_a_secret_only():
    manifest = gateway_pod_manifest(
        name="tau2-claude-code-abc-model-gateway",
        build="tau2-claude-code-abc",
        image=IMAGE,
        secret_name="tau2-claude-code-abc-model-gateway",
    )
    _assert_restricted(manifest)
    container = manifest["spec"]["containers"][0]
    assert container["command"] == ["python", "-m", "tau2.hyper.sandbox.model_gateway"]
    assert container["envFrom"] == [
        {"secretRef": {"name": "tau2-claude-code-abc-model-gateway"}}
    ]
    assert "env" not in container
    assert container["securityContext"]["readOnlyRootFilesystem"] is True
    # The gateway keeps the cluster's name service: it has to find its upstream.
    assert "dnsPolicy" not in manifest["spec"]
    service = gateway_service_manifest(
        name="tau2-claude-code-abc-model-gateway", build="tau2-claude-code-abc"
    )
    assert service["spec"]["selector"] == manifest["metadata"]["labels"]
    assert service["spec"]["ports"] == [
        {"port": MODEL_GATEWAY_PORT, "targetPort": MODEL_GATEWAY_PORT}
    ]


def test_network_rules_let_the_sandbox_reach_the_gateway_and_nothing_else():
    sandbox, gateway = network_policy_manifests(build="tau2-claude-code-abc")
    sandbox_labels = {LABEL_BUILD: "tau2-claude-code-abc", LABEL_ROLE: "sandbox"}
    gateway_labels = {LABEL_BUILD: "tau2-claude-code-abc", LABEL_ROLE: "model-gateway"}
    port = [{"protocol": "TCP", "port": MODEL_GATEWAY_PORT}]

    assert sandbox["spec"]["podSelector"] == {"matchLabels": sandbox_labels}
    assert sandbox["spec"]["policyTypes"] == ["Ingress", "Egress"]
    assert sandbox["spec"]["ingress"] == []
    assert sandbox["spec"]["egress"] == [
        {"to": [{"podSelector": {"matchLabels": gateway_labels}}], "ports": port}
    ]
    # The gateway takes calls from its own build's sandbox; its way out stays open.
    assert gateway["spec"]["podSelector"] == {"matchLabels": gateway_labels}
    assert gateway["spec"]["policyTypes"] == ["Ingress"]
    assert gateway["spec"]["ingress"] == [
        {"from": [{"podSelector": {"matchLabels": sandbox_labels}}], "ports": port}
    ]


def test_scoring_pod_has_no_network_and_a_read_only_root():
    policy = scoring_policy_manifest()
    assert policy["spec"]["podSelector"] == {"matchLabels": {LABEL_ROLE: "scoring"}}
    assert policy["spec"]["policyTypes"] == ["Ingress", "Egress"]
    assert policy["spec"]["ingress"] == [] and policy["spec"]["egress"] == []

    manifest = scoring_pod_manifest(name="tau2-candidate-abc", image=IMAGE)
    _assert_restricted(manifest)
    assert manifest["metadata"]["labels"] == {LABEL_ROLE: "scoring"}
    assert manifest["spec"]["dnsPolicy"] == "None"
    assert manifest["spec"]["dnsConfig"]["nameservers"] == ["127.0.0.1"]
    container = manifest["spec"]["containers"][0]
    assert container["securityContext"]["readOnlyRootFilesystem"] is True
    assert container["env"] == [{"name": "HOME", "value": "/tmp"}]
    assert "resources" not in container


@pytest.mark.parametrize(
    "docker,kube",
    [("4g", "4Gi"), ("512m", "512Mi"), ("2GB", "2Gi"), ("4Gi", "4Gi"), (None, None)],
)
def test_docker_memory_sizes_become_kubernetes_quantities(docker, kube):
    assert _memory_quantity(docker) == kube


def _tar(entries: dict[str, str], extra=None) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, text in entries.items():
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        for info in extra or []:
            archive.addfile(info)
    return buffer.getvalue()


def test_copy_out_mirrors_the_pod_and_keeps_the_hosts_git(tmp_path):
    kit = tmp_path / "kit"
    (kit / ".git").mkdir(parents=True)
    (kit / ".git" / "HEAD").write_text("baseline")
    (kit / "workspace").mkdir()
    (kit / "workspace" / "agent.py").write_text("old")
    (kit / "workspace" / "removed.py").write_text("gone in the pod")

    _mirror_tar_into(
        _tar({"./workspace/agent.py": "new", "./notes.md": "written in the pod"}), kit
    )

    assert (kit / "workspace" / "agent.py").read_text() == "new"
    assert not (kit / "workspace" / "removed.py").exists()
    assert (kit / "notes.md").read_text() == "written in the pod"
    assert (kit / ".git" / "HEAD").read_text() == "baseline"
    assert not list(tmp_path.glob(".kit-pull-*"))


def test_copy_out_leaves_out_what_would_leave_the_kit(tmp_path):
    kit = tmp_path / "kit"
    kit.mkdir()
    outside = tmp_path / "outside.txt"

    def link(name: str, target: str, kind=tarfile.SYMTYPE) -> tarfile.TarInfo:
        info = tarfile.TarInfo(name)
        info.type = kind
        info.linkname = target
        return info

    data = _tar(
        {
            "./ok.txt": "fine",
            "../outside.txt": "escaped",
            "./sub/../../outside.txt": "escaped",
            "./.git/config": "the pod's",
            "./sub/kept.txt": "kept",
        },
        extra=[
            link("./absolute", "/etc/passwd"),
            link("./climbs", "../outside.txt"),
            link("./hard", "ok.txt", tarfile.LNKTYPE),
            link("./sub/stays", "kept.txt"),
        ],
    )

    _mirror_tar_into(data, kit)

    assert (kit / "ok.txt").read_text() == "fine"
    assert not outside.exists()
    for name in ("absolute", "climbs", "hard"):
        assert not (kit / name).exists() and not (kit / name).is_symlink()
    assert not (kit / ".git").exists()
    # A link that stays inside its own folder comes out as it is.
    assert (kit / "sub" / "stays").is_symlink()
    assert (kit / "sub" / "stays").read_text() == "kept"


def test_tree_state_skips_the_hosts_git(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "index").write_text("x")
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "a.py").write_text("a")
    assert set(_tree_state(tmp_path)) == {"workspace/a.py"}


class _FakeCluster:
    """Stands in for the command line: records every call and answers the few that are read."""

    def __init__(self):
        self.calls: list[list[str]] = []
        self.created: list[dict] = []
        self.pod_workspace: bytes = b""

    def kube(self, settings, args, *, input=None, timeout=60):
        self.calls.append(list(args))
        stdout = ""
        if args[:2] in (["create", "-f"], ["apply", "-f"]):
            self.created.append(json.loads(input))
        elif args[0] == "get" and "clusterIP" in args[-1]:
            stdout = "10.217.4.20"
        elif args[0] == "get" and "imageID" in args[-1]:
            stdout = "registry.example/project/tau2@sha256:0123"
        elif args[0] == "get" and "scc" in args[-1]:
            stdout = "restricted-v2"
        elif args[0] == "exec" and "runtime_contract" in " ".join(args):
            stdout = json.dumps(
                {
                    "contract_version": CONSTRUCTION_RUNTIME_CONTRACT_VERSION,
                    "source_revision": "abc123",
                }
            )
        elif args[0] == "exec" and "memory.peak" in " ".join(args):
            stdout = "1048576\nusage_usec 2500000\nuser_usec 2000000\n"
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    def run_bytes(self, command, *, input=None, timeout=600):
        self.calls.append(["bytes", *command])
        if input is not None:
            self.pod_workspace = input
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        return SimpleNamespace(returncode=0, stdout=self.pod_workspace, stderr=b"")


@pytest.fixture
def cluster(monkeypatch):
    fake = _FakeCluster()
    monkeypatch.setattr(kube_runtime, "kube", fake.kube)
    monkeypatch.setattr(kube_runtime, "_run_bytes", fake.run_bytes)
    monkeypatch.setattr(kube_runtime, "_scoring_policy_applied", set())
    return fake


def _runtime(kit: Path, **kwargs) -> KubeSandboxRuntime:
    return KubeSandboxRuntime(
        kit, settings=KubeSettings(cli="oc", namespace="hyper-trial", image=IMAGE), **kwargs
    )


def test_start_makes_the_rules_before_the_pod_and_close_removes_everything(
    cluster, tmp_path
):
    kit = tmp_path / "kit"
    (kit / "workspace").mkdir(parents=True)
    (kit / "workspace" / "agent.py").write_text("agent")
    runtime = _runtime(kit, env={"TAU2_CALLBACK_TOKEN": "per-run"})

    runtime.start()

    kinds = [manifest["kind"] for manifest in cluster.created]
    assert kinds == ["NetworkPolicy", "NetworkPolicy", "Service", "Pod"]
    pod = cluster.created[-1]
    assert pod["spec"]["containers"][0]["image"] == IMAGE
    assert pod["spec"]["hostAliases"][0]["ip"] == "10.217.4.20"
    # The kit went in whole, by exec.
    with tarfile.open(fileobj=io.BytesIO(cluster.pod_workspace)) as archive:
        assert "workspace/agent.py" in archive.getnames()
    metadata = runtime.runtime_metadata()
    assert metadata["backend"] == "kubernetes-native"
    assert metadata["runtime_source_revision"] == "abc123"
    assert metadata["security_context_constraints"] == {
        runtime.container_name: "restricted-v2"
    }

    runtime.start_model_gateway(
        ModelGatewaySpec(
            provider="anthropic",
            model="claude-sonnet-4-5",
            token="scoped-token",
            upstream_api_key="raw-provider-key",
            expires_at=9999999999,
        )
    )
    secret, gateway = cluster.created[-2:]
    assert secret["kind"] == "Secret"
    assert secret["stringData"]["TAU2_MODEL_GATEWAY_UPSTREAM_KEY"] == "raw-provider-key"
    assert "raw-provider-key" not in json.dumps(gateway)
    # No token rides an argument list.
    assert not any("scoped-token" in " ".join(call) for call in cluster.calls)

    # The Developer changed a file and removed one; the pod's copy is what comes out.
    cluster.pod_workspace = _tar({"./workspace/agent.py": "built"})
    runtime.close()

    assert (kit / "workspace" / "agent.py").read_text() == "built"
    deletes = [call for call in cluster.calls if call[0] == "delete"]
    removed = {name for call in deletes for name in call[1:] if "/" in name}
    assert removed == {
        f"pod/{runtime.container_name}",
        f"pod/{runtime.gateway_container_name}",
        f"secret/{runtime.gateway_container_name}",
        f"service/{runtime.gateway_container_name}",
        f"networkpolicy/{runtime.container_name}-sandbox",
        f"networkpolicy/{runtime.container_name}-model-gateway",
    }
    # The counters were read while the pods were still there, into the record.
    assert metadata["usage"]["sandbox"] == {
        "memory_peak_bytes": 1048576,
        "cpu_seconds": 2.5,
    }
    kill = next(i for i, call in enumerate(cluster.calls) if "kill -KILL -1" in call)
    first_delete = next(i for i, call in enumerate(cluster.calls) if call[0] == "delete")
    assert kill < first_delete


def test_only_what_the_host_wrote_goes_back_into_the_pod(cluster, tmp_path):
    kit = tmp_path / "kit"
    (kit / "workspace").mkdir(parents=True)
    (kit / "workspace" / "agent.py").write_text("agent")
    runtime = _runtime(kit)
    runtime._push_workspace(changed_only=False)

    cluster.pod_workspace = b""
    runtime._push_workspace(changed_only=True)
    assert cluster.pod_workspace == b""

    (kit / "simulations").mkdir()
    (kit / "simulations" / "local_run_1.json").write_text("{}")
    runtime._push_workspace(changed_only=True)
    with tarfile.open(fileobj=io.BytesIO(cluster.pod_workspace)) as archive:
        assert archive.getnames() == ["simulations/local_run_1.json"]


def test_scoring_runs_in_a_pod_and_the_pod_goes_with_it(cluster, monkeypatch, tmp_path):
    monkeypatch.setenv("TAU2_SANDBOX_BACKEND", "kubernetes")
    monkeypatch.setenv("TAU2_KUBE_CLI", "oc")
    monkeypatch.setenv("TAU2_KUBE_IMAGE", IMAGE)
    monkeypatch.delenv("TAU2_KUBE_NAMESPACE", raising=False)
    monkeypatch.delenv("TAU2_KUBE_USAGE_LOG", raising=False)
    (tmp_path / ".git").mkdir()
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "agent.py").write_text("agent")
    started = []

    class _Process:
        stdin = stdout = stderr = None

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

    def fake_popen(command, **kwargs):
        started.append(command)
        return _Process()

    monkeypatch.setattr("tau2.hyper.sandbox.sealed_runner.subprocess.Popen", fake_popen)
    monkeypatch.setattr(CandidateProcess, "_configure", lambda self: None)

    runner = CandidateProcess(SealedRunnerConfig(kit_path=tmp_path, domain="retail"))

    kinds = [manifest["kind"] for manifest in cluster.created]
    assert kinds == ["NetworkPolicy", "Pod"]  # the rule exists before the pod
    pod = cluster.created[-1]
    assert pod["metadata"]["labels"] == {LABEL_ROLE: "scoring"}
    assert pod["spec"]["containers"][0]["image"] == IMAGE
    name = pod["metadata"]["name"]
    assert started[0][:5] == ["oc", "exec", "-i", name, "--"]
    assert started[0][5:8] == ["python", "-I", "-c"]
    with tarfile.open(fileobj=io.BytesIO(cluster.pod_workspace)) as archive:
        names = archive.getnames()
    assert "workspace/agent.py" in names and not any(n.startswith(".git") for n in names)

    runner.close()
    assert ["delete", f"pod/{name}"] == [
        call for call in cluster.calls if call[0] == "delete"
    ][-1][:2]


def test_docker_scoring_command_is_what_it_was(monkeypatch, tmp_path):
    monkeypatch.delenv("TAU2_SANDBOX_BACKEND", raising=False)
    runner = CandidateProcess.__new__(CandidateProcess)
    runner.config = SealedRunnerConfig(kit_path=tmp_path, domain="retail")
    command = runner._build_command()
    assert command[:3] == ["docker", "run", "--rm"]
    assert "--user" in command and "65534:65534" in command
    assert command[-4:-1] == ["python", "-I", "-c"]
    assert "candidate_server import main" in command[-1]


def test_callback_relay_carries_a_request_out_and_its_response_back(tmp_path):
    """The pod's side of the relay runs here as a plain process on two folders."""
    pod_dir = tmp_path / "pod"
    host_dir = tmp_path / "host"
    pod_dir.mkdir()
    host_dir.mkdir()
    events: list[str] = []

    class _Settings:
        def exec_argv(self, pod, command, *, stdin=False):
            assert command[0] == "python"
            return [sys.executable, *command[1:]]

    runtime = SimpleNamespace(
        settings=_Settings(),
        container_name="tau2-claude-code-abc",
        _pull_workspace=lambda: events.append("pull"),
        _push_workspace=lambda *, changed_only: events.append("push"),
    )

    def broker() -> None:
        # The host's broker: answers each request file with a response file.
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            for request in host_dir.glob("request-*.json"):
                request_id = request.stem.removeprefix("request-")
                events.append("dispatch")
                body = json.loads(request.read_text())
                request.unlink()
                (host_dir / f"response-{request_id}.json").write_text(
                    json.dumps({"ok": True, "result": body["tool"]})
                )
                return
            time.sleep(0.02)

    threading.Thread(target=broker, daemon=True).start()
    relay = _CallbackRelay(runtime, host_dir, str(pod_dir))
    relay.start()
    try:
        (pod_dir / "request-abc123.json").write_text(json.dumps({"tool": "submit"}))
        response = pod_dir / "response-abc123.json"
        deadline = time.monotonic() + 20
        while not response.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert json.loads(response.read_text()) == {"ok": True, "result": "submit"}
        # Copied out before the broker saw it, and back in before the answer.
        assert events == ["pull", "dispatch", "push"]
        assert not list(pod_dir.glob("relayed-*.json"))
        # The same request again (a relay that started over) is answered from memory.
        assert relay._answers.keys() == {"abc123"}
    finally:
        relay.stop()
