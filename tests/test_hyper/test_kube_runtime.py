"""The pod path of the construction sandbox, checked with no cluster."""

import io
import json
import signal
import sys
import tarfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tau2.hyper.runtime_contract import CONSTRUCTION_RUNTIME_CONTRACT_VERSION
from tau2.hyper.sandbox import kube_runtime, kube_tools
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
    # Docker's ceilings are limits here, never requests: eight pods that each
    # reserved 2 CPUs would not fit on a node that runs them side by side.
    assert container["resources"] == {
        "limits": {"memory": "4Gi", "cpu": "2"},
        "requests": {"cpu": "250m", "memory": "768Mi"},
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
    assert container["resources"] == {"requests": {"cpu": "50m", "memory": "64Mi"}}
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
    assert container["resources"] == {"requests": {"cpu": "100m", "memory": "256Mi"}}
    capped = scoring_pod_manifest(
        name="tau2-candidate-abc", image=IMAGE, memory="4g", cpus="2"
    )["spec"]["containers"][0]["resources"]
    assert capped == {
        "requests": {"cpu": "100m", "memory": "256Mi"},
        "limits": {"memory": "4Gi", "cpu": "2"},
    }


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


# --- a build started from outside: marked, found, removed ------------------------

MARKED = KubeSettings(
    cli="oc",
    namespace="strata",
    image=IMAGE,
    labels=(("strata.runner", "openshift"), ("strata.build", "41-1")),
    owner_pod="strata-runner-7c9d",
    owner_uid="0b1f-uid",
    pull_secrets=("icr-image-pull",),
    resources={
        "scoring": {
            "requests": {"cpu": "50m", "memory": "128Mi"},
            "limits": {"cpu": "500m", "memory": "512Mi"},
        },
        "sandbox": {"limits": {}},
    },
)
OWNER = [{"apiVersion": "v1", "kind": "Pod", "name": "strata-runner-7c9d", "uid": "0b1f-uid"}]


def test_settings_read_who_started_the_build(monkeypatch):
    monkeypatch.setenv("TAU2_KUBE_LABELS", "strata.runner=openshift, strata.build=41-1")
    monkeypatch.setenv("TAU2_KUBE_OWNER_POD", "strata-runner-7c9d")
    monkeypatch.setenv("TAU2_KUBE_OWNER_UID", "0b1f-uid")
    monkeypatch.setenv("TAU2_KUBE_PULL_SECRETS", "icr-image-pull, other")
    monkeypatch.setenv("TAU2_KUBE_RESOURCES", json.dumps(MARKED.resources))
    monkeypatch.setenv("TAU2_KUBE_LOG_DIR", "/runs/pods")
    settings = KubeSettings.from_environment()
    assert settings.labels == MARKED.labels
    assert (settings.owner_pod, settings.owner_uid) == ("strata-runner-7c9d", "0b1f-uid")
    assert settings.pull_secrets == ("icr-image-pull", "other")
    assert settings.resources == MARKED.resources
    assert settings.log_dir == "/runs/pods"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("TAU2_KUBE_LABELS", "no-equals-sign"),
        ("TAU2_KUBE_LABELS", "key=not a label value"),
        # This path's own labels are not the caller's to set.
        ("TAU2_KUBE_LABELS", "tau2.hyper/role=scoring"),
        ("TAU2_KUBE_RESOURCES", "not json"),
        ("TAU2_KUBE_RESOURCES", '{"database": {}}'),
        ("TAU2_KUBE_RESOURCES", '{"scoring": {"cpu": "1"}}'),
        # Half an owner is none.
        ("TAU2_KUBE_OWNER_POD", "strata-runner-7c9d"),
    ],
)
def test_settings_refuse_what_the_cluster_would(monkeypatch, name, value):
    for other in ("TAU2_KUBE_LABELS", "TAU2_KUBE_RESOURCES", "TAU2_KUBE_OWNER_POD", "TAU2_KUBE_OWNER_UID"):
        monkeypatch.delenv(other, raising=False)
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name.rsplit("_", 1)[0]):
        KubeSettings.from_environment()


def test_a_pod_is_marked_and_keeps_its_own_labels():
    plain = scoring_pod_manifest(name="tau2-candidate-abc", image=IMAGE, memory="4g", cpus="2")
    marked = kube_runtime.stamped(MARKED, plain)
    # The manifest the functions return is not changed: the mark is a copy.
    assert plain["metadata"]["labels"] == {LABEL_ROLE: "scoring"}
    assert marked["metadata"]["labels"] == {
        "strata.runner": "openshift",
        "strata.build": "41-1",
        LABEL_ROLE: "scoring",
    }
    assert marked["metadata"]["ownerReferences"] == OWNER
    assert marked["spec"]["imagePullSecrets"] == [{"name": "icr-image-pull"}]
    # The settings' numbers stand in for the measured requests and the task's ceilings.
    assert marked["spec"]["containers"][0]["resources"] == MARKED.resources["scoring"]
    _assert_restricted(marked)

    # An empty object takes the limits away; the requests it does not name stay.
    sandbox = kube_runtime.stamped(
        MARKED,
        sandbox_pod_manifest(name="b", image=IMAGE, env={}, gateway_ip="10.0.0.1", memory="4g", cpus="2"),
    )
    assert sandbox["spec"]["containers"][0]["resources"] == {
        "requests": {"cpu": "250m", "memory": "768Mi"}
    }
    # A kind the settings do not name keeps what the manifest has.
    gateway = kube_runtime.stamped(
        MARKED, gateway_pod_manifest(name="b-model-gateway", build="b", image=IMAGE, secret_name="s")
    )
    assert gateway["spec"]["containers"][0]["resources"] == {
        "requests": {"cpu": "50m", "memory": "64Mi"}
    }

    # What is not a pod gets the labels and the owner, and nothing of a pod's.
    service = kube_runtime.stamped(MARKED, gateway_service_manifest(name="b-model-gateway", build="b"))
    assert service["metadata"]["labels"]["strata.build"] == "41-1"
    assert service["metadata"]["ownerReferences"] == OWNER
    assert "imagePullSecrets" not in service["spec"]

    # No settings, no mark: the manifests are what they were.
    assert kube_runtime.stamped(KubeSettings(), plain) == plain


def test_everything_a_build_makes_is_marked_but_the_shared_rule(cluster, monkeypatch, tmp_path):
    kit = tmp_path / "kit"
    (kit / "workspace").mkdir(parents=True)
    runtime = KubeSandboxRuntime(kit, settings=MARKED)
    runtime.start()
    runtime.start_model_gateway(
        ModelGatewaySpec(
            provider="anthropic",
            model="claude-sonnet-4-5",
            token="scoped-token",
            upstream_api_key="raw-provider-key",
            expires_at=9999999999,
        )
    )
    kube_runtime.start_candidate_pod(image=IMAGE, kit_path=kit, settings=MARKED)

    kinds = [manifest["kind"] for manifest in cluster.created]
    assert kinds == [
        "NetworkPolicy", "NetworkPolicy", "Service", "Pod", "Secret", "Pod", "NetworkPolicy", "Pod",
    ]
    shared = cluster.created[-2]
    assert shared["metadata"]["name"] == "tau2-hyper-scoring"
    # Every build's scoring pods stand behind it, so no build owns it.
    assert "labels" not in shared["metadata"] and "ownerReferences" not in shared["metadata"]
    for manifest in cluster.created[:-2] + cluster.created[-1:]:
        assert manifest["metadata"]["labels"]["strata.build"] == "41-1", manifest["kind"]
        assert manifest["metadata"]["ownerReferences"] == OWNER, manifest["kind"]
        if manifest["kind"] == "Pod":
            assert manifest["spec"]["imagePullSecrets"] == [{"name": "icr-image-pull"}]


def test_a_pod_the_quota_refuses_waits_for_room(monkeypatch):
    answers = [
        SimpleNamespace(returncode=1, stdout="", stderr='pods "p" is forbidden: exceeded quota: compute, requested: pods=1, used: pods=20, limited: pods=20'),
        SimpleNamespace(returncode=1, stdout="", stderr='pods "p" is forbidden: exceeded quota: compute'),
        SimpleNamespace(returncode=0, stdout="", stderr=""),
    ]
    calls, naps = [], []
    monkeypatch.setattr(kube_runtime, "kube", lambda settings, args, **kw: (calls.append(args), answers.pop(0))[1])
    monkeypatch.setattr(kube_runtime.time, "sleep", naps.append)
    manifest = scoring_pod_manifest(name="p", image=IMAGE)

    kube_runtime._create(KubeSettings(), manifest)
    assert len(calls) == 3 and naps == [5.0, 5.0]

    # Any other refusal is passed on at once.
    answers[:] = [SimpleNamespace(returncode=1, stdout="", stderr="forbidden: unable to validate against any security context constraint")]
    with pytest.raises(RuntimeError, match="Failed to create Pod p: forbidden"):
        kube_runtime._create(KubeSettings(), manifest)

    # And the wait has an end: the time a pod gets to become ready.
    answers[:] = [SimpleNamespace(returncode=1, stdout="", stderr="exceeded quota: compute")] * 3
    clock = iter([0.0, 400.0])
    monkeypatch.setattr(kube_runtime.time, "monotonic", lambda: next(clock))
    with pytest.raises(RuntimeError, match="exceeded quota"):
        kube_runtime._create(KubeSettings(ready_timeout_seconds=300), manifest)


def test_what_goes_with_a_pod_is_kept(monkeypatch, tmp_path):
    settings = KubeSettings(log_dir=str(tmp_path / "pods"))
    said = {
        "memory.peak": "536870912\nusage_usec 4000000\nuser_usec 1\nlow 0\nhigh 0\nmax 3\noom 3\noom_kill 2\n",
        "logs": "gateway: 200 POST /v1/messages\n",
    }

    def fake(settings, args, *, input=None, timeout=60):
        if args[0] == "exec" and "memory.peak" in " ".join(args):
            return SimpleNamespace(returncode=0, stdout=said["memory.peak"], stderr="")
        if args[0] == "logs":
            return SimpleNamespace(returncode=0, stdout=said["logs"], stderr="")
        if args[0] == "get":
            return SimpleNamespace(returncode=0, stdout="Failed|Evicted|The node was low on resource: memory.||", stderr="")
        return SimpleNamespace(returncode=1, stdout="", stderr="pod is not running")

    monkeypatch.setattr(kube_runtime, "kube", fake)

    # A process killed at the pod's memory limit leaves the pod running: its counters say so.
    usage, ended = kube_runtime.pod_ending(settings, "tau2-candidate-abc")
    assert usage == {"memory_peak_bytes": 536870912, "cpu_seconds": 4.0, "oom_kills": 2}
    assert ended is None
    kube_runtime.log_pod_usage(settings, pod="tau2-candidate-abc", role="scoring", usage=usage, seconds=12.34)

    # A pod that is no longer running has no counters to read: the cluster's reason is kept.
    said["memory.peak"] = ""
    monkeypatch.setattr(kube_runtime, "read_pod_usage", lambda settings, pod: None)
    usage, ended = kube_runtime.pod_ending(settings, "tau2-candidate-def")
    assert usage is None
    assert ended == "Failed: Evicted, The node was low on resource: memory."
    kube_runtime.log_pod_usage(settings, pod="tau2-candidate-def", role="scoring", usage=usage, seconds=3, ended=ended)

    lines = [json.loads(line) for line in (tmp_path / "pods" / "pods.jsonl").read_text().splitlines()]
    assert lines == [
        {"pod": "tau2-candidate-abc", "role": "scoring", "lifetime_seconds": 12.3, "memory_peak_bytes": 536870912, "cpu_seconds": 4.0, "oom_kills": 2},
        {"pod": "tau2-candidate-def", "role": "scoring", "lifetime_seconds": 3, "ended": "Failed: Evicted, The node was low on resource: memory."},
    ]

    # The model gateway's log goes with its pod, so it is read first.
    kube_runtime.keep_pod_log(settings, "tau2-claude-code-abc-model-gateway")
    assert (tmp_path / "pods" / "tau2-claude-code-abc-model-gateway.log").read_text() == said["logs"]
    # No folder named, nothing kept and nothing asked of the cluster.
    monkeypatch.setattr(kube_runtime, "kube", lambda *a, **k: pytest.fail("the cluster was asked"))
    kube_runtime.keep_pod_log(KubeSettings(), "pod")


def test_sigterm_ends_a_run_as_an_interrupt_does():
    from tau2.cli import _end_on_sigterm

    before = signal.getsignal(signal.SIGTERM)
    try:
        _end_on_sigterm()
        assert signal.getsignal(signal.SIGTERM) is signal.default_int_handler
    finally:
        signal.signal(signal.SIGTERM, before)


# --- the two commands for whoever starts builds from outside ---------------------


class _ProbedCluster(_FakeCluster):
    """A cluster whose pods answer the probe's two scripts."""

    def __init__(self, *, enforces: bool, control_reaches: bool = True):
        super().__init__()
        self.enforces = enforces
        self.control_reaches = control_reaches

    def kube(self, settings, args, *, input=None, timeout=60):
        if args[0] == "exec" and kube_tools._CONNECT in args:
            self.calls.append(list(args))
            pod, targets = args[1], args[args.index(kube_tools._CONNECT) + 1 :]
            control = pod.endswith("-control")
            rules = any(m["kind"] == "NetworkPolicy" for m in self.created)
            reached = self.control_reaches if control else not (self.enforces and rules)
            entries = [
                {"target": t, "address": "10.217.5.9" if control else t.rpartition(":")[0], "port": int(t.rpartition(":")[2]), "reached": reached}
                for t in targets
            ]
            return SimpleNamespace(returncode=0, stdout=json.dumps(entries) + "\n", stderr="")
        if args[0] == "exec" and kube_tools._STARTS in args:
            self.calls.append(list(args))
            asked = args[args.index(kube_tools._STARTS) + 1 :]
            harnesses = {
                item.split("=")[0]: (
                    {"started": False, "detail": "the image has no `bob` command"}
                    if item.startswith("bob=")
                    else {"started": True, "version": "1.2.3"}
                )
                for item in asked
            }
            starts = {"uid": 1000650000, "gid": 0, "home": "/runtime-home", "home_writable": True, "harnesses": harnesses}
            return SimpleNamespace(returncode=0, stdout=json.dumps(starts) + "\n", stderr="")
        return super().kube(settings, args, input=input, timeout=timeout)


def _probe(monkeypatch, fake, **kwargs):
    monkeypatch.setattr(kube_runtime, "kube", fake.kube)
    return kube_tools.probe(
        MARKED,
        image=IMAGE,
        targets=["gateway.strata.svc:8080"],
        harnesses=["claude-code", "codex", "bob"],
        **kwargs,
    )


def test_probe_says_the_sandbox_is_isolated_where_the_rule_is_enforced(monkeypatch):
    fake = _ProbedCluster(enforces=True)
    result = _probe(monkeypatch, fake)

    # The build's own rules, then its Service, then the sandbox and the control beside it.
    assert [m["kind"] for m in fake.created] == ["NetworkPolicy", "NetworkPolicy", "Service", "Pod", "Pod"]
    sandbox, control = fake.created[-2:]
    assert sandbox["metadata"]["labels"][LABEL_ROLE] == "sandbox"
    assert control["metadata"]["labels"][LABEL_ROLE] == "probe-control"
    # The rules select the sandbox and never the control.
    for policy in fake.created[:2]:
        assert policy["spec"]["podSelector"]["matchLabels"][LABEL_ROLE] != "probe-control"
    # The probe's pods are marked as a build's are, so a sweep finds one that was left.
    for manifest in fake.created:
        assert manifest["metadata"]["labels"]["strata.runner"] == "openshift"
    _assert_restricted(control)

    assert result["isolated"] is True
    assert result["rules"] is True
    assert result["security_context_constraint"] == "restricted-v2"
    assert result["uid"] == 1000650000 and result["home_writable"] is True
    assert result["control"][0]["reached"] is True
    # The sandbox was given the number the control resolved, and its entry carries the name.
    connects = [call for call in fake.calls if kube_tools._CONNECT in call]
    assert connects[1][-1] == "10.217.5.9:8080"
    assert result["sandbox"] == [
        {"target": "gateway.strata.svc:8080", "address": "10.217.5.9", "port": 8080, "reached": False}
    ]
    assert result["harnesses"]["codex"] == {"started": True, "version": "1.2.3"}
    assert result["harnesses"]["bob"]["started"] is False

    # Everything it made is removed, the pods first.
    deletes = [call for call in fake.calls if call[0] == "delete"]
    assert [name.split("/")[0] for name in deletes[0][1:3]] == ["pod", "pod"]
    removed = {name for call in deletes for name in call[1:] if "/" in name}
    assert len(removed) == 5


def test_probe_says_no_where_the_rule_is_not_enforced(monkeypatch):
    result = _probe(monkeypatch, _ProbedCluster(enforces=False))
    assert result["isolated"] is False
    assert "does not enforce" in result["detail"]

    # The same answer on a cluster that enforces, with the rules left out.
    fake = _ProbedCluster(enforces=True)
    result = _probe(monkeypatch, fake, rules=False)
    assert [m["kind"] for m in fake.created] == ["Service", "Pod", "Pod"]
    assert result["isolated"] is False and result["rules"] is False


def test_probe_does_not_guess_when_nothing_answers_the_control(monkeypatch):
    fake = _ProbedCluster(enforces=True, control_reaches=False)
    result = _probe(monkeypatch, fake)
    assert result["isolated"] is None
    assert "could not be tested" in result["detail"]
    # Nothing to try from the sandbox, so it was not asked.
    assert len([call for call in fake.calls if kube_tools._CONNECT in call]) == 1


def test_probe_removes_what_it_made_when_it_fails(monkeypatch):
    fake = _ProbedCluster(enforces=True)
    monkeypatch.setattr(kube_runtime, "kube", fake.kube)
    monkeypatch.setattr(kube_tools, "_wait_ready", lambda settings, pod: (_ for _ in ()).throw(RuntimeError("Pod did not become ready")))
    with pytest.raises(RuntimeError, match="did not become ready"):
        kube_tools.probe(MARKED, image=IMAGE, targets=["a:1"], harnesses=[])
    removed = {name for call in fake.calls if call[0] == "delete" for name in call[1:] if "/" in name}
    assert len(removed) == 5


def test_sweep_removes_a_builds_leftovers_and_names_its_secret(monkeypatch):
    items = [
        {"kind": "Pod", "metadata": {"name": "tau2-claude-code-abc", "labels": {LABEL_BUILD: "tau2-claude-code-abc"}}},
        {"kind": "Pod", "metadata": {"name": "tau2-candidate-123", "labels": {LABEL_ROLE: "scoring"}}},
        {"kind": "Service", "metadata": {"name": "tau2-claude-code-abc-model-gateway", "labels": {LABEL_BUILD: "tau2-claude-code-abc"}}},
        {"kind": "NetworkPolicy", "metadata": {"name": "tau2-claude-code-abc-sandbox", "labels": {LABEL_BUILD: "tau2-claude-code-abc"}}},
    ]
    calls = []

    def fake(settings, args, *, input=None, timeout=60):
        calls.append(list(args))
        return SimpleNamespace(returncode=0, stdout=json.dumps({"items": items}) if args[0] == "get" else "", stderr="")

    monkeypatch.setattr(kube_runtime, "kube", fake)
    removed = kube_tools.sweep(KubeSettings(cli="oc", namespace="strata"), "strata.build=41-1")

    # Secrets are never listed: a right to list them is a right to read them all.
    assert calls[0][:2] == ["get", "pod,service,networkpolicy"]
    assert calls[0][2:4] == ["-l", "strata.build=41-1"]
    assert removed == [
        "pod/tau2-claude-code-abc",
        "pod/tau2-candidate-123",
        "secret/tau2-claude-code-abc-model-gateway",
        "service/tau2-claude-code-abc-model-gateway",
        "networkpolicy/tau2-claude-code-abc-sandbox",
    ]
    # The pods go first: a sandbox never stands without its rule.
    deletes = [call for call in calls if call[0] == "delete"]
    assert deletes[0][1:3] == ["pod/tau2-claude-code-abc", "pod/tau2-candidate-123"]
    assert "--ignore-not-found" in deletes[1]

    # A selector that selects everything is refused.
    for selector in ("", "strata.build"):
        with pytest.raises(ValueError, match="selector"):
            kube_tools.sweep(KubeSettings(), selector)


def test_the_commands_print_one_json_object(monkeypatch, capsys):
    monkeypatch.setenv("TAU2_KUBE_IMAGE", IMAGE)
    for name in ("TAU2_KUBE_LABELS", "TAU2_KUBE_OWNER_POD", "TAU2_KUBE_OWNER_UID", "TAU2_KUBE_RESOURCES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(kube_runtime, "kube", _ProbedCluster(enforces=True).kube)
    assert kube_tools.main(["probe", "--target", "gateway:8080", "--harness", "codex"]) == 0
    assert json.loads(capsys.readouterr().out)["isolated"] is True

    with pytest.raises(SystemExit):
        kube_tools.main(["probe", "--target", "no-port"])
    capsys.readouterr()

    monkeypatch.setattr(kube_runtime, "kube", lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="forbidden"))
    assert kube_tools.main(["sweep", "--selector", "strata.runner=openshift"]) == 1
    assert "Could not list" in json.loads(capsys.readouterr().out)["error"]
