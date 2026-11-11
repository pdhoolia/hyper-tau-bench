# Hyper-tau construction runtime

Build a commit-pinned Docker image used by construction-task sandbox runs:

```bash
REVISION=$(git rev-parse HEAD)
docker build \
  --build-arg TAU2_SOURCE_REVISION="$REVISION" \
  -f docker/hyper-construction/Dockerfile \
  -t "tau2-construction-runtime:$REVISION" \
  .
export TAU2_SANDBOX_DOCKER_IMAGE="tau2-construction-runtime:$REVISION"
```

The image includes tau2 framework/runtime code and user-simulator prompts.
`strip_runtime_src.py` (run at build) removes `src/tau2/domains` and
`data/tau2/domains`, removes the project-root `README.md`, `pyproject.toml`,
and `uv.lock` (they name the benchmark, its paper, and the repo; only the
pre-strip `uv sync` needs them), replaces the eager package `__init__` with a
minimal one
so importing framework modules does not pull in the domain registry, and
strips `src/tau2/hyper` down to a fail-closed allowlist of the modules the
in-container runtime executes: the sealed candidate server, the MCP stub, the
model-gateway sidecar, the Developer-facing contract surface
(`agent_context`, `client_api/__init__.py`), and the runtime-contract check.
Host-only machinery — including `client_api/defects.py` and
`client_api/runtime.py` — graders, client-simulator internals, and task
construction — never ships; a new module under `tau2/hyper` is stripped
unless allowlisted, and `tests/test_hyper/test_runtime_image_surface.py`
mirrors the allowlist at review time. Hyper task data is not copied into the
image. At runtime the orchestrator mounts only the generated developer kit at
`/workspace` and applies the benchmark-owned restricted network profile.

Final scoring uses the same image through the sealed runner with a stricter
profile: read-only root filesystem and kit mount, no network, no provider
credentials, an unprivileged user, dropped Linux capabilities, and bounded
process/model-call use. The trusted tau2 runtime is imported in Python isolated
mode before `/workspace` is added to `sys.path`, preventing a submission from
shadowing the runner at startup. Model requests are sent to a trusted host
broker, which enforces the model choices and constraints selected by the
scorer.

Sandbox startup checks that the image exposes the same construction-runtime
contract version as the host checkout. It also records the image's immutable
Docker ID and source revision in the run trajectory. Avoid `:latest` for
recorded experiments; select a commit tag as shown above or an image digest.

The image pins Codex and Claude Code and disables their automatic update
paths. It includes the Tau Python environment, so numerical/ML work and common
agent or prompt experiments can use NumPy, SciPy, pandas, scikit-learn,
matplotlib, seaborn, Plotly, tokenizers/tiktoken, Jinja, JSON Schema,
OpenAI/LiteLLM model clients, Pydantic, and pytest without a runtime install.

The engineering toolchain includes Git and Git LFS, ripgrep, tree, jq, curl,
make and compilers, patch/diff, ShellCheck, SQLite, process inspection,
OpenSSH, rsync, and archive utilities. Git has a benchmark-local identity so
the Developer may commit, branch, inspect history, and use worktrees.

Document tooling mirrors what common agent sandboxes preinstall: Poppler
utilities (`pdftotext`, `pdftoppm`, `pdfinfo`), `qpdf`, and the `pypdf`,
`openpyxl`, `python-docx`, and `python-pptx` Python libraries. Kits carry
evidence in every office format (pdf/xlsx/docx/pptx alongside text, zip,
and sqlite), and the no-egress network means a Developer cannot install a
missing parser at runtime, so the standard tools are baked in. OCR
(tesseract) and speech-to-text are deliberately NOT included:
screenshot-, recording-, and audio-gated facts are part of the evaluation
design, and transcription would convert media-only evidence into text for
text-only builder models (ffmpeg remains available for decoding and frame
extraction).

Runtime access to apt, PyPI, npm, GitHub, and other public endpoints is blocked
by the benchmark-owned internal Docker network. `curl` and the language
clients remain useful for local development and benchmark-owned endpoints;
their presence does not create an internet route.

See [Sealed scoring](../../README.md#sealed-scoring) in the repository README
for the trust boundary and data-flow contract.

## The same build as pods (Kubernetes, OpenShift)

Docker is the default. `TAU2_SANDBOX_BACKEND=kubernetes` starts a build's
containers as pods through the cluster's command line instead
(`src/tau2/hyper/sandbox/kube_runtime.py`). The benchmark still runs on the
host; only the containers move.

```bash
export TAU2_SANDBOX_BACKEND=kubernetes
export TAU2_KUBE_CLI=oc                       # default kubectl
export TAU2_KUBE_NAMESPACE=hyper-trial        # default: the current context's
export TAU2_KUBE_IMAGE=image-registry.openshift-image-registry.svc:5000/hyper-trial/tau2-construction-runtime:$REVISION
export TAU2_KUBE_USAGE_LOG=usage.jsonl        # optional: one line per pod when it ends (peak memory, CPU seconds)
uv run tau2 hyper-tau <task> --developer-harness claude-code …
```

`TAU2_KUBE_IMAGE` is this image as the cluster pulls it (a Docker image name
is a name in the machine's daemon). `TAU2_KUBE_CONTEXT` names a context other
than the current one.

| Docker | Pods |
|---|---|
| the sandbox container, `sleep infinity`, commands by `docker exec` | a pod that sleeps, commands by `exec` |
| the model gateway sidecar on the internal network | a pod beside it, behind a Service of its own; its keys in a Secret |
| the internal network (`--internal`) | two NetworkPolicies: the sandbox reaches the model gateway's port and nothing else, and nothing reaches the sandbox |
| the scoring container, `--network none`, fed over standard input | a pod with a NetworkPolicy that lets nothing in or out, fed over the standard input of an `exec` |
| the kit bound at `/workspace` | the kit copied into a volume of the sandbox pod, copied out before every callback and when the build ends; what the host writes during a callback is copied back in |
| the callback folder bound at `/run/tau2-callback` | a volume, with a relay that carries each request file out and each response back over one `exec` |

Every pod asks for nothing privileged and names no user id: no capability, no
privilege escalation, the runtime's default seccomp profile, `runAsNonRoot`,
no service account token. A cluster that assigns the user id (OpenShift's
`restricted-v2`) admits them, and the image's writable folders belong to group
0 for that. The rights the path needs are those of a project's admin: pods,
`pods/exec`, `pods/log`, services, secrets and network policies in one
namespace.

The sandbox and scoring pods have no name service (the rule would drop a
lookup silently and a library would wait out the resolver's timeout), so a
lookup there fails at once; the sandbox finds the model gateway by a hosts
line. The scoring pod's kit is a copy without its Git metadata and without
write bits.

### Builds started from outside (a runner in the cluster)

Whoever starts builds from outside marks what each one makes, so that a build
that was stopped or lost can be found and removed. Every setting is optional:

```bash
export TAU2_KUBE_LABELS=strata.runner=openshift,strata.build=41-1   # labels of the caller's own on every object a build makes
export TAU2_KUBE_OWNER_POD=strata-runner-7c9d                       # a pod of the same namespace that owns them all:
export TAU2_KUBE_OWNER_UID=$(oc get pod strata-runner-7c9d -o jsonpath='{.metadata.uid}')   # removed with it by the cluster
export TAU2_KUBE_PULL_SECRETS=icr-image-pull                        # the Secrets the build's pods pull the image with
export TAU2_KUBE_RESOURCES='{"scoring":{"requests":{"cpu":"100m","memory":"256Mi"},"limits":{"cpu":"1","memory":"768Mi"}}}'
export TAU2_KUBE_LOG_DIR=pods                                       # keeps what goes with a pod (below)
```

- **The marks.** The labels and the owner go on the sandbox, the model gateway,
  its Service and its Secret, the build's two NetworkPolicies and every scoring
  pod. A scoring pod has no other label that names its build. The one rule all
  scoring pods share (`tau2-hyper-scoring`) is no build's and is not marked:
  removed with one build, it would open the others'.
- **Requests and limits.** `TAU2_KUBE_RESOURCES` is an object keyed by the
  pod's kind (`sandbox`, `model-gateway`, `scoring`). `requests` there replaces
  the measured requests, `limits` replaces the task's ceilings, and an empty
  object takes them away. A kind it does not name keeps what it had. Use it
  where a quota counts limits.
- **A full quota.** A pod the project's quota refuses is tried again every five
  seconds for as long as a pod gets to become ready
  (`TAU2_KUBE_READY_TIMEOUT_SECONDS`, 300), not failed: the pods of other
  builds end by themselves.
- **What is kept.** With `TAU2_KUBE_LOG_DIR` the model gateway's log is written
  there before its pod is removed (`<pod>.log`), and `pods.jsonl` gets one line
  per pod: its peak memory and CPU seconds, `oom_kills` when a process was
  killed at the pod's memory limit (the pod stays, the process is gone), or
  `ended` with the cluster's reason for a pod that was no longer running
  (evicted, say). The same reasons are warnings in the run's log.
- **A stop.** SIGINT and SIGTERM both end `tau2 hyper-tau` as Ctrl-C does: the
  build closes, copies the kit out and removes what it made. A build that was
  killed removes nothing; the sweep below does.

Two commands, each printing one JSON object, reading the same environment:

```bash
# What builds with these labels left, removed: pods first, then each build's Secret (by its
# name: the path needs no right to list Secrets), Service and rules.
uv run python -m tau2.hyper.sandbox.kube_tools sweep --selector strata.build=41-1

# Is a sandbox pod isolated on this cluster? A sandbox behind a build's own rules and a
# control pod with none try the same addresses. `isolated` is true (the control reached
# one, the sandbox none), false (the sandbox reached one: the network plugin does not
# enforce NetworkPolicy), or null (nothing answered the control, so nothing was tested).
# It also says the sandbox's user id and whether each named harness's command starts there.
uv run python -m tau2.hyper.sandbox.kube_tools probe \
  --target gateway.strata.svc.cluster.local:8080 --target 1.1.1.1:443 \
  --harness claude-code --harness codex --harness opencode
```

`probe --no-rules` leaves the build's rules out and shows the answer of a
cluster that enforces none.

What a build leaves behind where no labels were set carries this path's own,
`tau2.hyper/build` and `tau2.hyper/role`:

```bash
oc delete pod,service,secret,networkpolicy -l tau2.hyper/build
oc delete pod -l tau2.hyper/role=scoring
```

A task's `docker_memory` and `docker_cpus` are ceilings under Docker, so here
they are the pod's limits and never its requests: eight scoring pods that each
reserved 2 CPUs would wait for a node that runs them side by side. Each kind
of pod requests what one was measured to use (`REQUESTS` in `kube_runtime.py`:
the sandbox 250m and 768Mi, a scoring pod 100m and 256Mi, the model gateway
50m and 64Mi).

Only a Claude Code build has been run on this path. The other harnesses go
through the same three calls of the sandbox, and `probe --harness` says whether
each one's command starts under the user id the cluster assigns; none has built
as pods. Not there yet: a user id for a cluster that assigns none (the image
names no user, so a plain Kubernetes refuses a pod that must not run as root),
and a kit copy that sends only what changed.
