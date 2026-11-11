"""Contract tests for the native Claude Code harness configuration."""

import json
from pathlib import Path
from types import SimpleNamespace

from tau2.hyper.harnesses.claude import (
    CLAUDE_CODE_HARNESS_VERSION,
    SKILLS_DIRECTIVE_PATH,
    ClaudeCodeSandboxBuilder,
    render_skills_directive,
    skill_name_of_call,
)
from tau2.hyper.sandbox.builder import BuildResult, BuildStep
from tau2.hyper.sandbox.orchestrator import _build_end_record
from tau2.hyper.live_experiment import LiveExperimentContext, SampleScenariosContext
from tau2.hyper.sandbox.builder import SandboxBuilder
from tau2.hyper.sandbox.native_runtime import NativeProcessEvent


def test_claude_builder_identity_is_harness_plus_model():
    builder = ClaudeCodeSandboxBuilder(
        "claude-opus-4-6",
        llm_args={"reasoning_effort": "high"},
        docker_image="runtime@sha256:abc",
    )

    assert isinstance(builder, SandboxBuilder)
    assert builder.harness_identity().to_dict() == {
        "name": "claude-code",
        "version": CLAUDE_CODE_HARNESS_VERSION,
        "config": {
            "interface": "print-stream-json",
            "permission_mode": "dontAsk-with-explicit-allowlist",
            "bare_mode": True,
            "web_tools": "removed",
            "browser": False,
            "background_tasks": False,
            "built_in_agents": False,
            "memory": False,
            "session_persistence": "runtime-home-only",
            "turn_loop": True,
            "nonessential_traffic": False,
            "model_gateway": "provider-only/per-run/model-scoped",
            "gateway_token_inherited_by_shell": True,
            "inner_bash_sandbox": "disabled/container-is-boundary",
            "mcp_servers": ["hyper_tau"],
        },
    }
    assert builder.model_identity().to_dict() == {
        "model": "claude-opus-4-6",
        "reasoning_effort": "high",
    }


def test_claude_command_removes_web_and_uses_only_private_mcp():
    builder = ClaudeCodeSandboxBuilder(
        "claude-opus-4-6", llm_args={"reasoning_effort": "medium"}
    )

    command = builder.harness_command()
    files = builder.runtime_files(include_client_tool=False)
    settings = json.loads(files[builder.runtime_config_path])
    mcp = json.loads(files[builder.mcp_config_path])

    assert command[:3] == ["python", "-m", "tau2.hyper.harnesses.turn_loop_driver"]
    assert command[3:5] == ["claude", "--print"]
    assert "--bare" in command
    assert command[command.index("--effort") + 1] == "medium"
    assert command[command.index("--permission-mode") + 1] == "dontAsk"
    assert (
        command[command.index("--allowedTools") + 1]
        == command[command.index("--tools") + 1]
    )
    assert command[command.index("--disallowedTools") + 1] == "WebSearch,WebFetch"
    assert "--strict-mcp-config" in command
    # The turn loop's `--continue` needs the session kept (below the private runtime home).
    assert "--no-session-persistence" not in command
    assert "--input-format" not in command, "the driver hands the prompt as an argument"
    assert "--no-chrome" in command
    assert settings["includeGitInstructions"] is True
    assert settings["disableClaudeAiConnectors"] is True
    assert settings["permissions"]["defaultMode"] == "dontAsk"
    assert settings["permissions"]["deny"] == ["WebSearch", "WebFetch"]
    assert list(mcp["mcpServers"]) == ["hyper_tau"]
    assert "talk_to_client" not in command[command.index("--tools") + 1]


def test_claude_enables_one_shot_live_experiment():
    builder = ClaudeCodeSandboxBuilder("claude-opus-4-6")
    builder.set_live_experiment_context(LiveExperimentContext(lambda: "result"))

    command = builder.harness_command()
    files = builder.runtime_files(
        include_client_tool=False,
        include_live_experiment_tool=True,
    )
    settings = json.loads(files[builder.runtime_config_path])

    assert "mcp__hyper_tau__run_live_experiment" in settings["permissions"]["allow"]
    assert (
        "mcp__hyper_tau__run_live_experiment" in command[command.index("--tools") + 1]
    )


def test_claude_enables_sample_scenarios_tool():
    # Regression: runtime_files() must accept the full kwarg set the native
    # build loop passes (a missing include_sample_scenarios_tool crashed
    # every claude-code run once the caller started sending it).
    builder = ClaudeCodeSandboxBuilder("claude-opus-4-6")
    builder.set_sample_scenarios_context(SampleScenariosContext(lambda: "result"))

    command = builder.harness_command()
    files = builder.runtime_files(
        include_client_tool=False,
        include_live_experiment_tool=False,
        include_sample_scenarios_tool=True,
    )
    settings = json.loads(files[builder.runtime_config_path])
    mcp = json.loads(files[builder.mcp_config_path])

    assert "mcp__hyper_tau__run_sample_scenarios" in settings["permissions"]["allow"]
    assert (
        "mcp__hyper_tau__run_sample_scenarios" in command[command.index("--tools") + 1]
    )
    assert (
        mcp["mcpServers"]["hyper_tau"]["env"]["TAU2_SAMPLE_SCENARIOS_TOOL_ENABLED"]
        == "${TAU2_SAMPLE_SCENARIOS_TOOL_ENABLED}"
    )


def test_claude_disables_nested_bash_sandbox():
    builder = ClaudeCodeSandboxBuilder("claude-opus-4-6")
    broker = SimpleNamespace(
        token="callback-token",
        client_tool_enabled=False,
    )

    environment = builder.runtime_environment(broker)

    # The construction container is the isolation boundary, so Claude Code's
    # inner Bash sandbox (a nested bubblewrap user namespace the container's
    # default security context forbids) is disabled; otherwise every Bash call
    # fails preflight. Env scrubbing also requires bubblewrap and is disabled;
    # the per-run gateway token is model-scoped and unexfiltratable (no egress).
    assert environment["CLAUDE_CODE_SANDBOXED"] == "1"
    assert environment["IS_SANDBOX"] == "1"
    assert environment["CLAUDE_CODE_SUBPROCESS_ENV_SCRUB"] == "0"
    assert environment["MCP_TIMEOUT"] == "28800000"


def test_construction_image_installs_bash_sandbox_binaries():
    dockerfile = (
        Path(__file__).resolve().parents[3]
        / "docker"
        / "hyper-construction"
        / "Dockerfile"
    ).read_text()

    # bubblewrap + socat are the Linux Bash-sandbox binaries; socat's absence
    # was what surfaced the nested-sandbox failure ("socat not installed").
    assert "        bubblewrap \\\n" in dockerfile
    assert "        socat \\\n" in dockerfile


def test_claude_stream_normalizes_text_tool_calls_and_results():
    builder = ClaudeCodeSandboxBuilder("claude-opus-4-6")
    assistant = NativeProcessEvent(
        sequence=0,
        channel="stdout",
        elapsed_seconds=1.0,
        text=json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "text", "text": "Inspecting the kit."},
                        {
                            "type": "tool_use",
                            "id": "tool-1",
                            "name": "Read",
                            "input": {"file_path": "/workspace/README.md"},
                        },
                    ]
                },
            }
        ),
    )
    user = NativeProcessEvent(
        sequence=1,
        channel="stdout",
        elapsed_seconds=2.0,
        text=json.dumps(
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool-1",
                            "content": "kit contents",
                            "is_error": False,
                        }
                    ]
                },
            }
        ),
    )

    assistant_step = builder.normalize_event(assistant)[0]
    result_step = builder.normalize_event(user)[0]

    assert assistant_step.content == "Inspecting the kit."
    assert assistant_step.tool_calls == [
        {
            "id": "tool-1",
            "name": "Read",
            "arguments": {"file_path": "/workspace/README.md"},
        }
    ]
    assert result_step.tool_results == [
        {
            "id": "tool-1",
            "name": "tool_result",
            "result": "kit contents",
            "is_error": False,
        }
    ]


def test_claude_skills_dir_rides_as_a_session_plugin(tmp_path):
    skills = tmp_path / "skills"
    (skills / "build-the-agent").mkdir(parents=True)
    (skills / "build-the-agent" / "SKILL.md").write_text("---\nname: build-the-agent\n---\nbody\n")
    (skills / "notes.txt").write_text("not a skill")

    builder = ClaudeCodeSandboxBuilder("claude-sonnet-4-5", skills_dir=skills)
    command = builder.harness_command()
    files = builder.runtime_files(include_client_tool=False)
    settings = json.loads(files[builder.runtime_config_path])

    assert builder.harness_identity().config["skills"] == ["build-the-agent"]
    assert builder.harness_identity().config["bare_mode"] is False
    assert builder.harness_identity().config["skills_directive"] == "claude-md"
    assert "--bare" not in command
    assert "--disable-slash-commands" not in command
    assert command[command.index("--plugin-dir") + 1] == "/runtime-home/plugins/strata-skills"
    assert "Skill" in command[command.index("--tools") + 1].split(",")
    assert "Skill" in settings["permissions"]["allow"]
    manifest = json.loads(files["/runtime-home/plugins/strata-skills/.claude-plugin/plugin.json"])
    assert manifest["name"] == "strata-skills"
    assert files["/runtime-home/plugins/strata-skills/skills/build-the-agent/SKILL.md"].endswith("body\n")
    # The CLAUDE.md Claude Code loads into every session's system prompt names the
    # skill and says to load it with the Skill tool first, then to keep going to submit.
    directive = files[SKILLS_DIRECTIVE_PATH]
    assert SKILLS_DIRECTIVE_PATH == "/runtime-home/.claude/CLAUDE.md"
    assert "`strata-skills:build-the-agent`" in directive
    assert "Skill tool" in directive
    assert "hyper_tau.submit" in directive
    # Every file stays below the private runtime home, never in the kit.
    assert all(path.startswith("/runtime-home/") for path in files)


def test_claude_without_skills_is_the_shipped_contract():
    builder = ClaudeCodeSandboxBuilder("claude-sonnet-4-5")
    command = builder.harness_command()

    assert "--disable-slash-commands" in command
    assert "--bare" in command
    assert "--plugin-dir" not in command
    assert "skills" not in builder.harness_identity().config
    assert "skills_directive" not in builder.harness_identity().config
    assert all(not path.endswith("CLAUDE.md") for path in builder.runtime_files(include_client_tool=False))
    assert builder.skills_read([]) is None, "no skills, nothing to read"
    assert "Skill" not in command[command.index("--tools") + 1].split(",")


def test_claude_empty_skills_dir_is_refused(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        ClaudeCodeSandboxBuilder("claude-sonnet-4-5", skills_dir=tmp_path)


def test_skills_directive_names_each_skill_with_its_description(tmp_path):
    skills = tmp_path / "skills"
    (skills / "policy-from-evidence").mkdir(parents=True)
    (skills / "policy-from-evidence" / "SKILL.md").write_text(
        "---\nname: policy-from-evidence\ndescription: \"Use when the policy is spread across evidence files.\"\n---\nbody\n"
    )
    (skills / "tau-agent-construction").mkdir()
    (skills / "tau-agent-construction" / "SKILL.md").write_text("---\nname: tau-agent-construction\n---\nno description\n")

    builder = ClaudeCodeSandboxBuilder("claude-sonnet-4-5", skills_dir=skills)
    assert [s["description"] for s in builder.skills] == [
        "Use when the policy is spread across evidence files.",
        "",
    ]
    directive = render_skills_directive(builder.skills)
    assert directive.startswith("# Skills for this build\n")
    assert "2 skills were handed to you" in directive
    assert (
        "- `strata-skills:policy-from-evidence` — Use when the policy is spread across evidence files."
        in directive
    )
    assert "- `strata-skills:tau-agent-construction`\n" in directive
    # Several skills: all of them first, whatever a description says.
    assert (
        "Load all 2 before anything else: your first message is 2 Skill calls, one for "
        "each skill. Each was picked for this build, so load it whatever its description "
        "says about when to use it, and even when another one is a complete procedure. "
        "A description is not the skill's text, and a skill that is not loaded is never "
        "read. Then follow them together.\n" in directive
    )
    assert directive.index("Load all 2") > directive.index("- `strata-skills:tau-agent-construction`")
    assert directive.index("Load all 2") < directive.index("Keep working until")


def test_one_skill_renders_the_directive_it_always_did():
    # The families measured on one skill keep their instruction byte for byte.
    directive = render_skills_directive(
        [{"name": "procedure", "content": "", "description": "steps", "files": {}}]
    )
    assert directive == (
        "# Skills for this build\n\nOne skill was handed to you for this build, as the plugin "
        "`strata-skills`. Before you read the kit, load each one with the Skill tool and follow "
        "what it says:\n\n- `strata-skills:procedure` — steps\n\nKeep working until you have "
        "called `hyper_tau.submit`: a session that ends without it is scored as the workspace "
        "stands, and the time left in your budget is lost.\n"
    )


def _stdout(elapsed: float, frame: dict) -> NativeProcessEvent:
    return NativeProcessEvent(sequence=0, channel="stdout", elapsed_seconds=elapsed, text=json.dumps(frame))


def test_claude_counts_turns_and_the_skills_the_developer_read(tmp_path):
    skills = tmp_path / "skills"
    for name in ("build-the-agent", "policy-from-evidence"):
        (skills / name).mkdir(parents=True)
        (skills / name / "SKILL.md").write_text(f"---\nname: {name}\n---\nbody\n")
    builder = ClaudeCodeSandboxBuilder("claude-sonnet-4-5", skills_dir=skills)
    builder.reset_build_state()

    steps = []
    assert builder.normalize_event(_stdout(0.5, {"type": "system", "subtype": "init", "session_id": "s1"})) == []
    steps += builder.normalize_event(
        _stdout(
            3.0,
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "tool_use", "id": "t1", "name": "Skill", "input": {"skill": "strata-skills:build-the-agent"}},
                    ]
                },
            },
        )
    )
    # A turn that ended with no submit: the loop continues it, and a second turn ends later.
    assert builder.normalize_event(_stdout(126.4, {"type": "result", "subtype": "success", "is_error": False, "num_turns": 28})) == []
    assert builder.normalize_event(_stdout(127.0, {"type": "system", "subtype": "init", "session_id": "s1"})) == []
    steps += builder.normalize_event(
        _stdout(
            300.0,
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "tool_use", "id": "t2", "name": "Skill", "input": {"skill": "strata-skills:build-the-agent"}},
                        {"type": "tool_use", "id": "t3", "name": "Read", "input": {"file_path": "/workspace/README.md"}},
                    ]
                },
            },
        )
    )
    builder.normalize_event(_stdout(900.0, {"type": "result", "subtype": "success", "is_error": False, "num_turns": 61, "result": "Submitted."}))
    # A third turn started and was killed by the host on submit: it has no end.
    assert builder.normalize_event(_stdout(901.0, {"type": "system", "subtype": "init", "session_id": "s1"})) == []

    assert builder.turns_started() == 3
    assert builder.build_turns() == [
        {"ended_at": 126.4, "subtype": "success", "is_error": False, "num_turns": 28},
        {"ended_at": 900.0, "subtype": "success", "is_error": False, "num_turns": 61},
    ]
    # Read once, named once; the other skill was never loaded.
    assert builder.skills_read(steps) == ["build-the-agent"]
    # The names a Skill call carries: with the plugin prefix, as a slash command, or bare.
    assert skill_name_of_call({"skill": "strata-skills:build-the-agent"}) == "build-the-agent"
    assert skill_name_of_call({"command": "/build-the-agent some args"}) == "build-the-agent"
    assert skill_name_of_call({"skill": "build-the-agent"}) == "build-the-agent"
    assert skill_name_of_call("not a dict") is None

    # What the orchestrator writes into the result file's build_result from the build's metadata.
    result = BuildResult(
        kit_path=tmp_path,
        steps=steps,
        metadata={
            "process": {"exit_code": 0, "timed_out": False, "cancelled": True, "event_count": 9, "stderr_tail": "x" * 5000},
            "turns": builder.build_turns(),
            "turns_started": builder.turns_started(),
            "skills_read": builder.skills_read(steps),
        },
    )
    record = _build_end_record(result)
    assert record["process"] == {"exit_code": 0, "timed_out": False, "cancelled": True, "stderr_tail": "x" * 1000}
    assert record["turns"] == 3, "the turns started: the third ended with the kill on submit"
    assert record["turn_ends"][0]["ended_at"] == 126.4
    assert record["skills_read"] == ["build-the-agent"]
    # A harness that records none of it leaves the keys out.
    assert _build_end_record(BuildResult(kit_path=tmp_path)) == {}


def _skill_with_files(tmp_path):
    skills = tmp_path / "skills"
    base = skills / "construction-procedure"
    (base / "scripts").mkdir(parents=True)
    (base / "references").mkdir()
    (base / "SKILL.md").write_text("---\nname: construction-procedure\n---\nrun scripts/audit.py\n")
    (base / "scripts" / "audit.py").write_text("print('audit')\n")
    (base / "references" / "policy.md").write_text("# Policy\n")
    (base / ".DS_Store").write_bytes(b"\x00\x01")
    (base / "scripts" / "__pycache__").mkdir()
    (base / "scripts" / "__pycache__" / "audit.cpython-312.pyc").write_bytes(b"\x00pyc")
    return skills


def test_claude_carries_a_skills_other_files_beside_its_skill_md(tmp_path):
    builder = ClaudeCodeSandboxBuilder("claude-sonnet-4-5", skills_dir=_skill_with_files(tmp_path))
    files = builder.runtime_files(include_client_tool=False)
    base = "/runtime-home/plugins/strata-skills/skills/construction-procedure"

    assert files[f"{base}/SKILL.md"].endswith("run scripts/audit.py\n")
    assert files[f"{base}/scripts/audit.py"] == "print('audit')\n"
    assert files[f"{base}/references/policy.md"] == "# Policy\n"
    # Hidden entries and bytecode stay behind; nothing else of the skill rides.
    assert sorted(p for p in files if p.startswith(base)) == [
        f"{base}/SKILL.md",
        f"{base}/references/policy.md",
        f"{base}/scripts/audit.py",
    ]
    assert builder.harness_identity().config["skill_files"] == {
        "construction-procedure": ["references/policy.md", "scripts/audit.py"]
    }


def test_a_skill_with_only_skill_md_records_no_files(tmp_path):
    skills = tmp_path / "skills"
    (skills / "build-the-agent").mkdir(parents=True)
    (skills / "build-the-agent" / "SKILL.md").write_text("---\nname: build-the-agent\n---\nbody\n")
    builder = ClaudeCodeSandboxBuilder("claude-sonnet-4-5", skills_dir=skills)

    assert builder.skills[0]["files"] == {}
    assert "skill_files" not in builder.harness_identity().config


def test_a_skill_file_that_is_binary_too_large_or_a_link_is_refused(tmp_path):
    import pytest

    from tau2.hyper.harnesses.claude import SKILL_FILE_MAX_BYTES, load_skills_dir

    def fresh(name):
        skills = tmp_path / name
        (skills / "s").mkdir(parents=True)
        (skills / "s" / "SKILL.md").write_text("---\nname: s\n---\nbody\n")
        return skills

    binary = fresh("binary")
    (binary / "s" / "logo.png").write_bytes(b"\x89PNG\x00\x00")
    with pytest.raises(ValueError, match="must be text"):
        load_skills_dir(binary)

    latin = fresh("latin")
    (latin / "s" / "notes.md").write_bytes("caf\xe9".encode("latin-1"))
    with pytest.raises(ValueError, match="UTF-8"):
        load_skills_dir(latin)

    large = fresh("large")
    (large / "s" / "big.md").write_text("x" * (SKILL_FILE_MAX_BYTES + 1))
    with pytest.raises(ValueError, match="over"):
        load_skills_dir(large)

    together = fresh("together")
    for index in range(5):
        (together / "s" / f"part{index}.md").write_text("x" * SKILL_FILE_MAX_BYTES)
    with pytest.raises(ValueError, match="together"):
        load_skills_dir(together)

    linked = fresh("linked")
    (tmp_path / "outside.md").write_text("outside\n")
    (linked / "s" / "outside.md").symlink_to(tmp_path / "outside.md")
    with pytest.raises(ValueError, match="symbolic link"):
        load_skills_dir(linked)


def test_a_skills_submit_gate_is_the_one_hook_on_the_submit_tool(tmp_path):
    skills = tmp_path / "skills"
    (skills / "procedure").mkdir(parents=True)
    (skills / "procedure" / "SKILL.md").write_text(
        "---\nname: procedure\ndescription: steps\n"
        "submit-gate: python3 scripts/before_submit.py /workspace --hook\n---\nbody\n"
    )
    (skills / "plain").mkdir()
    (skills / "plain" / "SKILL.md").write_text("---\nname: plain\n---\nbody\n")
    builder = ClaudeCodeSandboxBuilder("claude-sonnet-4-5", skills_dir=skills)
    settings = json.loads(builder.runtime_files(include_client_tool=False)[builder.runtime_config_path])

    assert settings["disableAllHooks"] is False
    assert settings["hooks"] == {
        "PreToolUse": [
            {
                "matcher": "mcp__hyper_tau__submit",
                "hooks": [
                    {
                        "type": "command",
                        "command": "cd /runtime-home/plugins/strata-skills/skills/procedure"
                        " && python3 scripts/before_submit.py /workspace --hook",
                        "timeout": 60,
                    }
                ],
            }
        ]
    }
    assert builder.harness_identity().config["submit_gates"] == {
        "procedure": "python3 scripts/before_submit.py /workspace --hook"
    }


def test_skills_without_a_submit_gate_keep_every_hook_off(tmp_path):
    skills = tmp_path / "skills"
    (skills / "plain").mkdir(parents=True)
    (skills / "plain" / "SKILL.md").write_text("---\nname: plain\n---\nsubmit-gate: in the body is not a gate\n")
    builder = ClaudeCodeSandboxBuilder("claude-sonnet-4-5", skills_dir=skills)
    settings = json.loads(builder.runtime_files(include_client_tool=False)[builder.runtime_config_path])

    assert settings["disableAllHooks"] is True
    assert "hooks" not in settings
    assert "submit_gates" not in builder.harness_identity().config
    bare = json.loads(ClaudeCodeSandboxBuilder("claude-sonnet-4-5").render_runtime_config(include_client_tool=False))
    assert bare["disableAllHooks"] is True and "hooks" not in bare


# What an arm carries besides skills: the rest of each plugin, below `_plugins`.


def _two_skills(tmp_path):
    skills = tmp_path / "skills"
    (skills / "procedure" / "scripts").mkdir(parents=True)
    (skills / "procedure" / "SKILL.md").write_text(
        "---\nname: procedure\ndescription: steps\n"
        "submit-gate: python3 scripts/before_submit.py /workspace --hook\n---\nbody\n"
    )
    (skills / "procedure" / "scripts" / "before_submit.py").write_text('print("gate")\n')
    (skills / "plain").mkdir()
    (skills / "plain" / "SKILL.md").write_text("---\nname: plain\n---\nbody\n")
    return skills


def _add_plugin(skills):
    """One plugin with an agent, a command, a hooks.json with its script, and an MCP server."""
    plugin = skills / "_plugins" / "review-kit"
    for folder in (".claude-plugin", "agents", "commands", "hooks", "servers"):
        (plugin / folder).mkdir(parents=True)
    (plugin / ".claude-plugin" / "plugin.json").write_text(
        '{"name": "review-kit", "version": "1.2.0", "description": "Review helpers"}\n'
    )
    (plugin / "agents" / "policy-reviewer.md").write_text(
        "---\nname: policy-reviewer\ndescription: Use to review the policy before submit.\n"
        "tools: Read, Grep, Glob\nmodel: haiku\n---\nYou review a policy file.\n"
    )
    (plugin / "commands" / "audit-kit.md").write_text(
        "---\ndescription: List what the kit holds.\n---\nList the kit's folders.\n"
    )
    (plugin / "hooks" / "hooks.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "SessionStart": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "node ${CLAUDE_PLUGIN_ROOT}/hooks/dispatch.mjs",
                                }
                            ]
                        }
                    ],
                    "PreToolUse": [{"matcher": "Bash", "hooks": []}],
                }
            }
        )
    )
    (plugin / "hooks" / "dispatch.mjs").write_text("process.exit(0);\n")
    (plugin / ".mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "kit-index": {
                        "command": "python",
                        "args": ["${CLAUDE_PLUGIN_ROOT}/servers/kit_index.py"],
                        "env": {"KIT_ROOT": "${CLAUDE_PLUGIN_ROOT}", "KEPT": "${HOME}"},
                    }
                }
            }
        )
    )
    (plugin / "servers" / "kit_index.py").write_text("print('server')\n")
    # Left behind: hidden entries other than the two Claude Code reads, and bytecode.
    (plugin / ".DS_Store").write_bytes(b"\x00\x01")
    (plugin / ".git").mkdir()
    (plugin / ".git" / "HEAD").write_text("ref\n")
    (plugin / "servers" / "__pycache__").mkdir()
    (plugin / "servers" / "__pycache__" / "kit_index.cpython-312.pyc").write_bytes(b"\x00")
    (skills / "_plugins" / ".DS_Store").write_bytes(b"\x00")
    return plugin


_SKILLS_ONLY_COMMAND = [
    "python",
    "-m",
    "tau2.hyper.harnesses.turn_loop_driver",
    "claude",
    "--print",
    "--output-format",
    "stream-json",
    "--verbose",
    "--model",
    "claude-sonnet-4-5",
    "--permission-mode",
    "dontAsk",
    "--tools",
    "Bash,Edit,Glob,Grep,NotebookEdit,Read,Write,mcp__hyper_tau__run_local_test,mcp__hyper_tau__submit,Skill",
    "--allowedTools",
    "Bash,Edit,Glob,Grep,NotebookEdit,Read,Write,mcp__hyper_tau__run_local_test,mcp__hyper_tau__submit,Skill",
    "--disallowedTools",
    "WebSearch,WebFetch",
    "--mcp-config",
    "/runtime-home/claude-mcp.json",
    "--strict-mcp-config",
    "--settings",
    "/runtime-home/.claude/settings.json",
    "--no-chrome",
    "--plugin-dir",
    "/runtime-home/plugins/strata-skills",
]
_SUBMIT_GATE_HOOKS = {
    "PreToolUse": [
        {
            "matcher": "mcp__hyper_tau__submit",
            "hooks": [
                {
                    "type": "command",
                    "command": "cd /runtime-home/plugins/strata-skills/skills/procedure"
                    " && python3 scripts/before_submit.py /workspace --hook",
                    "timeout": 60,
                }
            ],
        }
    ]
}
_SKILLS_ONLY_DIRECTIVE = (
    "# Skills for this build\n\n2 skills were handed to you for this build, as the plugin "
    "`strata-skills`. Before you read the kit, load each one with the Skill tool and follow "
    "what it says:\n\n- `strata-skills:plain`\n- `strata-skills:procedure` — steps\n\nLoad "
    "all 2 before anything else: your first message is 2 Skill calls, one for each skill. "
    "Each was picked for this build, so load it whatever its description says about when to "
    "use it, and even when another one is a complete procedure. A description is not the "
    "skill's text, and a skill that is not loaded is never read. Then follow them "
    "together.\n\nKeep working until you have called `hyper_tau.submit`: a session that "
    "ends without it is scored as the workspace stands, and the time left in your budget is "
    "lost.\n"
)
_BROKER = SimpleNamespace(token="callback-token", client_tool_enabled=False)


def test_a_skills_dir_with_no_plugins_renders_what_it_did_before(tmp_path):
    # The literals are what the driver rendered for this directory before it knew `_plugins`,
    # with the paragraph that two skills or more now get (load all of them first).
    skills = _two_skills(tmp_path)
    builder = ClaudeCodeSandboxBuilder("claude-sonnet-4-5", skills_dir=skills)
    files = builder.runtime_files(include_client_tool=False)

    assert builder.plugins == []
    assert builder.harness_command() == _SKILLS_ONLY_COMMAND
    assert list(files) == [
        "/runtime-home/.claude/settings.json",
        "/runtime-home/claude-mcp.json",
        "/runtime-home/.claude/CLAUDE.md",
        "/runtime-home/plugins/strata-skills/.claude-plugin/plugin.json",
        "/runtime-home/plugins/strata-skills/skills/plain/SKILL.md",
        "/runtime-home/plugins/strata-skills/skills/procedure/SKILL.md",
        "/runtime-home/plugins/strata-skills/skills/procedure/scripts/before_submit.py",
    ]
    assert files[SKILLS_DIRECTIVE_PATH] == _SKILLS_ONLY_DIRECTIVE
    assert json.loads(files[builder.runtime_config_path]) == {
        "autoMemoryEnabled": False,
        "disableAllHooks": False,
        "disableArtifact": True,
        "disableBundledSkills": True,
        "disableClaudeAiConnectors": True,
        "disableRemoteControl": True,
        "disableWorkflows": True,
        "enableAllProjectMcpServers": False,
        "hooks": _SUBMIT_GATE_HOOKS,
        "includeGitInstructions": True,
        "permissions": {
            "allow": _SKILLS_ONLY_COMMAND[13].split(","),
            "defaultMode": "dontAsk",
            "deny": ["WebSearch", "WebFetch"],
        },
    }
    assert list(json.loads(files[builder.mcp_config_path])["mcpServers"]) == ["hyper_tau"]
    assert builder.harness_config_metadata() == {
        "interface": "print-stream-json",
        "permission_mode": "dontAsk-with-explicit-allowlist",
        "bare_mode": False,
        "web_tools": "removed",
        "browser": False,
        "background_tasks": False,
        "built_in_agents": False,
        "memory": False,
        "session_persistence": "runtime-home-only",
        "turn_loop": True,
        "nonessential_traffic": False,
        "model_gateway": "provider-only/per-run/model-scoped",
        "gateway_token_inherited_by_shell": True,
        "inner_bash_sandbox": "disabled/container-is-boundary",
        "mcp_servers": ["hyper_tau"],
        "skills": ["plain", "procedure"],
        "skills_plugin": "strata-skills",
        "skills_directive": "claude-md",
        "skill_files": {"procedure": ["scripts/before_submit.py"]},
        "submit_gates": {"procedure": "python3 scripts/before_submit.py /workspace --hook"},
    }
    assert "CLAUDE_CODE_SUBAGENT_MODEL" not in builder.runtime_environment(_BROKER)

    # An empty `_plugins` carries nothing and changes nothing.
    (skills / "_plugins").mkdir()
    again = ClaudeCodeSandboxBuilder("claude-sonnet-4-5", skills_dir=skills)
    assert again.runtime_files(include_client_tool=False) == files
    assert again.harness_command() == builder.harness_command()
    assert again.harness_config_metadata() == builder.harness_config_metadata()


def test_claude_hands_over_the_rest_of_a_plugin(tmp_path):
    skills = _two_skills(tmp_path)
    plugin = _add_plugin(skills)
    builder = ClaudeCodeSandboxBuilder("claude-sonnet-4-5", skills_dir=skills)
    files = builder.runtime_files(include_client_tool=False)
    command = builder.harness_command()
    root = "/runtime-home/plugins/review-kit"

    # The files: every text file of the plugin at its own path, as it was written.
    assert sorted(path for path in files if path.startswith(root)) == [
        f"{root}/.claude-plugin/plugin.json",
        f"{root}/.mcp.json",
        f"{root}/agents/policy-reviewer.md",
        f"{root}/commands/audit-kit.md",
        f"{root}/hooks/dispatch.mjs",
        f"{root}/hooks/hooks.json",
        f"{root}/servers/kit_index.py",
    ]
    for relative in ("agents/policy-reviewer.md", "hooks/hooks.json", ".mcp.json"):
        assert files[f"{root}/{relative}"] == (plugin / relative).read_text()
    assert all(path.startswith("/runtime-home/") for path in files)
    # The skills ride as before, in their own plugin.
    assert files["/runtime-home/plugins/strata-skills/skills/plain/SKILL.md"].endswith("body\n")

    # The command: the skills run's, with the plugin's directory and the tools its parts need.
    tools = _SKILLS_ONLY_COMMAND[13] + ",Agent,mcp__plugin_review-kit_kit-index"
    assert command == [
        *_SKILLS_ONLY_COMMAND[:13],
        tools,
        "--allowedTools",
        tools,
        *_SKILLS_ONLY_COMMAND[16:],
        "--plugin-dir",
        root,
    ]
    assert "--bare" not in command and "--disable-slash-commands" not in command
    assert "--strict-mcp-config" in command

    # The settings: hooks on, the submit gate still the one hook the file names, the same allowlist.
    settings = json.loads(files[builder.runtime_config_path])
    assert settings["disableAllHooks"] is False
    assert settings["hooks"] == _SUBMIT_GATE_HOOKS
    assert settings["permissions"]["allow"] == tools.split(",")
    assert settings["permissions"]["deny"] == ["WebSearch", "WebFetch"]

    # The MCP config: the plugin's server beside hyper_tau, under the name Claude Code gives
    # a plugin's server, with the plugin's root resolved and other variables left to it.
    servers = json.loads(files[builder.mcp_config_path])["mcpServers"]
    assert list(servers) == ["hyper_tau", "plugin:review-kit:kit-index"]
    assert servers["plugin:review-kit:kit-index"] == {
        "command": "python",
        "args": [f"{root}/servers/kit_index.py"],
        "env": {"KIT_ROOT": root, "KEPT": "${HOME}"},
    }

    # Every sub-agent runs on the run's model: the gateway takes no other.
    assert builder.runtime_environment(_BROKER)["CLAUDE_CODE_SUBAGENT_MODEL"] == "claude-sonnet-4-5"

    # The record: what was handed, per plugin, under the names the session uses.
    config = builder.harness_identity().config
    assert config["plugins"] == {
        "review-kit": {
            "path": root,
            "version": "1.2.0",
            "agents": ["review-kit:policy-reviewer"],
            "commands": ["review-kit:audit-kit"],
            "hook_events": ["PreToolUse", "SessionStart"],
            "mcp_servers": ["plugin:review-kit:kit-index"],
            "files": [
                ".claude-plugin/plugin.json",
                ".mcp.json",
                "agents/policy-reviewer.md",
                "commands/audit-kit.md",
                "hooks/dispatch.mjs",
                "hooks/hooks.json",
                "servers/kit_index.py",
            ],
        }
    }
    assert config["mcp_servers"] == ["hyper_tau", "plugin:review-kit:kit-index"]
    assert config["subagent_model"] == "claude-sonnet-4-5"
    assert config["built_in_agents"] is False and config["bare_mode"] is False
    # The skills' keys are the ones a skills run records.
    assert config["skills"] == ["plain", "procedure"]
    assert config["skills_plugin"] == "strata-skills"
    assert config["skill_files"] == {"procedure": ["scripts/before_submit.py"]}
    assert config["submit_gates"] == {
        "procedure": "python3 scripts/before_submit.py /workspace --hook"
    }

    # The directive: the skills part as it was, then the agents and commands by name.
    directive = files[SKILLS_DIRECTIVE_PATH]
    assert directive.startswith(_SKILLS_ONLY_DIRECTIVE)
    assert directive[len(_SKILLS_ONLY_DIRECTIVE) :] == (
        "\n# Agents and commands for this build\n\n"
        "These were handed to you for this build. Use one when the work calls for it; "
        "you do not have to use them.\n\n"
        "Agent definitions, to delegate to with the Agent tool:\n\n"
        "- `review-kit:policy-reviewer` — Use to review the policy before submit.\n\n"
        "Commands, to run with the Skill tool:\n\n"
        "- `review-kit:audit-kit` — List what the kit holds.\n"
    )


def test_a_dir_with_plugins_and_no_skill_is_valid(tmp_path):
    skills = tmp_path / "skills"
    skills.mkdir()
    _add_plugin(skills)
    builder = ClaudeCodeSandboxBuilder("claude-sonnet-4-5", skills_dir=skills)
    files = builder.runtime_files(include_client_tool=False)
    command = builder.harness_command()
    settings = json.loads(files[builder.runtime_config_path])
    config = builder.harness_identity().config

    assert builder.skills == [] and builder.skills_read([]) is None
    assert not any(path.startswith("/runtime-home/plugins/strata-skills") for path in files)
    assert [command[i + 1] for i, arg in enumerate(command) if arg == "--plugin-dir"] == [
        "/runtime-home/plugins/review-kit"
    ]
    assert "--bare" not in command and "--disable-slash-commands" not in command
    # The Skill tool runs the plugin's command; the Agent tool reaches its agent.
    assert command[command.index("--tools") + 1].split(",")[-3:] == [
        "Skill",
        "Agent",
        "mcp__plugin_review-kit_kit-index",
    ]
    # A plugin's hooks run with no submit gate in the settings file.
    assert settings["disableAllHooks"] is False and "hooks" not in settings
    assert "skills" not in config and "skills_directive" not in config
    assert config["bare_mode"] is False
    assert list(config["plugins"]) == ["review-kit"]
    assert files[SKILLS_DIRECTIVE_PATH].startswith("# Agents and commands for this build\n")
    assert "Skills for this build" not in files[SKILLS_DIRECTIVE_PATH]

    # A plugin of hooks alone: no manifest, no tool added, nothing to name in a CLAUDE.md.
    hooks_only = tmp_path / "hooks-only"
    (hooks_only / "_plugins" / "just-hooks" / "hooks").mkdir(parents=True)
    (hooks_only / "_plugins" / "just-hooks" / "hooks" / "hooks.json").write_text(
        '{"hooks": {"Stop": []}}'
    )
    builder = ClaudeCodeSandboxBuilder("claude-sonnet-4-5", skills_dir=hooks_only)
    files = builder.runtime_files(include_client_tool=False)
    command = builder.harness_command()
    assert SKILLS_DIRECTIVE_PATH not in files
    assert command[command.index("--tools") + 1] == ",".join(_SKILLS_ONLY_COMMAND[13].split(",")[:-1])
    assert json.loads(files[builder.runtime_config_path])["disableAllHooks"] is False
    assert builder.harness_identity().config["plugins"]["just-hooks"] == {
        "path": "/runtime-home/plugins/just-hooks",
        "agents": [],
        "commands": [],
        "hook_events": ["Stop"],
        "mcp_servers": [],
        "files": ["hooks/hooks.json"],
    }
    assert "subagent_model" not in builder.harness_identity().config
    assert "CLAUDE_CODE_SUBAGENT_MODEL" not in builder.runtime_environment(_BROKER)

    # A plugin with no hooks leaves every hook off, as a run with no gate does.
    agent_only = tmp_path / "agent-only"
    (agent_only / "_plugins" / "one-agent" / "agents").mkdir(parents=True)
    (agent_only / "_plugins" / "one-agent" / "agents" / "helper.md").write_text(
        "---\nname: helper\ndescription: Helps.\n---\nbody\n"
    )
    builder = ClaudeCodeSandboxBuilder("claude-sonnet-4-5", skills_dir=agent_only)
    settings = json.loads(builder.runtime_files(include_client_tool=False)[builder.runtime_config_path])
    assert settings["disableAllHooks"] is True
    assert settings["permissions"]["allow"][-1] == "Agent"
    assert "Skill" not in settings["permissions"]["allow"]


def test_plugin_parts_are_named_as_the_session_names_them(tmp_path):
    from tau2.hyper.harnesses.claude import load_plugins_dir, mcp_server_rule

    skills = tmp_path / "skills"
    plugin = skills / "_plugins" / "probe-dir"
    for folder in (".claude-plugin", "agents/team", "commands/group"):
        (plugin / folder).mkdir(parents=True)
    # The manifest's name is the plugin's, not the directory's.
    (plugin / ".claude-plugin" / "plugin.json").write_text('{"name": "probe-kit"}')
    (plugin / "agents" / "reviewer-file.md").write_text(
        "---\nname: reviewer-name\ndescription: \"" + "x" * 300 + "\"\n---\nbody\n"
    )
    (plugin / "agents" / "team" / "helper.md").write_text("---\nname: helper\n---\nbody\n")
    (plugin / "agents" / "no-name.md").write_text("---\ndescription: No name line.\n---\nbody\n")
    (plugin / "commands" / "group" / "deep.md").write_text("---\nname: ignored\n---\nbody\n")
    (plugin / "commands" / "plain.md").write_text("No frontmatter.\n")
    (plugin / "commands" / "manual-only.md").write_text(
        "---\ndescription: For a person.\ndisable-model-invocation: true\n---\nbody\n"
    )
    # The flat form of .mcp.json, and a server name with a character a tool name cannot hold.
    (plugin / ".mcp.json").write_text('{"kit.index": {"command": "python"}}')

    (found,) = load_plugins_dir(skills)
    assert found["name"] == "probe-kit" and found["version"] is None
    assert found["path"] == "/runtime-home/plugins/probe-dir"
    assert [agent["name"] for agent in found["agents"]] == [
        "probe-kit:no-name",
        "probe-kit:reviewer-name",
        "probe-kit:team:helper",
    ]
    assert [command["name"] for command in found["commands"]] == [
        "probe-kit:group:deep",
        "probe-kit:manual-only",
        "probe-kit:plain",
    ]
    assert list(found["mcp_servers"]) == ["plugin:probe-kit:kit.index"]
    assert mcp_server_rule("plugin:probe-kit:kit.index") == "mcp__plugin_probe-kit_kit_index"

    builder = ClaudeCodeSandboxBuilder("claude-sonnet-4-5", skills_dir=skills)
    record = builder.harness_identity().config["plugins"]["probe-kit"]
    # A command a person types is carried as a file; the model cannot run it, and the record says so.
    assert record["commands"] == ["probe-kit:group:deep", "probe-kit:plain"]
    assert record["commands_user_only"] == ["probe-kit:manual-only"]
    assert "version" not in record
    directive = render_skills_directive(builder.skills, builder.plugins)
    assert "- `probe-kit:plain`\n" in directive
    assert "manual-only" not in directive
    assert "- `probe-kit:reviewer-name` — " + "x" * 200 + "…\n" in directive
    assert builder.harness_command()[-2:] == ["--plugin-dir", "/runtime-home/plugins/probe-dir"]


def test_a_plugin_that_breaks_a_rule_is_refused(tmp_path):
    import pytest

    from tau2.hyper.harnesses.claude import SKILL_FILE_MAX_BYTES, load_plugins_dir

    def fresh(name, plugin="kit"):
        skills = tmp_path / name
        (skills / "_plugins" / plugin / "agents").mkdir(parents=True)
        (skills / "_plugins" / plugin / "agents" / "helper.md").write_text(
            "---\nname: helper\n---\nbody\n"
        )
        return skills, skills / "_plugins" / plugin

    skills, plugin = fresh("binary")
    (plugin / "logo.png").write_bytes(b"\x89PNG\x00\x00")
    with pytest.raises(ValueError, match="A plugin file must be text"):
        load_plugins_dir(skills)

    skills, plugin = fresh("large")
    (plugin / "big.md").write_text("x" * (SKILL_FILE_MAX_BYTES + 1))
    with pytest.raises(ValueError, match="A plugin file is over"):
        load_plugins_dir(skills)

    skills, plugin = fresh("together")
    for index in range(5):
        (plugin / f"part{index}.md").write_text("x" * SKILL_FILE_MAX_BYTES)
    with pytest.raises(ValueError, match="together"):
        load_plugins_dir(skills)

    (tmp_path / "outside.md").write_text("outside\n")
    skills, plugin = fresh("linked-file")
    (plugin / "agents" / "outside.md").symlink_to(tmp_path / "outside.md")
    with pytest.raises(ValueError, match="symbolic link"):
        load_plugins_dir(skills)

    skills, plugin = fresh("linked-plugin")
    (skills / "_plugins" / "other").symlink_to(plugin)
    with pytest.raises(ValueError, match="symbolic link"):
        load_plugins_dir(skills)

    skills, plugin = fresh("linked-root")
    (tmp_path / "linked-root-2").mkdir()
    (tmp_path / "linked-root-2" / "_plugins").symlink_to(skills / "_plugins")
    with pytest.raises(ValueError, match="symbolic link"):
        load_plugins_dir(tmp_path / "linked-root-2")

    skills, plugin = fresh("with-skills")
    (plugin / "skills" / "s").mkdir(parents=True)
    (plugin / "skills" / "s" / "SKILL.md").write_text("---\nname: s\n---\nbody\n")
    with pytest.raises(ValueError, match="top of the skills directory"):
        load_plugins_dir(skills)

    skills, plugin = fresh("bad-dir-name", plugin="Review_Kit")
    with pytest.raises(ValueError, match="kebab-case"):
        load_plugins_dir(skills)

    skills, plugin = fresh("reserved", plugin="strata-skills")
    with pytest.raises(ValueError, match="strata-skills"):
        load_plugins_dir(skills)

    skills, plugin = fresh("bad-manifest-name")
    (plugin / ".claude-plugin").mkdir()
    (plugin / ".claude-plugin" / "plugin.json").write_text('{"name": "Review Kit"}')
    with pytest.raises(ValueError, match="kebab-case"):
        load_plugins_dir(skills)

    skills, plugin = fresh("same-name")
    (skills / "_plugins" / "twin" / ".claude-plugin").mkdir(parents=True)
    (skills / "_plugins" / "twin" / ".claude-plugin" / "plugin.json").write_text('{"name": "kit"}')
    with pytest.raises(ValueError, match="Two plugins"):
        load_plugins_dir(skills)

    for relative in (".claude-plugin/plugin.json", "hooks/hooks.json", ".mcp.json"):
        skills, plugin = fresh("not-json-" + relative.replace("/", "-").strip("."))
        (plugin / relative).parent.mkdir(exist_ok=True)
        (plugin / relative).write_text("{not json")
        with pytest.raises(ValueError, match="Not JSON"):
            load_plugins_dir(skills)

    skills, plugin = fresh("servers-not-objects")
    (plugin / ".mcp.json").write_text('{"mcpServers": {"a": "python"}}')
    with pytest.raises(ValueError, match="each server"):
        load_plugins_dir(skills)

    skills = tmp_path / "empty-plugin"
    (skills / "_plugins" / "kit" / "agents").mkdir(parents=True)
    with pytest.raises(ValueError, match="no file"):
        load_plugins_dir(skills)

    # Neither a skill nor a plugin: still a misconfiguration, as an empty directory is.
    nothing = tmp_path / "nothing"
    (nothing / "_plugins").mkdir(parents=True)
    with pytest.raises(ValueError, match="No <name>/SKILL.md"):
        ClaudeCodeSandboxBuilder("claude-sonnet-4-5", skills_dir=nothing)


def test_a_plugins_command_named_as_a_skill_is_not_that_skill_read(tmp_path):
    skills = _two_skills(tmp_path)
    plugin = _add_plugin(skills)
    (plugin / "commands" / "procedure.md").write_text("---\ndescription: Same name.\n---\nbody\n")
    builder = ClaudeCodeSandboxBuilder("claude-sonnet-4-5", skills_dir=skills)

    def step(skill):
        return BuildStep(
            step_idx=0,
            role="assistant",
            tool_calls=[{"id": "t", "name": "Skill", "arguments": {"skill": skill}}],
        )

    assert builder.skills_read([step("review-kit:procedure")]) == []
    assert builder.skills_read([step("review-kit:procedure"), step("strata-skills:procedure")]) == [
        "procedure"
    ]
    assert builder.skills_read([step("procedure"), step("/plain")]) == ["procedure", "plain"]
