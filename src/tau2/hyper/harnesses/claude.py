"""Claude Code integration for the Hyper-τ construction sandbox."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from tau2.hyper.sandbox.builder import BuildStep
from tau2.hyper.sandbox.model_gateway import ModelGatewaySpec
from tau2.hyper.sandbox.native_builder import NativeSandboxBuilder
from tau2.hyper.sandbox.native_runtime import NativeProcessEvent

CLAUDE_CODE_HARNESS_VERSION = "2.1.219"

# Skills a run hands the Developer ride as one plugin below the private runtime
# home (never the kit): ``--plugin-dir`` loads it for the session, and the
# harness's skills switch stays on for that run only. A CLAUDE.md beside it, the
# user-level file Claude Code puts in every session's context (a `# claudeMd`
# system reminder in the first user message, carried by every later call), tells
# the Developer to load each skill with the Skill tool before it reads the kit: a
# skill that is only described is never read (the first runs showed it), and a
# skill that is never read measures nothing. With more than one skill it also
# says to load all of them first, whatever a description says: builds that
# carried several loaded the one that reads as the build's procedure and never
# the rest. One skill renders as it always did.
SKILLS_PLUGIN_NAME = "strata-skills"
SKILLS_PLUGIN_PATH = f"/runtime-home/plugins/{SKILLS_PLUGIN_NAME}"
SKILLS_DIRECTIVE_PATH = "/runtime-home/.claude/CLAUDE.md"
_SKILL_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_DESCRIPTION_RE = re.compile(r"^description:\s*(.+?)\s*$", re.MULTILINE)
_SUBMIT_GATE_RE = re.compile(r"^submit-gate:\s*(.+?)\s*$", re.MULTILINE)
# A skill may name a submit gate in its frontmatter (``submit-gate: <command>``):
# a command run from the skill's own directory before every hyper_tau.submit. It
# exits 2 to refuse the submit, with the reason on stderr, which Claude Code shows
# to the Developer; any other exit lets the submit through, so a gate that breaks
# never traps a build. The driver installs it as a PreToolUse hook on the submit
# tool, the one hook a skills run turns on; every other hook stays off. The clock
# still ends a build that never submits, and the workspace is scored as it stands.
SUBMIT_TOOL = "mcp__hyper_tau__submit"
# A skill's other files (scripts/, references/, …) ride beside its SKILL.md at
# the same relative paths. Text only (UTF-8, no NUL byte), each file at most
# SKILL_FILE_MAX_BYTES and a skill's files together at most
# SKILL_FILES_MAX_BYTES; hidden entries and __pycache__ are left out, and a
# symbolic link is refused. A file that breaks a rule stops the run before it
# starts: a skill that names a file the sandbox does not have measures nothing.
SKILL_FILE_MAX_BYTES = 256 * 1024
SKILL_FILES_MAX_BYTES = 1024 * 1024
# The skills directory may also hold the rest of each plugin the run carries, one
# directory per plugin below the reserved child ``_plugins`` (never a skill's
# name: a skill is a kebab-case slug):
#
#   _plugins/<plugin>/.claude-plugin/plugin.json   optional manifest (name, version)
#   _plugins/<plugin>/agents/<name>.md             agent definitions
#   _plugins/<plugin>/commands/<name>.md           commands
#   _plugins/<plugin>/hooks/hooks.json             hooks, and the files its commands name
#   _plugins/<plugin>/.mcp.json                    MCP servers
#
# A plugin's skills are not here: they ride at the top of the skills directory,
# as every skill does. Each plugin is written to ``/runtime-home/plugins/<plugin>``
# and loaded with its own ``--plugin-dir``. What Claude Code 2.1.219 then makes of
# each part was checked in the image, with no network:
#
# - An agent definition is the sub-agent ``<plugin>:<name>`` (``name:`` of its
#   frontmatter, else the file's name; a folder below ``agents/`` is one more
#   ``:`` part). The session delegates with the ``Agent`` tool, which the
#   allowlist gains only then; the built-in agents stay off. The model gateway
#   takes the run's model alone, so ``CLAUDE_CODE_SUBAGENT_MODEL`` puts every
#   sub-agent on it: a definition's own ``model:`` would be refused there.
# - A command is ``<plugin>:<name>`` and the model runs it with the Skill tool,
#   as it loads a skill. One that says ``disable-model-invocation: true`` is for
#   a person to type, and a build has no person: the Skill tool refuses it.
# - ``hooks/hooks.json`` runs, with ``${CLAUDE_PLUGIN_ROOT}`` the plugin's path,
#   once the settings file no longer says ``disableAllHooks``. Runtime files are
#   written with no execute bit, so a hook's command names its interpreter
#   (``node ${CLAUDE_PLUGIN_ROOT}/hooks/dispatch.mjs``).
# - ``--strict-mcp-config`` leaves a plugin's ``.mcp.json`` unread, so its
#   servers are merged into the run's MCP config file under the names Claude
#   Code itself gives them, ``plugin:<plugin>:<server>``, with
#   ``${CLAUDE_PLUGIN_ROOT}`` resolved; their tools are then
#   ``mcp__plugin_<plugin>_<server>__<tool>``, the same as without the flag, and
#   the allowlist names each server. A server that cannot start (the sandbox
#   reaches the model gateway and nothing else) is ``failed`` or ``pending`` in
#   the session and the build goes on: ``MCP_TIMEOUT`` does not hold the first
#   turn back.
PLUGINS_DIR_NAME = "_plugins"
PLUGINS_ROOT = "/runtime-home/plugins"
PLUGIN_ROOT_VARIABLE = "${CLAUDE_PLUGIN_ROOT}"
AGENT_TOOL = "Agent"
_PLUGIN_HIDDEN_ENTRIES = frozenset({".claude-plugin", ".mcp.json"})
_NAME_RE = re.compile(r"^name:\s*(.+?)\s*$", re.MULTILINE)
_USER_ONLY_RE = re.compile(r"^disable-model-invocation:\s*true\s*$", re.MULTILINE)
_MCP_NAME_CHARS_RE = re.compile(r"[^a-zA-Z0-9_-]")
_DIRECTIVE_DESCRIPTION_CHARS = 200


def _skill_description(content: str) -> str:
    """The ``description:`` of a SKILL.md's frontmatter, unquoted; empty when none."""
    front = _FRONTMATTER_RE.match(content)
    if not front:
        return ""
    found = _DESCRIPTION_RE.search(front.group(1))
    if not found:
        return ""
    text = found.group(1).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        try:
            return str(json.loads(text)) if text[0] == '"' else text[1:-1]
        except json.JSONDecodeError:
            return text[1:-1]
    return text


def _skill_submit_gate(content: str) -> str | None:
    """The ``submit-gate:`` command of a SKILL.md's frontmatter, or None."""
    front = _FRONTMATTER_RE.match(content)
    found = _SUBMIT_GATE_RE.search(front.group(1)) if front else None
    return found.group(1).strip() if found else None


def _skill_files(
    skill_dir: Path,
    *,
    what: str = "skill",
    left_out: str | None = "SKILL.md",
    hidden_kept: frozenset[str] = frozenset(),
) -> dict[str, str]:
    """A skill's files other than SKILL.md, by POSIX path relative to its directory.

    A plugin's files follow the same rules (``what="plugin"``), with nothing
    left out and the hidden entries Claude Code reads kept at the plugin's root.
    """
    files: dict[str, str] = {}
    total = 0
    for current, dirs, names in os.walk(skill_dir, followlinks=False):
        here = Path(current)
        kept = hidden_kept if here == skill_dir else frozenset()
        dirs[:] = sorted(
            d
            for d in dirs
            if (not d.startswith(".") or d in kept) and d != "__pycache__"
        )
        for name in dirs:
            if (here / name).is_symlink():
                raise ValueError(f"A symbolic link is not carried: {here / name}")
        for name in sorted(names):
            path = here / name
            relative = path.relative_to(skill_dir).as_posix()
            if (name.startswith(".") and name not in kept) or relative == left_out:
                continue
            if path.is_symlink():
                raise ValueError(f"A symbolic link is not carried: {path}")
            data = path.read_bytes()
            if len(data) > SKILL_FILE_MAX_BYTES:
                raise ValueError(
                    f"A {what} file is over {SKILL_FILE_MAX_BYTES} bytes: {path}"
                )
            if b"\0" in data:
                raise ValueError(f"A {what} file must be text: {path}")
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"A {what} file must be UTF-8 text: {path}") from exc
            total += len(data)
            if total > SKILL_FILES_MAX_BYTES:
                raise ValueError(
                    f"A {what}'s files are over {SKILL_FILES_MAX_BYTES} bytes together: {skill_dir}"
                )
            files[relative] = text
    return files


def load_skills_dir(
    skills_dir: str | Path, *, required: bool = True
) -> list[dict[str, Any]]:
    """Read ``<dir>/<name>/SKILL.md`` and the files beside it, by name.

    Each skill is ``{"name", "content", "description", "files"}``, ``files``
    the skill's other text files by relative path (``scripts/audit.py``). The
    directory name is the skill's name (a kebab-case slug); a child with no
    SKILL.md is skipped, ``_plugins`` among them. An empty directory is an
    error: a skills arm with no skills is a misconfiguration, not the baseline.
    ``required=False`` is for a directory that carries plugin parts and no skill.
    """
    root = Path(skills_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Skills directory not found: {root}")
    skills: list[dict[str, Any]] = []
    for child in sorted(root.iterdir()):
        skill_md = child / "SKILL.md"
        if not child.is_dir() or not skill_md.is_file():
            continue
        if not _SKILL_NAME_RE.match(child.name):
            raise ValueError(
                f"Skill directory name must be a kebab-case slug: {child.name}"
            )
        content = skill_md.read_text()
        skills.append(
            {
                "name": child.name,
                "content": content,
                "description": _skill_description(content),
                "files": _skill_files(child),
                "submit_gate": _skill_submit_gate(content),
            }
        )
    if not skills and required:
        raise ValueError(f"No <name>/SKILL.md below the skills directory: {root}")
    return skills


def _frontmatter_name(content: str) -> str:
    """The ``name:`` of a definition's frontmatter, unquoted; empty when none."""
    front = _FRONTMATTER_RE.match(content)
    found = _NAME_RE.search(front.group(1)) if front else None
    return found.group(1).strip().strip("\"'") if found else ""


def _definitions(
    plugin: str, files: dict[str, str], folder: str, *, named_by_frontmatter: bool
) -> list[dict[str, Any]]:
    """The ``.md`` definitions below one folder of a plugin, under the names Claude Code gives them."""
    found: list[dict[str, Any]] = []
    for relative, content in files.items():
        parts = relative.split("/")
        if parts[0] != folder or len(parts) < 2 or not parts[-1].endswith(".md"):
            continue
        own = (_frontmatter_name(content) if named_by_frontmatter else "") or parts[-1][
            : -len(".md")
        ]
        front = _FRONTMATTER_RE.match(content)
        found.append(
            {
                "name": ":".join([plugin, *parts[1:-1], own]),
                "description": _skill_description(content),
                "user_only": bool(front and _USER_ONLY_RE.search(front.group(1))),
            }
        )
    return sorted(found, key=lambda definition: definition["name"])


def _json_file(files: dict[str, str], relative: str, plugin_dir: Path) -> Any:
    """One of a plugin's JSON files, parsed; None when the plugin has none."""
    if relative not in files:
        return None
    try:
        return json.loads(files[relative])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Not JSON: {plugin_dir / relative}") from exc


def _resolved(value: Any, plugin_path: str) -> Any:
    """A server's configuration with ``${CLAUDE_PLUGIN_ROOT}`` as the plugin's path in the container."""
    if isinstance(value, str):
        return value.replace(PLUGIN_ROOT_VARIABLE, plugin_path)
    if isinstance(value, list):
        return [_resolved(item, plugin_path) for item in value]
    if isinstance(value, dict):
        return {key: _resolved(item, plugin_path) for key, item in value.items()}
    return value


def load_plugins_dir(skills_dir: str | Path) -> list[dict[str, Any]]:
    """Read ``<dir>/_plugins/<plugin>/``: what each plugin carries besides its skills.

    Each plugin is ``{"name", "path", "version", "files", "agents", "commands",
    "hook_events", "mcp_servers"}``: ``name`` the manifest's, else the
    directory's; ``path`` where it rides in the container; ``files`` every text
    file by relative path, under a skill's rules; ``agents`` and ``commands``
    the definitions as ``{"name", "description", "user_only"}``, named as the
    session names them; ``hook_events`` the events of ``hooks/hooks.json``;
    ``mcp_servers`` the servers of ``.mcp.json`` by the name the session gives
    each, ``${CLAUDE_PLUGIN_ROOT}`` resolved. No ``_plugins``: an empty list.
    """
    root = Path(skills_dir) / PLUGINS_DIR_NAME
    if root.is_symlink():
        raise ValueError(f"A symbolic link is not carried: {root}")
    if not root.is_dir():
        return []
    plugins: list[dict[str, Any]] = []
    for child in sorted(root.iterdir()):
        if child.is_symlink():
            raise ValueError(f"A symbolic link is not carried: {child}")
        if not child.is_dir() or child.name.startswith("."):
            continue
        if not _SKILL_NAME_RE.match(child.name):
            raise ValueError(
                f"Plugin directory name must be a kebab-case slug: {child.name}"
            )
        if (child / "skills").exists():
            raise ValueError(
                f"A plugin's skills ride at the top of the skills directory, not in {child / 'skills'}"
            )
        files = _skill_files(
            child, what="plugin", left_out=None, hidden_kept=_PLUGIN_HIDDEN_ENTRIES
        )
        if not files:
            raise ValueError(f"A plugin directory with no file: {child}")
        manifest = _json_file(files, ".claude-plugin/plugin.json", child)
        if manifest is not None and not isinstance(manifest, dict):
            raise ValueError(f"A plugin manifest must be a JSON object: {child}")
        name = (manifest or {}).get("name", child.name)
        if not isinstance(name, str) or not _SKILL_NAME_RE.match(name):
            raise ValueError(f"Plugin name must be a kebab-case slug: {name!r}")
        if SKILLS_PLUGIN_NAME in (name, child.name):
            raise ValueError(
                f"The plugin name {SKILLS_PLUGIN_NAME!r} is the one the run's skills ride as"
            )
        if any(name == other["name"] for other in plugins):
            raise ValueError(f"Two plugins are named {name!r}")
        path = f"{PLUGINS_ROOT}/{child.name}"
        hooks = _json_file(files, "hooks/hooks.json", child)
        if hooks is not None and not isinstance(hooks, dict):
            raise ValueError(f"hooks.json must be a JSON object: {child}")
        events = (hooks or {}).get("hooks", hooks or {})
        servers = _json_file(files, ".mcp.json", child)
        if isinstance(servers, dict) and isinstance(servers.get("mcpServers"), dict):
            servers = servers["mcpServers"]
        if servers is not None and not (
            isinstance(servers, dict)
            and all(isinstance(server, dict) for server in servers.values())
        ):
            raise ValueError(f".mcp.json must map each server to an object: {child}")
        version = (manifest or {}).get("version")
        plugins.append(
            {
                "name": name,
                "path": path,
                "version": version if isinstance(version, str) else None,
                "files": files,
                "agents": _definitions(name, files, "agents", named_by_frontmatter=True),
                "commands": _definitions(
                    name, files, "commands", named_by_frontmatter=False
                ),
                "hook_events": sorted(events) if isinstance(events, dict) else [],
                "mcp_servers": {
                    f"plugin:{name}:{server}": _resolved(config, path)
                    for server, config in sorted((servers or {}).items())
                },
            }
        )
    return plugins


def plugins_metadata(plugins: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """What each plugin handed the session, for the run's record, under the session's names."""
    record: dict[str, dict[str, Any]] = {}
    for plugin in plugins:
        commands = plugin["commands"]
        user_only = [c["name"] for c in commands if c["user_only"]]
        record[plugin["name"]] = {
            "path": plugin["path"],
            **({"version": plugin["version"]} if plugin["version"] else {}),
            "agents": [agent["name"] for agent in plugin["agents"]],
            "commands": [c["name"] for c in commands if not c["user_only"]],
            **({"commands_user_only": user_only} if user_only else {}),
            "hook_events": list(plugin["hook_events"]),
            "mcp_servers": list(plugin["mcp_servers"]),
            "files": sorted(plugin["files"]),
        }
    return record


def mcp_server_rule(server: str) -> str:
    """The allowlist entry for every tool of one MCP server, as Claude Code spells the server's name in a tool's."""
    return f"mcp__{_MCP_NAME_CHARS_RE.sub('_', server)}"


def skill_files_metadata(skills: list[dict[str, Any]]) -> dict[str, list[str]]:
    """The files each skill carries beside its SKILL.md, for the run's record; only skills with some."""
    return {
        skill["name"]: sorted(skill["files"])
        for skill in skills
        if skill.get("files")
    }


def submit_gate_hooks(skills: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The settings' ``hooks`` for the skills' submit gates, or None when no skill names one."""
    gates = [
        {
            "type": "command",
            "command": f"cd {SKILLS_PLUGIN_PATH}/skills/{skill['name']} && {skill['submit_gate']}",
            "timeout": 60,
        }
        for skill in skills
        if skill.get("submit_gate")
    ]
    if not gates:
        return None
    return {"PreToolUse": [{"matcher": SUBMIT_TOOL, "hooks": gates}]}


def _named(definition: dict[str, Any]) -> str:
    """One line of the directive for an agent definition or a command."""
    description = definition.get("description") or ""
    if len(description) > _DIRECTIVE_DESCRIPTION_CHARS:
        description = description[:_DIRECTIVE_DESCRIPTION_CHARS].rstrip() + "…"
    return f"- `{definition['name']}`" + (f" — {description}" if description else "")


def render_skills_directive(
    skills: list[dict[str, Any]], plugins: list[dict[str, Any]] | None = None
) -> str:
    """The CLAUDE.md that tells the Developer to load the run's skills first.

    With plugins it also names their agent definitions and commands, so that
    the Developer knows they are there; it does not say to use them. Empty when
    there is nothing to say.
    """
    lines: list[str] = []
    if skills:
        count = "One skill was" if len(skills) == 1 else f"{len(skills)} skills were"
        lines += [
            "# Skills for this build",
            "",
            f"{count} handed to you for this build, as the plugin `{SKILLS_PLUGIN_NAME}`. "
            "Before you read the kit, load each one with the Skill tool and follow what it says:",
            "",
        ]
        for skill in skills:
            description = skill.get("description") or ""
            lines.append(
                f"- `{SKILLS_PLUGIN_NAME}:{skill['name']}`"
                + (f" — {description}" if description else "")
            )
        if len(skills) > 1:
            lines += [
                "",
                f"Load all {len(skills)} before anything else: your first message is "
                f"{len(skills)} Skill calls, one for each skill. Each was picked for this "
                "build, so load it whatever its description says about when to use it, "
                "and even when another one is a complete procedure. A description is not "
                "the skill's text, and a skill that is not loaded is never read. Then "
                "follow them together.",
            ]
        lines += [
            "",
            "Keep working until you have called `hyper_tau.submit`: a session that ends "
            "without it is scored as the workspace stands, and the time left in your "
            "budget is lost.",
            "",
        ]
    agents = [agent for plugin in plugins or [] for agent in plugin["agents"]]
    commands = [
        command
        for plugin in plugins or []
        for command in plugin["commands"]
        if not command["user_only"]
    ]
    if agents or commands:
        lines += [
            "# Agents and commands for this build",
            "",
            "These were handed to you for this build. Use one when the work calls "
            "for it; you do not have to use them.",
            "",
        ]
        if agents:
            lines += [
                "Agent definitions, to delegate to with the Agent tool:",
                "",
                *[_named(agent) for agent in agents],
                "",
            ]
        if commands:
            lines += [
                "Commands, to run with the Skill tool:",
                "",
                *[_named(command) for command in commands],
                "",
            ]
    return "\n".join(lines)


def skill_name_of_call(arguments: Any) -> str | None:
    """The skill a ``Skill`` tool call names, without the plugin prefix."""
    if not isinstance(arguments, dict):
        return None
    raw = arguments.get("skill") or arguments.get("command") or arguments.get("name")
    if not isinstance(raw, str) or not raw.strip():
        return None
    name = raw.strip().lstrip("/")
    if ":" in name:
        name = name.rsplit(":", 1)[1]
    return name.split()[0] if name else None


def _text_content(value: Any) -> str:
    """Normalize Claude's string-or-block tool output for trajectory storage."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or item))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if value is None:
        return ""
    return str(value)


def _enabled_tools(
    *,
    include_client_tool: bool,
    include_live_experiment_tool: bool = False,
    include_sample_scenarios_tool: bool = False,
    include_skill_tool: bool = False,
    include_agent_tool: bool = False,
    plugin_mcp_servers: tuple[str, ...] = (),
) -> list[str]:
    """Return the complete native tool allowlist for one task shape."""
    tools = [
        "Bash",
        "Edit",
        "Glob",
        "Grep",
        "NotebookEdit",
        "Read",
        "Write",
        "mcp__hyper_tau__run_local_test",
        "mcp__hyper_tau__submit",
    ]
    if include_live_experiment_tool:
        tools.append("mcp__hyper_tau__run_live_experiment")
    if include_sample_scenarios_tool:
        tools.append("mcp__hyper_tau__run_sample_scenarios")
    if include_client_tool:
        tools.append("mcp__hyper_tau__talk_to_client")
    if include_skill_tool:
        tools.append("Skill")
    if include_agent_tool:
        tools.append(AGENT_TOOL)
    tools.extend(mcp_server_rule(server) for server in plugin_mcp_servers)
    return tools


class ClaudeCodeSandboxBuilder(NativeSandboxBuilder):
    """Run the pinned Claude Code CLI as an autonomous JSONL process."""

    harness_name = "claude-code"
    harness_version = CLAUDE_CODE_HARNESS_VERSION
    model_gateway_provider = "anthropic"
    runtime_config_path = "/runtime-home/.claude/settings.json"
    mcp_config_path = "/runtime-home/claude-mcp.json"

    def __init__(self, llm: str, *, skills_dir: str | Path | None = None, **kwargs):
        super().__init__(llm, **kwargs)
        self.plugins = load_plugins_dir(skills_dir) if skills_dir is not None else []
        self.skills = (
            load_skills_dir(skills_dir, required=not self.plugins)
            if skills_dir is not None
            else []
        )
        # One entry per Claude Code turn that ended (a `result` frame): the turn
        # loop starts the next one until submit, the budget or the allowance.
        self._turn_ends: list[dict] = []
        # One per Claude Code process the loop started (an `init` frame): the
        # last turn usually ends with the host's kill on submit and has no end.
        self._turns_started = 0

    def harness_config_metadata(self) -> dict:
        metadata = {
            "interface": "print-stream-json",
            "permission_mode": "dontAsk-with-explicit-allowlist",
            "bare_mode": not (self.skills or self.plugins),
            "web_tools": "removed",
            "browser": False,
            "background_tasks": False,
            "built_in_agents": False,
            "memory": False,
            # Sessions are kept below the private runtime home (a tmpfs that
            # goes with the container) so that the turn loop's `--continue`
            # finds the one it continues; nothing of them leaves the run.
            "session_persistence": "runtime-home-only",
            "turn_loop": True,
            "nonessential_traffic": False,
            "model_gateway": "provider-only/per-run/model-scoped",
            "gateway_token_inherited_by_shell": True,
            "inner_bash_sandbox": "disabled/container-is-boundary",
            "mcp_servers": ["hyper_tau", *self._plugin_mcp_servers()],
        }
        if self.plugins:
            metadata["plugins"] = plugins_metadata(self.plugins)
            if self._has_agents():
                metadata["subagent_model"] = self.llm
        if self.skills:
            metadata["skills"] = [skill["name"] for skill in self.skills]
            metadata["skills_plugin"] = SKILLS_PLUGIN_NAME
            metadata["skills_directive"] = "claude-md"
            skill_files = skill_files_metadata(self.skills)
            if skill_files:
                metadata["skill_files"] = skill_files
            gates = {s["name"]: s["submit_gate"] for s in self.skills if s.get("submit_gate")}
            if gates:
                metadata["submit_gates"] = gates
        return metadata

    def _has_agents(self) -> bool:
        return any(plugin["agents"] for plugin in self.plugins)

    def _plugin_mcp_servers(self) -> tuple[str, ...]:
        return tuple(
            server for plugin in self.plugins for server in plugin["mcp_servers"]
        )

    def _carried_tools(self) -> dict[str, Any]:
        """What the run's skills and plugins add to the tool allowlist."""
        return {
            # A plugin's command is run with the Skill tool, as a skill is loaded.
            "include_skill_tool": bool(self.skills)
            or any(plugin["commands"] for plugin in self.plugins),
            "include_agent_tool": self._has_agents(),
            "plugin_mcp_servers": self._plugin_mcp_servers(),
        }

    def reset_build_state(self) -> None:
        self._turn_ends = []
        self._turns_started = 0

    def build_turns(self) -> list[dict] | None:
        return list(self._turn_ends)

    def turns_started(self) -> int | None:
        return self._turns_started

    def skills_read(self, steps) -> list[str] | None:
        """The run's skills the Developer loaded with the Skill tool, in order of first load."""
        if not self.skills:
            return None
        wanted = {skill["name"] for skill in self.skills}
        read: list[str] = []
        for step in steps:
            for call in getattr(step, "tool_calls", None) or []:
                if not isinstance(call, dict) or call.get("name") != "Skill":
                    continue
                arguments = call.get("arguments")
                name = skill_name_of_call(arguments)
                if name in wanted and name not in read and not self._of_a_plugin(arguments):
                    read.append(name)
        return read

    def _of_a_plugin(self, arguments: dict) -> bool:
        """Whether a ``Skill`` call names a carried plugin's command, not one of the run's skills."""
        raw = arguments.get("skill") or arguments.get("command") or arguments.get("name")
        prefix, colon, _ = str(raw).strip().lstrip("/").partition(":")
        return bool(colon) and any(prefix == plugin["name"] for plugin in self.plugins)

    def runtime_environment(self, broker) -> dict[str, str]:
        environment = super().runtime_environment(broker)
        callback_timeout_ms = str(
            int(float(environment["TAU2_CALLBACK_TIMEOUT_SECONDS"]) * 1000)
        )
        environment.update(
            {
                "CLAUDE_CODE_DISABLE_ARTIFACT": "1",
                "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
                "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
                "CLAUDE_CODE_DISABLE_BUNDLED_SKILLS": "1",
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                "CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL": "1",
                "CLAUDE_CODE_DISABLE_WORKFLOWS": "1",
                "CLAUDE_CODE_SKIP_PROMPT_HISTORY": "1",
                # The construction container is itself the isolation boundary
                # (internal-only network, no egress except the model gateway
                # sidecar, cap-scoped). Claude Code's inner Bash sandbox nests
                # a bubblewrap user namespace, which the container's default
                # security context forbids ("No permissions to create new
                # namespace"), so every Bash call failed preflight ("socat not
                # installed" / bwrap). Tell Claude Code it is already sandboxed
                # so Bash runs directly, matching the Codex lane's
                # danger-full-access model. Env scrubbing also requires
                # bubblewrap, so it is disabled; the per-run gateway token it
                # would scrub is model-scoped and unexfiltratable (no egress).
                "CLAUDE_CODE_SANDBOXED": "1",
                "IS_SANDBOX": "1",
                "CLAUDE_CODE_SUBPROCESS_ENV_SCRUB": "0",
                "CLAUDE_AGENT_SDK_DISABLE_BUILTIN_AGENTS": "1",
                "DISABLE_LOGIN_COMMAND": "1",
                "DISABLE_LOGOUT_COMMAND": "1",
                "DISABLE_UPDATES": "1",
                "DISABLE_UPGRADE_COMMAND": "1",
                "MCP_TIMEOUT": callback_timeout_ms,
            }
        )
        if self._has_agents():
            # The model gateway takes the run's model and no other: without this
            # a definition that names its own model is refused there.
            environment["CLAUDE_CODE_SUBAGENT_MODEL"] = self.llm
        thinking = self.llm_args.get("thinking")
        if isinstance(thinking, dict) and thinking.get("budget_tokens"):
            environment["MAX_THINKING_TOKENS"] = str(thinking["budget_tokens"])
        return environment

    def model_gateway_environment(self, spec: ModelGatewaySpec) -> dict[str, str]:
        return {
            "ANTHROPIC_AUTH_TOKEN": spec.token,
            "ANTHROPIC_BASE_URL": spec.base_url,
        }

    def render_runtime_config(
        self,
        *,
        include_client_tool: bool,
        include_live_experiment_tool: bool = False,
        include_sample_scenarios_tool: bool = False,
    ) -> str:
        enabled_tools = _enabled_tools(
            include_client_tool=include_client_tool,
            include_live_experiment_tool=include_live_experiment_tool,
            include_sample_scenarios_tool=include_sample_scenarios_tool,
            **self._carried_tools(),
        )
        hooks = submit_gate_hooks(self.skills)
        plugin_hooks = any(plugin["hook_events"] for plugin in self.plugins)
        return json.dumps(
            {
                "autoMemoryEnabled": False,
                # A plugin's hooks.json runs only with this off; the submit gate
                # stays the one hook the settings file itself names.
                "disableAllHooks": hooks is None and not plugin_hooks,
                **({"hooks": hooks} if hooks else {}),
                "disableArtifact": True,
                "disableBundledSkills": True,
                "disableClaudeAiConnectors": True,
                "disableRemoteControl": True,
                "disableWorkflows": True,
                "enableAllProjectMcpServers": False,
                "includeGitInstructions": True,
                "permissions": {
                    "allow": enabled_tools,
                    "defaultMode": "dontAsk",
                    "deny": ["WebSearch", "WebFetch"],
                },
            },
            indent=2,
            sort_keys=True,
        )

    def runtime_files(
        self,
        *,
        include_client_tool: bool,
        include_live_experiment_tool: bool = False,
        include_sample_scenarios_tool: bool = False,
    ) -> dict[str, str]:
        files = {
            self.runtime_config_path: self.render_runtime_config(
                include_client_tool=include_client_tool,
                include_live_experiment_tool=include_live_experiment_tool,
                include_sample_scenarios_tool=include_sample_scenarios_tool,
            ),
            self.mcp_config_path: json.dumps(
                {
                    "mcpServers": {
                        "hyper_tau": {
                            "type": "stdio",
                            "command": "python",
                            "args": ["-m", "tau2.hyper.sandbox.callback_mcp"],
                            "env": {
                                "TAU2_CALLBACK_DIR": "${TAU2_CALLBACK_DIR}",
                                "TAU2_CALLBACK_TOKEN": "${TAU2_CALLBACK_TOKEN}",
                                "TAU2_CALLBACK_TIMEOUT_SECONDS": (
                                    "${TAU2_CALLBACK_TIMEOUT_SECONDS}"
                                ),
                                "TAU2_CLIENT_TOOL_ENABLED": (
                                    "${TAU2_CLIENT_TOOL_ENABLED}"
                                ),
                                "TAU2_LIVE_EXPERIMENT_TOOL_ENABLED": (
                                    "${TAU2_LIVE_EXPERIMENT_TOOL_ENABLED}"
                                ),
                                "TAU2_SAMPLE_SCENARIOS_TOOL_ENABLED": (
                                    "${TAU2_SAMPLE_SCENARIOS_TOOL_ENABLED}"
                                ),
                            },
                        },
                        # --strict-mcp-config reads this file alone, so a
                        # plugin's servers are named here.
                        **{
                            server: config
                            for plugin in self.plugins
                            for server, config in plugin["mcp_servers"].items()
                        },
                    }
                },
                indent=2,
                sort_keys=True,
            ),
        }
        files.update(self.skills_plugin_files())
        return files

    def skills_plugin_files(self) -> dict[str, str]:
        """The plugin that carries the run's skills (a manifest, each SKILL.md and the files beside it), the files of each plugin the run carries, and the CLAUDE.md that says what was handed."""
        files: dict[str, str] = {}
        directive = render_skills_directive(self.skills, self.plugins)
        if directive:
            files[SKILLS_DIRECTIVE_PATH] = directive
        if self.skills:
            files[f"{SKILLS_PLUGIN_PATH}/.claude-plugin/plugin.json"] = json.dumps(
                {
                    "name": SKILLS_PLUGIN_NAME,
                    "description": "Skills handed to the Developer for this run",
                    "version": "0.0.0",
                },
                indent=2,
                sort_keys=True,
            )
        for skill in self.skills:
            base = f"{SKILLS_PLUGIN_PATH}/skills/{skill['name']}"
            files[f"{base}/SKILL.md"] = skill["content"]
            for relative, text in skill.get("files", {}).items():
                files[f"{base}/{relative}"] = text
        for plugin in self.plugins:
            for relative, text in plugin["files"].items():
                files[f"{plugin['path']}/{relative}"] = text
        return files

    def harness_command(self) -> list[str]:
        enabled_tools = _enabled_tools(
            include_client_tool=self._client_ctx is not None,
            include_live_experiment_tool=self._live_experiment_ctx is not None,
            include_sample_scenarios_tool=self._sample_scenarios_ctx is not None,
            **self._carried_tools(),
        )
        enabled_tool_list = ",".join(enabled_tools)
        carries = bool(self.skills or self.plugins)

        # Claude Code 2.1.219 loads no skill under --bare, from a --plugin-dir or
        # the home's skills directory alike (checked in the image: the model
        # reports NONE with it and both skills without it). A skills run drops
        # bare mode; the settings file still disables hooks (all but a skill's
        # submit gate, on the submit tool alone), and the env keeps
        # memory, updates and nonessential traffic off. Nor does --bare load the
        # CLAUDE.md below the runtime home (checked the same way), which is why
        # the skills directive needs the skills run's mode. A run that carries
        # plugin parts is in the same mode: --bare skips hooks and plugins.
        #
        # `claude --print` exits when the turn ends, that is when a reply comes
        # back with no tool call: a model that stops early, or a reply cut short
        # by the gateway, ends the build with the budget unspent and no submit
        # (both Developers of the run of 2026-11-06 ended that way at 126 s). The
        # turn loop driver, which OpenCode already runs under, hands the prompt
        # as the first turn and continues the session until the host ends the
        # process (submit, the wall clock, the step budget), a non-zero exit, or
        # its allowance; `--continue` finds the session because persistence stays
        # on below the private runtime home.
        command = [
            "python",
            "-m",
            "tau2.hyper.harnesses.turn_loop_driver",
            "claude",
            "--print",
            *([] if carries else ["--bare"]),
            "--output-format",
            "stream-json",
            "--verbose",
            "--model",
            self.llm,
        ]
        reasoning_effort = self.llm_args.get("reasoning_effort")
        if reasoning_effort and reasoning_effort != "none":
            command.extend(["--effort", str(reasoning_effort)])
        command.extend(
            [
                "--permission-mode",
                "dontAsk",
                "--tools",
                enabled_tool_list,
                "--allowedTools",
                enabled_tool_list,
                "--disallowedTools",
                "WebSearch,WebFetch",
                "--mcp-config",
                self.mcp_config_path,
                "--strict-mcp-config",
                "--settings",
                self.runtime_config_path,
                "--no-chrome",
            ]
        )
        if self.skills:
            command.extend(["--plugin-dir", SKILLS_PLUGIN_PATH])
        for plugin in self.plugins:
            command.extend(["--plugin-dir", plugin["path"]])
        if not carries:
            command.append("--disable-slash-commands")
        return command

    def normalize_event(self, event: NativeProcessEvent) -> list[BuildStep]:
        if event.channel != "stdout":
            return []
        try:
            frame = json.loads(event.text)
        except json.JSONDecodeError:
            return []

        frame_type = frame.get("type")
        if frame_type == "system" and frame.get("subtype") == "init":
            self._turns_started += 1
            return []
        if frame_type == "assistant":
            return self._assistant_steps(frame)
        if frame_type == "user":
            return self._tool_result_steps(frame)
        if frame_type == "result":
            self._turn_ends.append(
                {
                    "ended_at": event.elapsed_seconds,
                    "subtype": frame.get("subtype"),
                    "is_error": bool(frame.get("is_error", False)),
                    "num_turns": frame.get("num_turns"),
                }
            )
            result_text = frame.get("result")
            if result_text:
                return [
                    BuildStep(
                        step_idx=0,
                        role="assistant",
                        content=_text_content(result_text),
                    )
                ]
        return []

    @staticmethod
    def _assistant_steps(frame: dict) -> list[BuildStep]:
        blocks = frame.get("message", {}).get("content", [])
        text_parts: list[str] = []
        calls: list[dict] = []
        for block in blocks if isinstance(blocks, list) else []:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type in {"text", "thinking"}:
                text = block.get("text") or block.get("thinking")
                if text:
                    text_parts.append(str(text))
            elif block_type == "tool_use":
                calls.append(
                    {
                        "id": block.get("id", ""),
                        "name": block.get("name", ""),
                        "arguments": block.get("input", {}),
                    }
                )
        if not text_parts and not calls:
            return []
        return [
            BuildStep(
                step_idx=0,
                role="assistant",
                content="\n".join(text_parts) or None,
                tool_calls=calls or None,
            )
        ]

    @staticmethod
    def _tool_result_steps(frame: dict) -> list[BuildStep]:
        blocks = frame.get("message", {}).get("content", [])
        results: list[dict] = []
        for block in blocks if isinstance(blocks, list) else []:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            results.append(
                {
                    "id": block.get("tool_use_id", ""),
                    "name": "tool_result",
                    "result": _text_content(block.get("content")),
                    "is_error": bool(block.get("is_error", False)),
                }
            )
        if not results:
            return []
        return [BuildStep(step_idx=0, role="tool", tool_results=results)]
