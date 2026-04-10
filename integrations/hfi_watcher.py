"""
Passive HFI session monitor.

Watches the /tmp/claude-hfi/ directory for result files
to detect when turns complete. Read-only - never writes
to HFI's files or interacts with tmux sessions.
"""

import glob
import json
import os
import time
from pathlib import Path

from core.config import HFI_POLL_INTERVAL, HFI_POLL_TIMEOUT, WSL_DISTRO
from core.utils import wsl_exec, read_wsl_file


def find_session_dir() -> str | None:
    """Find the most recent HFI session directory."""
    code, out, _ = wsl_exec("ls -td /tmp/claude-hfi/*/ 2>/dev/null | head -1")
    if code == 0 and out.strip():
        return out.strip().rstrip("/")
    return None


def find_all_sessions() -> list[str]:
    """List all HFI session directories, newest first."""
    code, out, _ = wsl_exec("ls -td /tmp/claude-hfi/*/ 2>/dev/null")
    if code == 0 and out.strip():
        return [s.strip().rstrip("/") for s in out.strip().split("\n") if s.strip()]
    return []


def _resolve_result_idx(session_dir: str, turn_num: int) -> int:
    """Determine the actual result file index for a turn.

    HFI resets result numbering to 0 each time it's relaunched with
    ``--continue``, so turn 3 in the original task might map to
    ``result-0-*.json`` in a freshly started session.  We first try the
    canonical index (``turn_num - 1``); if that doesn't exist we scan
    for the highest-numbered result pair that does.
    """
    canonical = turn_num - 1
    check = wsl_exec(
        f"test -f '{session_dir}/result-{canonical}-A.json' && echo yes || echo no"
    )
    if "yes" in check[1]:
        return canonical

    # Scan for the highest result index present in this session
    code, out, _ = wsl_exec(
        f"ls '{session_dir}'/result-*-A.json 2>/dev/null | sort -t- -k2 -n"
    )
    if code == 0 and out.strip():
        lines = [l.strip() for l in out.strip().split("\n") if l.strip()]
        if lines:
            last = lines[-1]
            try:
                idx_str = last.rsplit("/result-", 1)[1].split("-A.json")[0]
                return int(idx_str)
            except (IndexError, ValueError):
                pass

    return canonical


def check_turn_complete(session_dir: str, turn_num: int) -> dict:
    """
    Check if a turn's trajectories are complete.
    Returns {"complete": bool, "a_done": bool, "b_done": bool, "result_idx": int}.
    """
    result_idx = _resolve_result_idx(session_dir, turn_num)
    a_path = f"{session_dir}/result-{result_idx}-A.json"
    b_path = f"{session_dir}/result-{result_idx}-B.json"

    a_done = "yes" in (wsl_exec(f"test -f '{a_path}' && echo yes || echo no")[1])
    b_done = "yes" in (wsl_exec(f"test -f '{b_path}' && echo yes || echo no")[1])

    return {
        "complete": a_done and b_done,
        "a_done": a_done,
        "b_done": b_done,
        "result_idx": result_idx,
    }


def wait_for_turn(session_dir: str, turn_num: int,
                  timeout: int = None, poll_interval: int = None,
                  status_callback=None) -> bool:
    """
    Poll until both result files exist for a turn.
    Returns True if complete, False if timeout.
    """
    if timeout is None:
        timeout = HFI_POLL_TIMEOUT
    if poll_interval is None:
        poll_interval = HFI_POLL_INTERVAL

    start = time.time()
    while time.time() - start < timeout:
        status = check_turn_complete(session_dir, turn_num)
        if status["complete"]:
            if status_callback:
                status_callback(f"Turn {turn_num} complete! (result index {status['result_idx']})")
            return True

        elapsed = int(time.time() - start)
        if status_callback:
            a_str = "done" if status["a_done"] else "running"
            b_str = "done" if status["b_done"] else "running"
            status_callback(
                f"Waiting for Turn {turn_num}... A:{a_str} B:{b_str} ({elapsed}s)"
            )
        time.sleep(poll_interval)

    return False


def get_turn_result(session_dir: str, turn_num: int, trajectory: str) -> dict:
    """Read a result JSON file for a trajectory."""
    result_idx = _resolve_result_idx(session_dir, turn_num)
    path = f"{session_dir}/result-{result_idx}-{trajectory}.json"
    content = read_wsl_file(path)
    if content:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
    return {}


def get_session_file_path(session_dir: str, turn_num: int, trajectory: str) -> str:
    """Extract the JSONL session file path from a result JSON."""
    result = get_turn_result(session_dir, turn_num, trajectory)
    return result.get("sessionFilePath", "")


def _shorten_path(path: str) -> str:
    """Strip long HFI cache prefixes to save space in traces."""
    # /home/user/.cache/claude-hfi/-home-user-marlin-tasks-.../A/modules/reindex/Foo.java
    # -> A/modules/reindex/Foo.java
    for marker in ("/A/", "/B/"):
        idx = path.find(marker)
        if idx != -1:
            return path[idx + 1:]
    # Also trim generic home paths
    for prefix in ("/home/user/", "/root/"):
        if path.startswith(prefix):
            return "~/" + path[len(prefix):]
    return path


def extract_trace(jsonl_path: str, max_entries: int = 300) -> str:
    """Extract a structured trace from a JSONL session file.

    Captures assistant reasoning (thinking blocks), text messages, tool calls,
    and tool results so feedback generation can evaluate the model's agency,
    communication, and engineering process — not just its final diff.

    Keeps tool results short (just status/errors) to maximise coverage within
    the character budget used by feedback_generator.
    """
    content = read_wsl_file(jsonl_path)
    if not content:
        return ""

    entries = []
    last_tool_name = ""

    for line in content.strip().split("\n"):
        if not line.strip():
            continue
        try:
            obj = json.loads(line.strip())
        except json.JSONDecodeError:
            continue

        etype = obj.get("type", "")
        msg = obj.get("message", {})
        blocks = msg.get("content", [])
        if isinstance(blocks, str):
            blocks = [{"type": "text", "text": blocks}]

        if etype == "assistant":
            for blk in blocks:
                bt = blk.get("type", "")
                if bt == "thinking":
                    thought = blk.get("thinking", "")[:400]
                    if thought.strip():
                        entries.append(f"[THINKING]: {thought}")
                elif bt == "text":
                    entries.append("[ASSISTANT]: " + blk["text"][:800])
                elif bt == "tool_use":
                    inp = blk.get("input", {})
                    tool_name = blk.get("name", "unknown")
                    last_tool_name = tool_name
                    if tool_name in ("Bash", "bash"):
                        raw_cmd = inp.get("command", str(inp)[:300])
                        detail = _shorten_path(raw_cmd)
                    elif tool_name in ("Read", "read"):
                        detail = _shorten_path(inp.get("file_path", str(inp)[:300]))
                    elif tool_name in ("Write", "write", "Edit", "edit"):
                        fp = _shorten_path(inp.get("file_path", ""))
                        detail = f"{fp} ({len(str(inp.get('content','')))}) chars"
                    else:
                        detail = inp.get("command", inp.get("description", str(inp)[:300]))
                    entries.append(f"[TOOL:{tool_name}]: {str(detail)[:400]}")

        elif etype == "user":
            for blk in blocks:
                if not isinstance(blk, dict):
                    continue
                bt = blk.get("type", "")
                if bt == "tool_result":
                    is_error = blk.get("is_error", False)
                    result_content = blk.get("content", "")
                    if isinstance(result_content, list):
                        result_content = " ".join(
                            b.get("text", "") for b in result_content if isinstance(b, dict)
                        )
                    result_str = str(result_content)

                    if is_error:
                        entries.append(f"[TOOL_ERROR]: {_shorten_path(result_str[:300])}")
                    elif last_tool_name in ("Read", "read"):
                        entries.append(f"[TOOL_RESULT]: ({len(result_str)} chars read)")
                    elif last_tool_name in ("Write", "write", "Edit", "edit"):
                        entries.append(f"[TOOL_RESULT]: file written")
                    else:
                        entries.append(f"[TOOL_RESULT]: {_shorten_path(result_str[:150])}")

        if len(entries) >= max_entries:
            break

    return "\n".join(entries)


def extract_diffs_from_worktrees(session_dir: str, turn_num: int) -> tuple[str, str]:
    """
    Extract git diffs from HFI worktrees.
    Returns (diff_a, diff_b).

    Uses the most recently modified worktree cache directory so multiple
    tasks don't collide.
    """
    code, repo_path, _ = wsl_exec(
        "ls -td ~/.cache/claude-hfi/*/A 2>/dev/null | head -1"
    )
    if code != 0 or not repo_path.strip():
        return "", ""

    base_path = repo_path.strip().rsplit("/A", 1)[0]
    worktree_a = f"{base_path}/A"
    worktree_b = f"{base_path}/B"

    diff_a = _get_worktree_diff(worktree_a)
    diff_b = _get_worktree_diff(worktree_b)

    return diff_a, diff_b


_DIFF_EXCLUDES = [
    "claude-hfi",
    ".claude",
    ".claude/**",
    ".gitignore",
]


def _get_worktree_diff(worktree_path: str) -> str:
    """Get git diff from a worktree, excluding setup artifacts.

    Tries multiple strategies in order:
      1. Unstaged + staged changes vs HEAD (covers both git-add'd and modified files)
      2. Last commit vs its parent (models sometimes auto-commit)
    """
    excludes = " ".join(f"':(exclude){p}'" for p in _DIFF_EXCLUDES)

    # Strategy 1a: unstaged working-tree changes
    code, stat_out, _ = wsl_exec(
        f"cd '{worktree_path}' && git diff --stat HEAD -- . {excludes} 2>/dev/null"
    )
    _, diff_out, _ = wsl_exec(
        f"cd '{worktree_path}' && git diff HEAD -- . {excludes} 2>/dev/null"
    )

    # Strategy 1b: staged (cached) changes — models may `git add` without committing
    _, staged_stat, _ = wsl_exec(
        f"cd '{worktree_path}' && git diff --cached --stat -- . {excludes} 2>/dev/null"
    )
    _, staged_diff, _ = wsl_exec(
        f"cd '{worktree_path}' && git diff --cached -- . {excludes} 2>/dev/null"
    )

    result = ""
    combined_stat = (stat_out or "").strip()
    combined_diff = (diff_out or "").strip()
    if staged_stat.strip() and staged_stat.strip() not in combined_stat:
        combined_stat += "\n" + staged_stat.strip()
    if staged_diff.strip() and staged_diff.strip() not in combined_diff:
        combined_diff += "\n" + staged_diff.strip()

    if combined_stat:
        result += f"=== DIFF STAT ===\n{combined_stat}\n"
    if combined_diff:
        result += f"=== FULL DIFF ===\n{combined_diff}\n"

    # Strategy 2: committed changes (HEAD~1..HEAD)
    if not result:
        code3, stat3, _ = wsl_exec(
            f"cd '{worktree_path}' && git diff --stat HEAD~1..HEAD -- . {excludes} 2>/dev/null"
        )
        _, diff4, _ = wsl_exec(
            f"cd '{worktree_path}' && git diff HEAD~1..HEAD -- . {excludes} 2>/dev/null"
        )
        if code3 == 0 and stat3.strip():
            result += f"=== DIFF STAT ===\n{stat3}\n"
        if diff4 and diff4.strip():
            result += f"=== FULL DIFF ===\n{diff4}\n"

    return result


def get_hfi_launch_commands(repo_dir: str, is_continue: bool = False,
                            hfi_bin: str = "") -> list[str]:
    """Generate the commands the user needs to run to launch/continue HFI."""
    binary = hfi_bin or "~/marlin-tools/claude-hfi"
    cmds = [f"cd {repo_dir}"]
    if is_continue:
        cmds.append(f"{binary} --tmux --continue")
    else:
        cmds.append(f"{binary} --tmux")
    return cmds


def get_between_turn_steps(turn_just_finished: int) -> list[str]:
    """Step-by-step checklist the user must follow between HFI turns."""
    return [
        f"1. Paste your Turn {turn_just_finished} feedback answers into HFI",
        f'2. Select "Continue conversation" when prompted',
        "3. Press Ctrl+C twice to exit HFI",
        "4. Relaunch HFI (commands shown below)",
        "5. Type /clear and press Enter to reset context",
        f"6. Paste the Turn {turn_just_finished + 1} prompt",
    ]
