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

from config import HFI_POLL_INTERVAL, HFI_POLL_TIMEOUT, WSL_DISTRO
from utils import wsl_exec, read_wsl_file


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


def check_turn_complete(session_dir: str, turn_num: int) -> dict:
    """
    Check if a turn's trajectories are complete.
    Returns {"complete": bool, "a_done": bool, "b_done": bool}.
    """
    result_idx = turn_num - 1
    a_path = f"{session_dir}/result-{result_idx}-A.json"
    b_path = f"{session_dir}/result-{result_idx}-B.json"

    code_a, _, _ = wsl_exec(f"test -f '{a_path}' && echo yes || echo no")
    code_b, _, _ = wsl_exec(f"test -f '{b_path}' && echo yes || echo no")

    a_done = "yes" in (wsl_exec(f"test -f '{a_path}' && echo yes || echo no")[1])
    b_done = "yes" in (wsl_exec(f"test -f '{b_path}' && echo yes || echo no")[1])

    return {"complete": a_done and b_done, "a_done": a_done, "b_done": b_done}


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
                status_callback(f"Turn {turn_num} complete!")
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
    result_idx = turn_num - 1
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


def extract_trace(jsonl_path: str, max_entries: int = 50) -> str:
    """Extract assistant messages from a JSONL session file."""
    content = read_wsl_file(jsonl_path)
    if not content:
        return ""

    entries = []
    for line in content.strip().split("\n"):
        if not line.strip():
            continue
        try:
            obj = json.loads(line.strip())
        except json.JSONDecodeError:
            continue

        if obj.get("type") != "assistant":
            continue

        for blk in obj.get("message", {}).get("content", []):
            if blk.get("type") == "text":
                entries.append("[ASSISTANT]: " + blk["text"][:1500])
            elif blk.get("type") == "tool_use":
                inp = blk.get("input", {})
                cmd = inp.get("command", inp.get("description", str(inp)[:200]))
                entries.append(f"[TOOL:{blk['name']}]: {str(cmd)[:400]}")

        if len(entries) >= max_entries:
            break

    return "\n".join(entries)


def extract_diffs_from_worktrees(session_dir: str, turn_num: int) -> tuple[str, str]:
    """
    Extract git diffs from HFI worktrees.
    Returns (diff_a, diff_b).
    """
    code, repo_path, _ = wsl_exec(
        "ls -d ~/.cache/claude-hfi/*/A 2>/dev/null | head -1"
    )
    if code != 0 or not repo_path.strip():
        return "", ""

    base_path = repo_path.strip().rsplit("/A", 1)[0]
    worktree_a = f"{base_path}/A"
    worktree_b = f"{base_path}/B"

    diff_a = _get_worktree_diff(worktree_a)
    diff_b = _get_worktree_diff(worktree_b)

    return diff_a, diff_b


def _get_worktree_diff(worktree_path: str) -> str:
    """Get git diff from a worktree."""
    code, stat_out, _ = wsl_exec(
        f"cd '{worktree_path}' && git diff --stat HEAD~1..HEAD 2>/dev/null"
    )
    code2, diff_out, _ = wsl_exec(
        f"cd '{worktree_path}' && git diff HEAD~1..HEAD 2>/dev/null"
    )

    result = ""
    if code == 0 and stat_out.strip():
        result += f"=== DIFF STAT ===\n{stat_out}\n"
    if code2 == 0 and diff_out.strip():
        result += f"=== FULL DIFF ===\n{diff_out}\n"

    if not result:
        code3, diff_out3, _ = wsl_exec(
            f"cd '{worktree_path}' && git diff HEAD 2>/dev/null"
        )
        if code3 == 0:
            result = diff_out3

    return result


def get_hfi_launch_commands(repo_dir: str, is_continue: bool = False) -> list[str]:
    """Generate the commands the user needs to run to launch/continue HFI."""
    cmds = []
    if not is_continue:
        cmds.append("tmux new-session -s hfi")
        cmds.append(f"cd {repo_dir}")
        cmds.append("./claude-hfi --tmux")
    else:
        cmds.append(f"cd {repo_dir}")
        cmds.append("./claude-hfi --tmux --continue")
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
