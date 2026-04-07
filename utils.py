"""Shared utility functions - auto-detects Windows vs WSL environment."""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from config import WSL_DISTRO, DATA_DIR

# Detect if we're already running inside WSL/Linux
_INSIDE_WSL = sys.platform.startswith("linux")


def wsl_exec(cmd: str, timeout: int = 300) -> tuple[int, str, str]:
    """Run a shell command. Uses WSL wrapper on Windows, direct bash inside WSL."""
    if _INSIDE_WSL:
        full_cmd = ["bash", "-c", cmd]
    else:
        full_cmd = ["wsl", "-d", WSL_DISTRO, "--", "bash", "-c", cmd]
    try:
        result = subprocess.run(
            full_cmd, capture_output=True, timeout=timeout,
            encoding="utf-8", errors="replace",
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Command timed out after {timeout}s"
    except Exception as e:
        return -1, "", str(e)


def wsl_exec_script(script_content: str, timeout: int = 600) -> tuple[int, str, str]:
    """Write a temp script and execute it."""
    if _INSIDE_WSL:
        tmp_path = "/tmp/marlin_auto_tmp.sh"
        Path(tmp_path).write_text(script_content, encoding="utf-8", newline="\n")
        return wsl_exec(f"chmod +x '{tmp_path}' && bash '{tmp_path}'", timeout)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".sh", delete=False,
        encoding="utf-8", newline="\n",
    ) as f:
        f.write(script_content)
        win_path = f.name

    try:
        wsl_path = _win_to_wsl(win_path)
        return wsl_exec(f"chmod +x '{wsl_path}' && bash '{wsl_path}'", timeout)
    finally:
        try:
            os.unlink(win_path)
        except OSError:
            pass


def _win_to_wsl(win_path: str) -> str:
    """Convert a Windows path to WSL /mnt/c/... path."""
    drive = win_path[0].lower()
    rest = win_path[2:].replace("\\", "/")
    return f"/mnt/{drive}{rest}"


def read_wsl_file(wsl_path: str) -> str:
    """Read a file from WSL filesystem."""
    code, out, _ = wsl_exec(f"cat '{wsl_path}' 2>/dev/null")
    return out if code == 0 else ""


def write_wsl_file(wsl_path: str, content: str):
    """Write content to a file on the Linux filesystem."""
    if _INSIDE_WSL:
        os.makedirs(os.path.dirname(wsl_path), exist_ok=True)
        Path(wsl_path).write_text(content, encoding="utf-8", newline="\n")
        return

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False,
        encoding="utf-8", newline="\n",
    ) as f:
        f.write(content)
        win_path = f.name

    try:
        wsl_src = _win_to_wsl(win_path)
        wsl_exec(f"mkdir -p \"$(dirname '{wsl_path}')\" && cp '{wsl_src}' '{wsl_path}'")
    finally:
        try:
            os.unlink(win_path)
        except OSError:
            pass


def ensure_task_dir(task_name: str) -> str:
    """Create and return path to task data directory."""
    task_dir = os.path.join(DATA_DIR, task_name)
    os.makedirs(task_dir, exist_ok=True)
    return task_dir


def write_task_file(task_name: str, filename: str, content: str) -> str:
    """Write a file into the task data directory. Returns the full path."""
    task_dir = ensure_task_dir(task_name)
    fpath = os.path.join(task_dir, filename)
    Path(fpath).write_text(content, encoding="utf-8")
    return fpath


def read_task_file(task_name: str, filename: str) -> str:
    """Read a file from the task data directory."""
    fpath = os.path.join(DATA_DIR, task_name, filename)
    if os.path.exists(fpath):
        return Path(fpath).read_text(encoding="utf-8", errors="replace")
    return ""


def load_json(path: str) -> dict:
    """Safely load a JSON file."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, FileNotFoundError, OSError):
        return {}


def curl_json(url: str, timeout: int = 30) -> dict | None:
    """Fetch JSON from a URL using curl in WSL."""
    code, out, err = wsl_exec(f"curl -sL --max-time {timeout} '{url}'")
    if code != 0 or not out.strip():
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None


def curl_text(url: str, timeout: int = 30) -> str:
    """Fetch raw text from a URL using curl in WSL."""
    code, out, _ = wsl_exec(f"curl -sL --max-time {timeout} '{url}'")
    return out if code == 0 else ""
