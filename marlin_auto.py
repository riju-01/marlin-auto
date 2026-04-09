#!/usr/bin/env python3
"""
Marlin V3 Automation - Main Orchestrator

Entry point that drives the entire workflow:
  1. PR selection and ranking
  2. Turn 1 prompt generation
  3. Environment setup (Phase 3) - automatic via WSL
  4. Per-turn loop: wait -> extract diffs -> generate feedback -> humanize
  5. Turn 2/3 prompt generation

Supports Gemini, OpenAI, and Claude. Set your API key in .env file
or pass via command line.

Usage:
    python marlin_auto.py                          # uses .env
    python marlin_auto.py --gemini YOUR_KEY         # Gemini
    python marlin_auto.py --openai YOUR_KEY         # OpenAI
    python marlin_auto.py --anthropic YOUR_KEY      # Claude
    python marlin_auto.py --pr URL --skip-setup    # skip setup
"""

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from rich.panel import Panel
from rich import box as rbox

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MARLIN_V2, SCRIPTS_DIR, WORKSPACE, DATA_DIR, MARLIN_AUTO_DIR, HFI_LOCAL_DIR,
    INTERFACE_CODE, AI_SCORE_TARGET,
)
from utils import (
    wsl_exec, wsl_exec_script, read_wsl_file,
    write_task_file, read_task_file, ensure_task_dir,
)
from llm_client import detect_provider, provider_info
from pr_fetcher import rank_prs, fetch_pr_diff, parse_pr_url, fetch_and_rank_pr
from prompt_generator import generate_phase2_doc, format_phase2_md, generate_turn1_prompt
from turn_prompt_generator import generate_turn_prompt
from feedback_generator import generate_all_feedback, format_feedback_md, regenerate_single_field
from ai_scorer import score_field, full_validation
from hfi_watcher import (
    find_session_dir, wait_for_turn, extract_diffs_from_worktrees,
    get_session_file_path, extract_trace, get_hfi_launch_commands,
    get_between_turn_steps,
)
from tui import (
    console, print_header, print_phase, print_status, print_success,
    print_warning, print_error, get_pr_urls, display_pr_rankings,
    display_prompt, get_edited_prompt, display_setup_progress,
    display_hfi_commands, display_waiting_for_turn, display_feedback_answers,
    display_feedback_file_path, ask_continue_or_view, ask_field_number, wait_for_user,
    display_between_turns, display_completion,
)


_TEXTAREA_KEYS = [
    "expected_model_response",
    "model_a_solution_quality", "model_a_agency", "model_a_communication",
    "model_b_solution_quality", "model_b_agency", "model_b_communication",
]


def _compute_field_scores(answers: dict) -> dict:
    """Score each textarea field for AI detection, matching the keys display_feedback_answers expects."""
    scores = {}
    all_text_parts = []
    for i, key in enumerate(_TEXTAREA_KEYS):
        text = answers.get(key, "")
        s = score_field(text) if len(text.strip()) > 30 else 0.0
        scores[f"section_{i}"] = s
        if text.strip():
            all_text_parts.append(text)

    justification = answers.get("overall_preference_justification", "")
    if len(justification.strip()) > 30:
        all_text_parts.append(justification)

    combined = "\n\n".join(all_text_parts)
    scores["overall"] = score_field(combined) if len(combined.strip()) > 50 else 0.0
    return scores


def _display_checklist_progress(checklist: list[dict], completed: list[int], turn_num: int):
    """Show a Rich panel with checklist items and their completion status."""
    lines = []
    for item in checklist:
        done = item["id"] in completed
        icon = "[green]DONE[/green]" if done else "[yellow]PENDING[/yellow]"
        lines.append(f"  {icon}  {item['id']}. {item['description']}")
    done_count = sum(1 for i in checklist if i["id"] in completed)
    total = len(checklist)
    console.print(Panel(
        "\n".join(lines),
        title=f"[bold]Checklist Progress (before Turn {turn_num}): {done_count}/{total}[/bold]",
        box=rbox.ROUNDED, padding=(0, 1),
    ))
    console.print()


def _generate_claude_md(repo_dir: str, owner: str, repo: str, pr_data: dict):
    """Generate a CLAUDE.md if marlin_setup.sh didn't create one."""
    _, lang_out, _ = wsl_exec(
        f"cd '{repo_dir}' && "
        "if [ -f package.json ]; then echo node; "
        "elif [ -f requirements.txt ] || [ -f setup.py ] || [ -f pyproject.toml ]; then echo python; "
        "elif [ -f go.mod ]; then echo go; "
        "elif [ -f Cargo.toml ]; then echo rust; "
        "elif [ -f Gemfile ]; then echo ruby; "
        "else echo unknown; fi"
    )
    lang = lang_out.strip() or "unknown"

    install_cmds = {
        "node": "npm install",
        "python": "pip install -e '.[dev]'",
        "go": "go mod download",
        "rust": "cargo build",
        "ruby": "bundle install",
    }
    test_cmds = {
        "node": "npm test",
        "python": "pytest",
        "go": "go test ./...",
        "rust": "cargo test",
        "ruby": "bundle exec rspec",
    }

    content = (
        f"# CLAUDE.md\n\n"
        f"## Repository Overview\n"
        f"{owner}/{repo} — {lang} project\n\n"
        f"## Dev Setup\n```bash\n{install_cmds.get(lang, '[Fill in]')}\n```\n\n"
        f"## Testing\n```bash\n{test_cmds.get(lang, '[Fill in test command]')}\n```\n\n"
        f"## Code Conventions\n"
        f"Follow existing patterns in the codebase. Match naming style of surrounding code.\n\n"
        f"## Architecture\n"
        f"See the PR for context on the modules being changed.\n"
    )

    wsl_exec(f"cat > '{repo_dir}/CLAUDE.md' << 'CLAUDEEOF'\n{content}\nCLAUDEEOF")


def _find_and_copy_hfi(repo_dir: str):
    """Search common locations for the claude-hfi binary and copy it."""
    local_hfi = f"{HFI_LOCAL_DIR}/claude-hfi"
    local_bin = f"{MARLIN_AUTO_DIR}/bin/claude-hfi"
    search_script = (
        '#!/bin/bash\n'
        f'for f in "{local_hfi}" "{local_bin}" '
        '~/Downloads/claude-hfi '
        '~/Downloads/linux-amd64 '
        '~/Downloads/linux-arm64 '
        '/mnt/c/Users/*/Downloads/claude-hfi '
        '/mnt/c/Users/*/Downloads/linux-amd64 '
        '/mnt/c/Users/*/Desktop/*/claude-hfi '
        '/mnt/c/Users/*/marlin-tools/claude-hfi '
        '$HOME/.local/bin/claude-hfi '
        '/usr/local/bin/claude-hfi; do\n'
        '    if [ -f "$f" ]; then echo "FOUND=$f"; exit 0; fi\n'
        'done\n'
        'echo "FOUND="\n'
    )
    _, out, _ = wsl_exec_script(search_script)
    found = ""
    for line in out.strip().split("\n"):
        if line.startswith("FOUND="):
            found = line.split("=", 1)[1].strip()

    if found:
        wsl_exec(f"cp '{found}' '{repo_dir}/claude-hfi' && chmod +x '{repo_dir}/claude-hfi'")


def _run_setup(task_name: str, pr_data: dict, task_state: str) -> str:
    """Run Phase 3 environment setup with a live progress bar."""
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

    print_phase(3, "ENVIRONMENT SETUP")

    owner = pr_data["owner"]
    repo = pr_data["repo"]
    base_sha = pr_data["meta"]["base_sha"]
    # Clone to native Linux filesystem to avoid symlink/ENOTSUP issues on /mnt/c/
    wsl_task_dir = f"$HOME/marlin-tasks/{task_name}"
    wsl_exec(f"mkdir -p $HOME/marlin-tasks/{task_name}")

    SETUP_STEPS = [
        ("Clone repository", 15),
        ("Install dependencies & configure env", 55),
        ("Generate CLAUDE.md", 10),
        ("Copy HFI binary", 10),
        ("Verify artifacts", 10),
    ]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        total_task = progress.add_task("[bold cyan]Overall setup", total=100)
        step_task = progress.add_task("", total=1, visible=False)
        done_pct = 0

        def _advance_step(idx: int, label: str):
            nonlocal done_pct
            weight = SETUP_STEPS[idx][1]
            progress.update(step_task, description=f"  {label}", completed=0, total=1, visible=True)
            return weight

        def _finish_step(weight: int, ok: bool = True):
            nonlocal done_pct
            progress.update(step_task, completed=1, visible=False)
            done_pct += weight
            progress.update(total_task, completed=done_pct)

        # --- Step 1: Clone ---
        w = _advance_step(0, "Cloning repository...")
        repo_clone_dir = f"{wsl_task_dir}/{repo}"
        clone_script = (
            f'#!/bin/bash\n'
            f'set -e\n'
            f'TASK_DIR="{wsl_task_dir}"\n'
            f'CLONE_DIR="{repo_clone_dir}"\n'
            f'mkdir -p "$TASK_DIR"\n'
            f'if [ -d "$CLONE_DIR/.git" ]; then\n'
            f'    echo "Already cloned"\n'
            f'    echo "REPO_DIR=$CLONE_DIR"\n'
            f'    exit 0\n'
            f'fi\n'
            f'EXISTING=$(find "$TASK_DIR" -mindepth 1 -maxdepth 1 -type d '
            f'-not -name ".git" -not -name ".venv" -not -name ".task-bridge" '
            f'-not -name ".marlin-bridge" 2>/dev/null | head -1)\n'
            f'if [ -n "$EXISTING" ] && [ -d "$EXISTING/.git" ]; then\n'
            f'    echo "REPO_DIR=$EXISTING"\n'
            f'    exit 0\n'
            f'fi\n'
            f'echo "Cloning {owner}/{repo} at {base_sha[:12]}..."\n'
            f'git clone --depth 1 "https://github.com/{owner}/{repo}.git" "$CLONE_DIR"\n'
            f'cd "$CLONE_DIR"\n'
            f'git fetch --depth 1 origin "{base_sha}" 2>/dev/null || true\n'
            f'git checkout "{base_sha}" 2>/dev/null || echo "Using default branch"\n'
            f'echo "REPO_DIR=$CLONE_DIR"\n'
        )

        code, out, err = wsl_exec_script(clone_script, timeout=300)
        repo_dir = ""
        for line in out.strip().split("\n"):
            if line.startswith("REPO_DIR="):
                repo_dir = line.split("=", 1)[1].strip()

        if not repo_dir:
            progress.stop()
            print_error(f"Failed to clone/unpack repo. exit={code}")
            if err:
                print_error(f"  {err[:300]}")
            _, ls_out, _ = wsl_exec(f"ls -d {wsl_task_dir}/*/ 2>/dev/null | head -1")
            if ls_out.strip():
                repo_dir = ls_out.strip().rstrip("/")
                print_status(f"Found existing dir: {repo_dir}")
            if not repo_dir:
                return ""

        _finish_step(w)

        # --- Step 2: Dependencies & config via marlin_setup.sh ---
        w = _advance_step(1, "Running marlin_setup.sh (deps, config)...")
        setup_script = (
            f'#!/bin/bash\n'
            f'export TASK_NAME="{task_name}"\n'
            f'export TASK_DIR="{wsl_task_dir}"\n'
            f'export REPO_DIR="{repo_dir}"\n'
            f'export BASE_COMMIT="{base_sha}"\n'
            f'export OWNER="{owner}"\n'
            f'export REPO="{repo}"\n'
            f'export MARLIN_TOOLS_DIR="$HOME/marlin-tools"\n'
            f'export SKIP_CLONE=true\n'
            f'export SKIP_TESTS=true\n'
            f'\n'
            f'cd {MARLIN_V2}\n'
            f'bash scripts/marlin_setup.sh 2>&1\n'
            f'echo ""\n'
            f'echo "MARLIN_SETUP_EXIT_CODE=$?"\n'
        )

        code, out, err = wsl_exec_script(setup_script, timeout=900)
        full_output = out or ""

        for line in reversed(full_output.strip().split("\n")):
            line = line.strip()
            if not line or line.startswith("MARLIN_SETUP"):
                continue
            try:
                setup_json = json.loads(line)
                json_repo = setup_json.get("repo_dir", "")
                if json_repo:
                    repo_dir = json_repo
                break
            except (json.JSONDecodeError, ValueError):
                continue

        setup_exit_ok = "MARLIN_SETUP_EXIT_CODE=0" in full_output
        _finish_step(w)

        # --- Step 3: CLAUDE.md — generate if marlin_setup.sh didn't ---
        w = _advance_step(2, "Generating CLAUDE.md...")
        _, claude_check, _ = wsl_exec(f"test -f '{repo_dir}/CLAUDE.md' && echo YES || echo NO")
        if "YES" not in claude_check:
            _generate_claude_md(repo_dir, owner, repo, pr_data)
        _finish_step(w)

        # --- Step 4: HFI binary — search common locations ---
        w = _advance_step(3, "Locating HFI binary...")
        _, hfi_check, _ = wsl_exec(f"test -x '{repo_dir}/claude-hfi' && echo YES || echo NO")
        if "YES" not in hfi_check:
            _find_and_copy_hfi(repo_dir)
        # Remove symlinks that break claude-hfi worktree creation (ENOTSUP copyfile)
        wsl_exec(
            f"find '{repo_dir}' -type l -name 'lib64' -delete 2>/dev/null; "
            f"find '{repo_dir}' -maxdepth 3 -type l ! -readable -delete 2>/dev/null; "
            f"echo 'Cleaned symlinks'"
        )
        _finish_step(w)

        # --- Step 5: Verify artifacts ---
        w = _advance_step(4, "Verifying setup artifacts...")
        checks = {
            "Repo exists": f"test -d '{repo_dir}'",
            "Git init":    f"test -d '{repo_dir}/.git'",
            "CLAUDE.md":   f"test -f '{repo_dir}/CLAUDE.md'",
            "HFI binary":  f"test -x '{repo_dir}/claude-hfi'",
        }
        verify_results = {}
        for label, cmd in checks.items():
            c, _, _ = wsl_exec(f"{cmd} && echo YES || echo NO")
            verify_results[label] = "YES" in (wsl_exec(f"{cmd} && echo YES || echo NO")[1])
        _finish_step(w)

    # Print results outside progress bar
    console.print()
    for label, ok_val in verify_results.items():
        if ok_val:
            print_success(f"{label}: OK")
        else:
            print_warning(f"{label}: missing")

    if not verify_results.get("HFI binary", False):
        console.print()
        console.print(Panel(
            "[bold yellow]claude-hfi binary not found![/bold yellow]\n\n"
            "The automation searched these locations:\n"
            f"  {HFI_LOCAL_DIR}/claude-hfi  [bold](put the binary here)[/bold]\n"
            f"  {MARLIN_AUTO_DIR}/bin/claude-hfi  (legacy fallback)\n"
            "  ~/marlin-tools/claude-hfi\n"
            "  ~/Downloads/claude-hfi\n"
            "  /mnt/c/Users/*/Downloads/claude-hfi\n\n"
            "[bold]To fix:[/bold]\n"
            "  1. Download claude-hfi from Anthropic\n"
            "     (check your Snorkel task instructions or team Slack)\n"
            f"  2. Copy it to the repo:\n"
            f"     [cyan]cp ~/Downloads/claude-hfi {repo_dir}/claude-hfi[/cyan]\n"
            f"     [cyan]chmod +x {repo_dir}/claude-hfi[/cyan]\n\n"
            "[dim]The rest of the automation will still work — prompts and\n"
            "feedback answers will be generated. You just can't run HFI\n"
            "sessions until the binary is in place.[/dim]",
            title="[bold]HFI Binary Missing[/bold]",
            box=rbox.ROUNDED,
            padding=(1, 2),
        ))

    if not setup_exit_ok:
        print_warning("Setup completed with warnings (this is often OK).")
        output_lines = [l for l in full_output.split("\n") if l.strip()]
        for line in output_lines[-5:]:
            safe = line.encode("ascii", errors="replace").decode("ascii").strip()
            if safe:
                print_status(f"  {safe}")
    else:
        print_success("Environment setup complete!")

    write_task_file(task_name, "task_state.env",
                    task_state + f"REPO_DIR={repo_dir}\nPHASE=3_COMPLETE\n")
    print_success(f"Repo dir: {repo_dir}")

    return repo_dir


def main():
    parser = argparse.ArgumentParser(
        description="Marlin V3 Automation",
        epilog="API keys are loaded from .env file. Copy .env.example to .env and add your key.",
    )
    parser.add_argument("--gemini", help="Gemini API key (overrides .env)")
    parser.add_argument("--openai", help="OpenAI API key (overrides .env)")
    parser.add_argument("--anthropic", help="Anthropic API key (overrides .env)")
    parser.add_argument("--skip-setup", action="store_true",
                        help="Skip Phase 3 environment setup")
    parser.add_argument("--skip-ranking", action="store_true",
                        help="Skip PR ranking (use first --pr directly without scoring)")
    parser.add_argument("--pr", help="Single PR URL (skips interactive selection)")
    parser.add_argument("--repo-dir", help="WSL path to repo dir (skips setup)")
    parser.add_argument("--resume", metavar="TASK_NAME",
                        help="Resume an existing task (e.g. apache_kafka_10438)")
    parser.add_argument("--turn", type=int, choices=[1, 2, 3],
                        help="Start from this turn number (use with --resume)")
    args = parser.parse_args()

    if args.gemini:
        os.environ["GEMINI_API_KEY"] = args.gemini
    if args.openai:
        os.environ["OPENAI_API_KEY"] = args.openai
    if args.anthropic:
        os.environ["ANTHROPIC_API_KEY"] = args.anthropic

    os.makedirs(DATA_DIR, exist_ok=True)

    print_header()

    info = provider_info()
    if info["status"] != "ready":
        print_error("No LLM API key found!")
        print_error("Copy .env.example to .env and add your API key, or pass --gemini/--openai/--anthropic")
        return

    provider_name, api_key = detect_provider()
    print_success(f"LLM Provider: {info['provider'].upper()} ({info['model']})")
    print_status(f"Key: {info['key_preview']}")
    console.print()

    # ------------------------------------------------------------------
    # RESUME: skip Phase 1-3 if resuming an existing task
    # ------------------------------------------------------------------
    if args.resume:
        task_name = args.resume
        state_raw = read_task_file(task_name, "task_state.env")
        if not state_raw:
            print_error(f"No task_state.env found for task '{task_name}'")
            return

        state = {}
        for line in state_raw.strip().split("\n"):
            if "=" in line:
                k, v = line.split("=", 1)
                state[k.strip()] = v.strip()

        pr_url = state.get("PR_URL", "")
        if not pr_url:
            print_error("task_state.env missing PR_URL")
            return

        print_status(f"Resuming task: {task_name}")
        print_status(f"PR: {pr_url}")

        pr_data = fetch_and_rank_pr(pr_url)
        if not pr_data:
            print_error(f"Failed to fetch PR data from {pr_url}")
            return

        repo_dir = args.repo_dir or state.get("REPO_DIR", "")
        prompt_text = read_task_file(task_name, "turn1_prompt.txt")
        if not prompt_text:
            phase2_raw = read_task_file(task_name, "phase2.md")
            prompt_text = phase2_raw if phase2_raw else "(no prompt found)"

        diff_text = read_task_file(task_name, "pr_diff.txt")
        _resume_start_turn = args.turn or 1
        print_success(f"Resuming from Turn {_resume_start_turn}")
        console.print()

        # Load checklist
        checklist_raw = read_task_file(task_name, "checklist.json")
        checklist = []
        if checklist_raw:
            try:
                checklist = json.loads(checklist_raw)
            except json.JSONDecodeError:
                pass

        # Load completed items from previous turns
        ci_raw = read_task_file(task_name, "completed_items.json")
        _resume_completed: list[int] = []
        if ci_raw:
            try:
                _resume_completed = json.loads(ci_raw)
            except json.JSONDecodeError:
                pass

        # Skip Phase 1-3 entirely, jump to turn loop below
    else:
        # ----------------------------------------------------------
        # PHASE 1: PR Selection
        # ----------------------------------------------------------
        print_phase(1, "PR SELECTION")

        skip_ranking = args.skip_ranking
        pr_url_arg = args.pr

        if not pr_url_arg:
            from rich.prompt import Prompt as _Prompt
            choice = _Prompt.ask(
                "  [bold][P]aste PRs to rank  or  [S]kip with a single PR URL[/bold]",
                choices=["p", "s"],
                default="p",
            )
            if choice.lower() == "s":
                pr_url_arg = _Prompt.ask("  [bold]Paste PR URL[/bold]")
                skip_ranking = True

        if pr_url_arg and skip_ranking:
            print_status(f"Using PR directly: {pr_url_arg}")
            pr_data = fetch_and_rank_pr(pr_url_arg)
            if not pr_data:
                print_error(f"Failed to fetch PR: {pr_url_arg}")
                return
            print_success(f"PR: {pr_data['owner']}/{pr_data['repo']}#{pr_data['number']}")
            print_success(f"Title: {pr_data['meta']['title'][:70]}")
        else:
            if pr_url_arg:
                urls = [pr_url_arg]
            else:
                urls = get_pr_urls()

            if not urls:
                print_error("No PR URLs provided. Exiting.")
                return

            ranked = rank_prs(urls, status_callback=lambda m: print_status(m))

            if not ranked:
                print_error("Failed to fetch any PRs. Check your URLs and network.")
                return

            selected_idx = display_pr_rankings(ranked)
            if selected_idx is None:
                print_error("No PR selected. Exiting.")
                return

            pr_data = ranked[selected_idx]

        task_name = pr_data["task_name"]
        ensure_task_dir(task_name)

        print_success(f"Selected: {pr_data['owner']}/{pr_data['repo']}#{pr_data['number']}")

        task_state = (
            f"PR_URL={pr_data['url']}\n"
            f"OWNER={pr_data['owner']}\n"
            f"REPO={pr_data['repo']}\n"
            f"PR_NUMBER={pr_data['number']}\n"
            f"TASK_NAME={task_name}\n"
            f"BASE_COMMIT={pr_data['meta']['base_sha']}\n"
        )
        write_task_file(task_name, "task_state.env", task_state)

        # ----------------------------------------------------------
        # PHASE 2: Prompt Preparation (full phase2.md)
        # ----------------------------------------------------------
        print_phase(2, "PROMPT PREPARATION")

        print_status("Fetching PR diff...")
        diff_text = fetch_pr_diff(pr_data["owner"], pr_data["repo"], pr_data["number"])
        write_task_file(task_name, "pr_diff.txt", diff_text)

        prompt_text = None
        phase2_doc = None

        while True:
            phase2_doc = generate_phase2_doc(
                pr_data, diff_text, api_key,
                status_callback=lambda m: print_status(m),
            )

            if not phase2_doc:
                print_error("Failed to generate Phase 2 analysis. Retrying...")
                time.sleep(3)
                continue

            prompt_text = phase2_doc["prompt"]

            console.print()
            if phase2_doc.get("repo_def"):
                console.print(Panel(
                    phase2_doc["repo_def"],
                    title="[bold]Repo Definition[/bold]",
                    box=rbox.ROUNDED, padding=(0, 1),
                ))
            if phase2_doc.get("pr_def"):
                console.print(Panel(
                    phase2_doc["pr_def"],
                    title="[bold]PR Definition[/bold]",
                    box=rbox.ROUNDED, padding=(0, 1),
                ))
            if phase2_doc.get("edge_cases"):
                console.print(Panel(
                    phase2_doc["edge_cases"],
                    title="[bold]Edge Cases[/bold]",
                    box=rbox.ROUNDED, padding=(0, 1),
                ))
            if phase2_doc.get("acceptance_criteria"):
                console.print(Panel(
                    phase2_doc["acceptance_criteria"],
                    title="[bold]Acceptance Criteria[/bold]",
                    box=rbox.ROUNDED, padding=(0, 1),
                ))
            checklist = phase2_doc.get("checklist", [])
            if checklist:
                cl_lines = []
                for item in checklist:
                    files = ", ".join(item.get("files", []))
                    cl_lines.append(
                        f"[bold]{item['id']}.[/bold] [{item.get('complexity', '?')}] "
                        f"{item['description']}\n   [dim]{files}[/dim]"
                    )
                console.print(Panel(
                    "\n".join(cl_lines),
                    title=f"[bold]PR Change Checklist ({len(checklist)} items)[/bold]",
                    subtitle="[dim]Turn 1 covers 1-5 · Turns 2-3 adapt based on model progress[/dim]",
                    box=rbox.ROUNDED, padding=(0, 1),
                ))

            validation = full_validation(prompt_text)
            action = display_prompt(prompt_text, validation["overall"], turn=1)

            if action == "accept":
                break
            elif action == "edit":
                prompt_text = get_edited_prompt()
                phase2_doc["prompt"] = prompt_text
                break

        phase2_md = format_phase2_md(pr_data, phase2_doc)
        write_task_file(task_name, "phase2.md", phase2_md)
        print_success("phase2.md saved (repo def, PR def, edge cases, acceptance criteria)")

        checklist = phase2_doc.get("checklist", [])
        if checklist:
            write_task_file(task_name, "checklist.json", json.dumps(checklist, indent=2))
            print_success(f"Checklist saved ({len(checklist)} items)")

        write_task_file(task_name, "turn1_prompt.txt", prompt_text)
        print_success(f"Turn 1 prompt saved ({len(prompt_text.split())} words)")

        # ----------------------------------------------------------
        # PHASE 3: Environment Setup
        # ----------------------------------------------------------
        repo_dir = args.repo_dir or ""

        if not args.skip_setup and not repo_dir:
            repo_dir = _run_setup(task_name, pr_data, task_state)
        elif args.skip_setup:
            print_status("Skipping setup (--skip-setup)")
        elif repo_dir:
            print_status(f"Using provided repo dir: {repo_dir}")

        _resume_start_turn = 1
        _resume_completed = []

    # ------------------------------------------------------------------
    # TURN LOOP (1-3)
    # ------------------------------------------------------------------
    prompts = [prompt_text]
    acceptance_criteria = read_task_file(task_name, "phase2.md") or prompt_text
    completed_items: list[int] = list(_resume_completed)

    # Get HEAD commit for survey answers
    head_commit = ""
    if repo_dir:
        _, hc_out, _ = wsl_exec(f"cd '{repo_dir}' && git rev-parse HEAD 2>/dev/null")
        head_commit = hc_out.strip()

    for turn_num in range(_resume_start_turn, 4):
        print_phase(4 + turn_num - 1, f"TURN {turn_num}")

        if turn_num == 1:
            is_continue = False
            console.print(Panel(
                f"[bold]Fill these into the HFI pre-thread survey:[/bold]\n\n"
                f"  [cyan]Repository:[/cyan]   https://github.com/{pr_data['owner']}/{pr_data['repo']}\n"
                f"  [cyan]PR URL:[/cyan]       {pr_data['url']}\n"
                f"  [cyan]HEAD commit:[/cyan]  {head_commit or 'N/A'}\n"
                f"  [cyan]Workspace:[/cyan]    {repo_dir}\n"
                f"  [cyan]Interface:[/cyan]    {INTERFACE_CODE}",
                title="[bold]Pre-Thread Survey Answers[/bold]",
                box=rbox.ROUNDED,
                padding=(1, 2),
            ))
            console.print()
        else:
            is_continue = True
            print_status("Analyzing model progress against checklist...")
            prev_diffs = read_task_file(task_name, f"turn{turn_num-1}_diffs.txt")

            next_prompt, completed_items = generate_turn_prompt(
                turn_num, prev_diffs, prompts, acceptance_criteria, api_key,
                checklist=checklist, completed_items=completed_items,
            )

            # Show checklist progress
            if checklist:
                _display_checklist_progress(checklist, completed_items, turn_num)

            if next_prompt:
                validation = full_validation(next_prompt)
                action = display_prompt(next_prompt, validation["overall"], turn=turn_num)
                if action == "edit":
                    next_prompt = get_edited_prompt()
                elif action == "regenerate":
                    next_prompt, completed_items = generate_turn_prompt(
                        turn_num, prev_diffs, prompts, acceptance_criteria, api_key,
                        checklist=checklist, completed_items=completed_items,
                    )
                prompts.append(next_prompt)
                write_task_file(task_name, f"turn{turn_num}_prompt.txt", next_prompt)
                write_task_file(task_name, "completed_items.json",
                                json.dumps(completed_items))
            else:
                print_error("Failed to generate prompt. Please write one manually.")
                next_prompt = get_edited_prompt()
                prompts.append(next_prompt)

        hfi_cmds = get_hfi_launch_commands(repo_dir or "<REPO_DIR>", is_continue)
        current_prompt = prompts[turn_num - 1]
        display_hfi_commands(hfi_cmds)

        console.print()
        console.print(Panel(
            current_prompt,
            title=f"[bold]Turn {turn_num} Prompt — Copy and paste into HFI[/bold]",
            box=rbox.ROUNDED,
            padding=(1, 2),
        ))

        console.print()
        console.print("[bold]After pasting the prompt into HFI, wait for both trajectories.[/bold]")
        console.print("[dim]Auto-detection will find result files, or type 'done' to proceed manually.[/dim]")

        session_dir = find_session_dir()
        turn_complete = False

        if session_dir:
            print_status(f"Found HFI session: {session_dir}")
            detect_thread_done = threading.Event()

            def _poll_turn():
                result = wait_for_turn(
                    session_dir, turn_num, timeout=1800,
                    status_callback=lambda m: print_status(m)
                )
                if result:
                    detect_thread_done.set()

            detector = threading.Thread(target=_poll_turn, daemon=True)
            detector.start()

            console.print("[dim]  (Auto-detection running in background. Type [bold]done[/bold] + Enter to proceed manually.)[/dim]")
            while not detect_thread_done.is_set():
                try:
                    user_input = input().strip().lower()
                    if user_input == "done":
                        break
                    if user_input:
                        console.print("[dim]  Type 'done' when both trajectories are finished.[/dim]")
                except (EOFError, KeyboardInterrupt):
                    break

            turn_complete = detect_thread_done.is_set()
        else:
            console.print("[dim]No HFI session detected yet. Type [bold]done[/bold] + Enter when Turn is complete.[/dim]")
            while True:
                try:
                    user_input = input().strip().lower()
                    if user_input == "done":
                        break
                    if user_input:
                        console.print("[dim]  Type 'done' when both trajectories are finished.[/dim]")
                except (EOFError, KeyboardInterrupt):
                    break
            session_dir = find_session_dir()

        # Extract diffs
        print_status("Extracting diffs from worktrees...")
        diffs_a, diffs_b = extract_diffs_from_worktrees(session_dir or "", turn_num)

        if not diffs_a and not diffs_b:
            print_warning("Could not auto-extract diffs. Trying marlin_review.sh...")
            review_script = f"""#!/bin/bash
export REPO_DIR="{repo_dir}"
export OUTPUT_DIR="{MARLIN_V2}/tasks/{task_name}"
export TURN_NUM="{turn_num}"
export NO_POLL=true
cd {MARLIN_V2}
bash scripts/marlin_review.sh 2>&1
"""
            wsl_exec_script(review_script, timeout=120)
            diffs_combined = read_task_file(task_name, f"turn{turn_num}_diffs.txt")
            if diffs_combined:
                mid = len(diffs_combined) // 2
                diffs_a = diffs_combined[:mid]
                diffs_b = diffs_combined[mid:]

        combined_diffs = f"=== TRAJECTORY A ===\n{diffs_a}\n\n=== TRAJECTORY B ===\n{diffs_b}"
        write_task_file(task_name, f"turn{turn_num}_diffs.txt", combined_diffs)
        print_success(f"Diffs saved: turn{turn_num}_diffs.txt")

        # Extract traces
        traces_a, traces_b = "", ""
        if session_dir:
            print_status("Extracting traces...")
            jsonl_a = get_session_file_path(session_dir, turn_num, "A")
            jsonl_b = get_session_file_path(session_dir, turn_num, "B")
            if jsonl_a:
                traces_a = extract_trace(jsonl_a)
            if jsonl_b:
                traces_b = extract_trace(jsonl_b)

        # Generate feedback
        print_status("Generating 17 feedback answers...")
        answers = generate_all_feedback(
            turn=turn_num,
            diffs_a=diffs_a,
            diffs_b=diffs_b,
            traces_a=traces_a,
            traces_b=traces_b,
            acceptance_criteria=acceptance_criteria,
            api_key=api_key,
            status_callback=lambda m: print_status(m),
        )

        # Format and save (already humanized inside generate_all_feedback)
        feedback_md = format_feedback_md(answers, turn_num)

        feedback_path = write_task_file(
            task_name, f"FEEDBACK_ANSWERS_TURN{turn_num}.md", feedback_md
        )

        # Score each field for AI detection display
        field_scores = _compute_field_scores(answers)

        # Display
        display_feedback_answers(answers, field_scores, turn_num)
        display_feedback_file_path(feedback_path)

        while True:
            action = ask_continue_or_view()
            if action == "view":
                console.print()
                console.print(feedback_md)
                console.print()
            elif action == "regenerate":
                print_status("Regenerating all fields...")
                answers = generate_all_feedback(
                    turn=turn_num, diffs_a=diffs_a, diffs_b=diffs_b,
                    traces_a=traces_a, traces_b=traces_b,
                    acceptance_criteria=acceptance_criteria,
                    api_key=api_key,
                    status_callback=lambda m: print_status(m),
                )
                feedback_md = format_feedback_md(answers, turn_num)
                feedback_path = write_task_file(
                    task_name, f"FEEDBACK_ANSWERS_TURN{turn_num}.md", feedback_md
                )
                field_scores = _compute_field_scores(answers)
                display_feedback_answers(answers, field_scores, turn_num)
                display_feedback_file_path(feedback_path)
            elif action == "regen_field":
                field_num = ask_field_number()
                print_status(f"Regenerating field {field_num}...")
                field_key, new_text = regenerate_single_field(
                    field_num=field_num, turn=turn_num,
                    diffs_a=diffs_a, diffs_b=diffs_b,
                    traces_a=traces_a, traces_b=traces_b,
                    acceptance_criteria=acceptance_criteria,
                    api_key=api_key,
                )
                answers[field_key] = new_text
                feedback_md = format_feedback_md(answers, turn_num)
                feedback_path = write_task_file(
                    task_name, f"FEEDBACK_ANSWERS_TURN{turn_num}.md", feedback_md
                )
                field_scores = _compute_field_scores(answers)
                display_feedback_answers(answers, field_scores, turn_num)
                display_feedback_file_path(feedback_path)
            else:
                break

        if turn_num < 3:
            between_steps = get_between_turn_steps(turn_num)
            next_hfi_cmds = get_hfi_launch_commands(repo_dir or "<REPO_DIR>", is_continue=True)
            display_between_turns(turn_num, between_steps, next_hfi_cmds)
            wait_for_user("Press Enter after completing all steps above...")
        else:
            console.print()
            console.print(Panel(
                '[bold]1.[/bold] Paste your Turn 3 feedback answers into HFI\n'
                '[bold]2.[/bold] Select [bold]"Finish conversation"[/bold]\n'
                '[bold]3.[/bold] Fill the post-thread survey if prompted\n'
                '[bold]4.[/bold] Submit on Snorkel Expert Platform',
                title="[bold green]FINAL STEPS[/bold green]",
                box=rbox.HEAVY,
                padding=(1, 2),
                border_style="green",
            ))

    # ------------------------------------------------------------------
    # DONE
    # ------------------------------------------------------------------
    task_dir = os.path.join(DATA_DIR, task_name)
    display_completion(task_name, task_dir)


if __name__ == "__main__":
    main()
