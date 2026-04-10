"""
Rich terminal UI for Marlin V3 Automation.

Provides panels for PR ranking, prompt display, feedback answers,
AI scores, and status updates. Everything is copy-paste friendly.
"""

import re
import sys
import threading
import time

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich import box

console = Console()


def print_header():
    console.print()
    console.print(
        Panel(
            "[bold cyan]MARLIN V3 AUTOMATION[/bold cyan]\n"
            "[dim]Generate humanized feedback answers for HFI[/dim]",
            box=box.DOUBLE,
            padding=(1, 4),
        )
    )
    console.print()


def print_phase(phase_num: int, title: str):
    console.print()
    console.print(f"[bold yellow]{'='*60}[/bold yellow]")
    console.print(f"[bold yellow]  PHASE {phase_num}: {title}[/bold yellow]")
    console.print(f"[bold yellow]{'='*60}[/bold yellow]")
    console.print()


def print_status(msg: str, style: str = "dim"):
    console.print(f"  [{style}]{msg}[/{style}]")


def print_success(msg: str):
    console.print(f"  [bold green]✓[/bold green] {msg}")


def print_warning(msg: str):
    console.print(f"  [bold yellow]![/bold yellow] {msg}")


def print_error(msg: str):
    console.print(f"  [bold red]✗[/bold red] {msg}")


def _extract_urls(text: str) -> list[str]:
    """Pull all GitHub PR URLs out of arbitrary text."""
    return re.findall(r'https?://github\.com/[^\s,;\"\'<>]+/pull/\d+', text)


def get_pr_urls() -> list[str]:
    """Prompt user to paste PR URLs - supports bulk paste of many URLs at once."""
    console.print(
        "[bold]Paste GitHub PR URLs below.[/bold]\n"
        "[dim]  Paste all at once (bulk) or one per line.\n"
        "  Type [bold]done[/bold] or press Enter on an empty line when finished.[/dim]"
    )
    console.print()

    all_lines: list[str] = []
    collecting = True

    def _read_loop():
        nonlocal collecting
        while collecting:
            try:
                line = sys.stdin.readline()
            except (EOFError, OSError):
                collecting = False
                return
            if line == '':
                collecting = False
                return
            all_lines.append(line)

    reader = threading.Thread(target=_read_loop, daemon=True)
    reader.start()

    while collecting:
        reader.join(timeout=0.3)

        if not all_lines:
            continue

        last = all_lines[-1].strip().lower()
        if last in ("done", "d", "quit", "q"):
            collecting = False
            break

        has_urls = any(
            "github.com" in ln and "/pull/" in ln for ln in all_lines
        )
        if not has_urls:
            continue

        time.sleep(0.8)

        still_reading = reader.is_alive()
        line_count_before = len(all_lines)
        if still_reading:
            time.sleep(0.5)
        line_count_after = len(all_lines)

        if line_count_after == line_count_before:
            last_stripped = all_lines[-1].strip() if all_lines else ""
            if last_stripped == "":
                collecting = False
                break

    big_blob = "\n".join(all_lines)
    urls = _extract_urls(big_blob)

    seen = set()
    unique = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)

    if unique:
        print_success(f"Got {len(unique)} PR URL{'s' if len(unique) != 1 else ''}")
        for i, u in enumerate(unique, 1):
            console.print(f"  [dim]{i:>3}. {u}[/dim]")
        console.print()
    else:
        print_warning("No valid GitHub PR URLs found in input.")

    return unique


def display_pr_rankings(ranked_prs: list[dict]) -> int | None:
    """Display ranked PRs and let user pick one. Returns index or None."""
    if not ranked_prs:
        print_error("No valid PRs found.")
        return None

    table = Table(title="PR Rankings", box=box.ROUNDED, show_lines=True)
    table.add_column("#", style="bold", width=3)
    table.add_column("Score", style="bold cyan", width=7)
    table.add_column("PR", style="bold", min_width=40)
    table.add_column("Details", min_width=30)

    for i, pr in enumerate(ranked_prs):
        score = pr["ranking"]["score"]
        stars = "★" * (score // 20) + "☆" * (5 - score // 20)

        if score >= 70:
            score_style = "bold green"
        elif score >= 50:
            score_style = "bold yellow"
        else:
            score_style = "bold red"

        meta = pr["meta"]
        title_str = f"{pr['owner']}/{pr['repo']}#{pr['number']}\n{meta['title'][:60]}"
        details = "\n".join(pr["ranking"]["reasons"][:4])

        table.add_row(
            str(i + 1),
            Text(f"{score}/100\n{stars}", style=score_style),
            title_str,
            details,
        )

    console.print()
    console.print(table)
    console.print()

    if len(ranked_prs) == 1:
        if Confirm.ask(f"  Use this PR?", default=True):
            return 0
        return None

    choice = IntPrompt.ask(
        f"  Select PR [1-{len(ranked_prs)}]",
        choices=[str(i+1) for i in range(len(ranked_prs))],
    )
    return choice - 1


def display_prompt(prompt_text: str, ai_score: float, turn: int = 1) -> str:
    """Display a generated prompt with score. Returns action: accept/edit/regenerate."""
    score_color = "green" if ai_score < 0.30 else "yellow" if ai_score < 0.45 else "red"
    score_str = f"{ai_score:.0%}"

    console.print(Panel(
        f"{prompt_text}\n\n"
        f"[{score_color}]AI Score: {score_str}[/{score_color}]"
        f"{'  ✓' if ai_score < 0.30 else '  ⚠ Above target'}",
        title=f"[bold]Turn {turn} Prompt ({len(prompt_text.split())} words)[/bold]",
        box=box.ROUNDED,
        padding=(1, 2),
    ))

    action = Prompt.ask(
        "  [bold][A]ccept  [E]dit  [R]egenerate[/bold]",
        choices=["a", "e", "r"],
        default="a",
    )
    return {"a": "accept", "e": "edit", "r": "regenerate"}[action.lower()]


def get_edited_prompt() -> str:
    """Let user type/paste an edited prompt."""
    console.print("[bold]Type or paste your edited prompt (blank line to finish):[/bold]")
    lines = []
    while True:
        line = Prompt.ask("  ", default="")
        if not line.strip() and lines:
            break
        lines.append(line)
    return "\n".join(lines)


def display_setup_progress():
    """Return a Progress context for Phase 3 setup."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    )


def display_hfi_commands(commands: list[str]):
    """Show the user what commands to run for HFI."""
    cmd_text = "\n".join(f"  {c}" for c in commands)
    console.print(Panel(
        f"[bold]Run these in a separate WSL terminal:[/bold]\n\n"
        f"[cyan]{cmd_text}[/cyan]",
        title="[bold]HFI Launch Commands[/bold]",
        box=box.ROUNDED,
        padding=(1, 2),
    ))

    display_hfi_navigation()


def display_hfi_navigation():
    """Show tmux window map so the user knows where everything is."""
    console.print()
    console.print(Panel(
        "[bold]When you run [cyan]./claude-hfi --tmux[/cyan] it creates a tmux session "
        "with 3 windows:[/bold]\n\n"
        "  [cyan]Window 0[/cyan]  [bold]Control / Feedback[/bold]   ← paste prompt here, fill feedback later\n"
        "  [cyan]Window 1[/cyan]  [bold]Model A trajectory[/bold]   ← watch model A work\n"
        "  [cyan]Window 2[/cyan]  [bold]Model B trajectory[/bold]   ← watch model B work\n\n"
        "[bold]How to switch windows (you must be INSIDE the tmux session):[/bold]\n\n"
        "  [yellow]Ctrl+B[/yellow]  then  [yellow]0[/yellow]   → Go to control/feedback window\n"
        "  [yellow]Ctrl+B[/yellow]  then  [yellow]1[/yellow]   → Watch Model A coding\n"
        "  [yellow]Ctrl+B[/yellow]  then  [yellow]2[/yellow]   → Watch Model B coding\n"
        "  [yellow]Ctrl+B[/yellow]  then  [yellow]n[/yellow]   → Next window\n"
        "  [yellow]Ctrl+B[/yellow]  then  [yellow]p[/yellow]   → Previous window\n\n"
        "  [dim]Press Ctrl+B first, RELEASE it, THEN press the number/letter.[/dim]\n"
        "  [dim]If Ctrl+B doesnt work, you might not be inside tmux.[/dim]\n\n"
        "[bold]If you closed the terminal and need to get back in:[/bold]\n\n"
        "  [cyan]tmux attach -t hfi[/cyan]     (or just [cyan]tmux a[/cyan] if only one session)\n\n"
        "[bold]Workflow:[/bold]\n"
        "  1. Run [cyan]./claude-hfi --tmux[/cyan] → it prints a session ID\n"
        "  2. Run [cyan]tmux attach -t <session-id>[/cyan] to enter the session\n"
        "     (HFI prints the exact command for you, just copy-paste it)\n"
        "  3. You land in Window 0 → paste the prompt\n"
        "  4. Both models start in Windows 1 and 2 → switch to watch\n"
        "  5. When both finish, [yellow]Ctrl+B then 0[/yellow] → fill feedback form\n"
        "  6. Tab/Arrows to select question → Enter to open → paste → Enter",
        title="[bold]HFI tmux Navigation Guide[/bold]",
        box=box.HEAVY,
        padding=(1, 2),
        border_style="cyan",
    ))


def display_waiting_for_turn(turn_num: int):
    console.print()
    console.print(Panel(
        f"[bold]Waiting for Turn {turn_num} trajectories to complete...[/bold]\n\n"
        "[dim]The system is polling for result files. You can also\n"
        "press Enter manually when both A and B are done.[/dim]",
        box=box.ROUNDED,
    ))


def display_change_summary(summaries: dict, turn: int):
    """Show what each model did in this turn — side-by-side-ish panels."""
    console.print()
    console.print(f"[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold cyan]  TURN {turn} — WHAT THE MODELS DID[/bold cyan]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]")
    console.print()

    console.print(Panel(
        summaries.get("summary_a", "(no summary)"),
        title="[bold blue]Model A Changes[/bold blue]",
        box=box.ROUNDED,
        padding=(1, 2),
        border_style="blue",
    ))
    console.print()
    console.print(Panel(
        summaries.get("summary_b", "(no summary)"),
        title="[bold magenta]Model B Changes[/bold magenta]",
        box=box.ROUNDED,
        padding=(1, 2),
        border_style="magenta",
    ))
    console.print()


def display_feedback_answers(answers: dict, field_scores: dict, turn: int):
    """Display all 21 HFI answers with per-field AI scores."""
    console.print()
    console.print(f"[bold green]{'='*60}[/bold green]")
    console.print(f"[bold green]  TURN {turn} FEEDBACK ANSWERS READY[/bold green]")
    console.print(f"[bold green]{'='*60}[/bold green]")
    console.print()

    textarea_labels = [
        ("expected_model_response", "1. Senior engineer expectations"),
        ("model_a_solution_quality", "2. Model A — Solution quality"),
        ("model_a_agency", "3. Model A — Agency"),
        ("model_a_communication", "4. Model A — Communication"),
        ("model_b_solution_quality", "5. Model B — Solution quality"),
        ("model_b_agency", "6. Model B — Agency"),
        ("model_b_communication", "7. Model B — Communication"),
    ]

    for i, (key, label) in enumerate(textarea_labels):
        text = answers.get(key, "")
        score = field_scores.get(f"section_{i}", 0)
        _print_answer_panel(label, text, score)

    console.print()
    console.print("[bold]  Axis Preferences:[/bold]")
    axis_fields = [
        "correctness", "mergeability", "instruction_following",
        "scope_calibration", "risk_management", "honesty",
        "intellectual_independence", "verification",
        "clarification_behavior", "engineering_process",
        "tone_understandability",
    ]
    for axis in axis_fields:
        pref = answers.get(axis, "same")
        console.print(f"    {axis}: [cyan]{pref}[/cyan]")
    console.print()

    pref = answers.get("preference", "same")
    axes = answers.get("key_axes", "")
    justification = answers.get("overall_preference_justification", "")
    overall_score = field_scores.get("overall", 0)

    _print_answer_panel(
        f"Overall: [bold]{pref}[/bold]  Axes: {axes}",
        justification, overall_score
    )

    console.print()
    score_color = "green" if overall_score < 0.30 else "yellow" if overall_score < 0.45 else "red"
    console.print(
        f"  [bold]Overall AI Score: [{score_color}]{overall_score:.0%}[/{score_color}]"
        f"{'  ✓ PASS' if overall_score < 0.30 else '  ⚠ CHECK'}[/bold]"
    )
    console.print()


def _print_answer_panel(title: str, text: str, score: float):
    score_color = "green" if score < 0.30 else "yellow" if score < 0.45 else "red"
    score_badge = f"[{score_color}]{score:.0%}[/{score_color}]"

    display_text = text[:500]
    if len(text) > 500:
        display_text += "..."

    console.print(Panel(
        display_text,
        title=f"[bold]{title}[/bold]  {score_badge}",
        box=box.ROUNDED,
        padding=(0, 1),
    ))


def display_feedback_file_path(path: str):
    console.print(Panel(
        f"[bold]Full feedback file saved to:[/bold]\n\n"
        f"  [cyan]{path}[/cyan]\n\n"
        "[bold]To paste answers:[/bold]\n"
        "  1. Open the file above and copy each answer\n"
        "  2. In HFI tmux, press [yellow]Ctrl+B then 0[/yellow] to go to the feedback form\n"
        "  3. Use Tab/Arrow keys to highlight a question\n"
        "  4. Press Enter to open it → paste → Enter to submit\n"
        "  5. Repeat for all 21 fields",
        box=box.ROUNDED,
        padding=(1, 2),
    ))


def ask_continue_or_view() -> str:
    """After showing feedback, ask what to do next."""
    action = Prompt.ask(
        "  [bold][C]ontinue to next step  [V]iew full file  [R]egenerate all  [N]umber to regen one field[/bold]",
        choices=["c", "v", "r", "n"],
        default="c",
    )
    return {"c": "continue", "v": "view", "r": "regenerate", "n": "regen_field"}[action.lower()]


def ask_field_number() -> int:
    """Ask which field number to regenerate (1-7 or 21)."""
    console.print("  [dim]Field numbers: 1=Senior engineer expectations, 2=Model A solution quality,[/dim]")
    console.print("  [dim]3=Model A agency, 4=Model A communication, 5=Model B solution quality,[/dim]")
    console.print("  [dim]6=Model B agency, 7=Model B communication, 21=Overall justification[/dim]")
    while True:
        val = Prompt.ask("  Enter field number [1-7 or 21]")
        try:
            num = int(val.strip())
            if num in (1, 2, 3, 4, 5, 6, 7, 21):
                return num
        except ValueError:
            pass
        console.print("  [red]Invalid. Enter a number 1-7 or 21.[/red]")


def wait_for_user(msg: str = "Press Enter when ready..."):
    console.print()
    Prompt.ask(f"  [bold]{msg}[/bold]", default="")


def display_between_turns(turn_finished: int, steps: list[str], hfi_cmds: list[str]):
    """Show checklist of steps the user must do between HFI turns."""
    steps_text = "\n".join(f"  [bold]{s}[/bold]" for s in steps)
    cmds_text = "\n".join(f"  [cyan]{c}[/cyan]" for c in hfi_cmds)

    console.print()
    console.print(Panel(
        f"[bold yellow]Before starting Turn {turn_finished + 1}, "
        f"complete these steps in your HFI terminal:[/bold yellow]\n\n"
        f"{steps_text}\n\n"
        f"[bold]HFI relaunch commands:[/bold]\n\n"
        f"{cmds_text}\n\n"
        "[dim]After relaunching, type [bold]/clear[/bold] then Enter, "
        "then paste the next prompt.[/dim]\n\n"
        "[bold]Remember:[/bold] [yellow]Ctrl+B then 0[/yellow] = feedback form, "
        "[yellow]Ctrl+B then 1[/yellow] = Model A, "
        "[yellow]Ctrl+B then 2[/yellow] = Model B",
        title=f"[bold red]BETWEEN TURNS {turn_finished} → {turn_finished + 1}[/bold red]",
        box=box.HEAVY,
        padding=(1, 2),
        border_style="red",
    ))


def display_completion(task_name: str, data_dir: str):
    console.print()
    console.print(Panel(
        f"[bold green]All 3 turns complete![/bold green]\n\n"
        f"Task: {task_name}\n"
        f"Artifacts: {data_dir}\n\n"
        "[bold]Next steps:[/bold]\n"
        '1. Select "Finish conversation" in HFI\n'
        "2. Fill post-thread survey if prompted\n"
        "3. Submit on Snorkel Expert Platform",
        title="[bold]COMPLETE[/bold]",
        box=box.DOUBLE,
        padding=(1, 2),
    ))
