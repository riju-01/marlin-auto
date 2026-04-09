"""Quick standalone script to regenerate FEEDBACK_ANSWERS_TURN1.md from saved diffs.
Usage: python -m tools.regen_feedback <task_name> [--turn N]

Examples:
  python -m tools.regen_feedback numpy_numpy_31092
  python -m tools.regen_feedback numpy_numpy_31092 --turn 2
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_REPO_ROOT, ".env"))

from pipeline.feedback_generator import generate_all_feedback, format_feedback_md
from core.llm_client import detect_provider

TASKS_ROOT = os.path.join(_REPO_ROOT, "tasks")

def main():
    if len(sys.argv) < 2:
        available = []
        if os.path.isdir(TASKS_ROOT):
            available = [d for d in os.listdir(TASKS_ROOT) if os.path.isdir(os.path.join(TASKS_ROOT, d))]
        print("Usage: python regen_feedback.py <task_name> [--turn N]")
        if available:
            print(f"\nAvailable tasks: {', '.join(available)}")
        sys.exit(1)

    task_name = sys.argv[1]
    turn = 1
    if "--turn" in sys.argv:
        idx_arg = sys.argv.index("--turn")
        if idx_arg + 1 < len(sys.argv):
            turn = int(sys.argv[idx_arg + 1])

    task_dir = os.path.join(TASKS_ROOT, task_name)
    if not os.path.isdir(task_dir):
        print(f"ERROR: Task directory not found: {task_dir}")
        sys.exit(1)

    provider, api_key = detect_provider()
    if not api_key:
        print("ERROR: No API key found. Set GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY in .env")
        sys.exit(1)
    print(f"Using provider: {provider}")

    diffs_path = os.path.join(task_dir, f"turn{turn}_diffs.txt")
    if not os.path.exists(diffs_path):
        print(f"ERROR: {diffs_path} not found")
        sys.exit(1)

    raw = open(diffs_path, encoding="utf-8").read()
    marker = "=== TRAJECTORY B ==="
    idx = raw.find(marker)
    if idx == -1:
        print("ERROR: Could not find TRAJECTORY B marker in diffs file")
        sys.exit(1)

    diffs_a = raw[:idx].replace("=== TRAJECTORY A ===", "").strip()
    diffs_b = raw[idx + len(marker):].strip()

    phase2_path = os.path.join(task_dir, "phase2.md")
    acceptance = ""
    if os.path.exists(phase2_path):
        acceptance = open(phase2_path, encoding="utf-8").read()

    print(f"Task: {task_name} | Turn: {turn}")
    print(f"Diffs A: {len(diffs_a)} chars, Diffs B: {len(diffs_b)} chars")
    print("Generating feedback answers...")

    answers = generate_all_feedback(
        turn=turn,
        diffs_a=diffs_a,
        diffs_b=diffs_b,
        traces_a="",
        traces_b="",
        acceptance_criteria=acceptance,
        api_key=api_key,
        status_callback=lambda m: print(f"  > {m}"),
    )

    feedback_md = format_feedback_md(answers, turn=turn)

    out_path = os.path.join(task_dir, f"FEEDBACK_ANSWERS_TURN{turn}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(feedback_md)

    print(f"\nDone! Written to: {out_path}")

if __name__ == "__main__":
    main()
