# Marlin V3 Automation

Rich terminal UI that automates Marlin V3 task generation for the Snorkel
evaluation program. Handles PR ranking, adaptive multi-turn prompt generation,
feedback answer generation (21 HFI fields), and multi-pass humanization to
keep AI detection scores below 30%.

You run HFI separately in tmux and copy-paste the pre-generated answers.

> **Platform**: WSL (Ubuntu) on Windows only. macOS support coming soon.

---

## Quick Start

```bash
# 1. Install deps
cd /path/to/marlin-auto
pip3 install -r requirements.txt

# 2. Set up API key
cp .env.example .env
# Edit .env and add at least one LLM key (Gemini, OpenAI, or Claude)

# 3. Place claude-hfi in HFI/claude-hfi (see Step 4 below)

# 4. Run
python3 marlin_auto.py
```

---

## Prerequisites

- **Windows 10/11** with **WSL** installed (Ubuntu recommended)
- **Python 3.10+** inside WSL
- WSL packages: `git`, `bash`, `tmux`, `curl` (all default on Ubuntu)
- **claude-hfi binary** from Anthropic (not open source, see setup below)
- At least one LLM API key (Gemini, OpenAI, or Claude)

---

## Setup Guide (Step by Step)

### Step 1: Install Python dependencies

From inside **WSL**:

```bash
cd /path/to/marlin-auto
pip3 install -r requirements.txt
```

Or from **Windows PowerShell**:

```powershell
cd C:\path\to\marlin-auto
pip install -r requirements.txt
```

### Step 2: Configure your LLM API key

```bash
cp .env.example .env
```

Open `.env` and fill in **at least one** API key. The system auto-detects
which provider you set (priority: Claude > OpenAI > Gemini):

```
# Pick ONE:
GEMINI_API_KEY=AIzaSy...        # Free tier: https://aistudio.google.com/apikey
OPENAI_API_KEY=sk-...           # https://platform.openai.com/api-keys
ANTHROPIC_API_KEY=sk-ant-...    # https://console.anthropic.com/settings/keys
```

### Step 3: Add a GitHub token (recommended)

Without a token, GitHub API limits you to 60 requests/hour. With a token,
you get 5,000/hour.

1. Go to https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. No special scopes needed for public repos
4. Add to `.env`:

```
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### Step 4: Download and place claude-hfi

The `claude-hfi` binary is required for running HFI sessions. It is
distributed by Anthropic and is **not open source**.

1. **Get the binary** from your Snorkel task instructions, team Slack, or
   Anthropic's distribution channel.

2. **Put the binary in the `HFI` folder inside this repo** (recommended):

   The repo includes an empty `HFI/` directory (tracked via `.gitkeep`).
   Copy your downloaded binary there and make it executable:

   ```bash
   cd /path/to/marlin-auto    # same folder as marlin_auto.py
   cp /path/to/downloaded/claude-hfi HFI/claude-hfi
   chmod +x HFI/claude-hfi
   ```

   `HFI/claude-hfi` is gitignored so it is never pushed to GitHub.

   **Fallback** (if you prefer not to use `HFI/`): `~/marlin-tools/claude-hfi`,
   Downloads, `bin/claude-hfi`, etc. See search order below.

3. **During Phase 3**, the automation searches these locations (first match wins)
   and **copies** the binary into the **cloned task repo** (where you run
   `./claude-hfi --tmux`). You do **not** run HFI from inside `marlin-auto`;
   `marlin-auto` only holds your master copy.

   Search order includes:
   - `<marlin-auto>/HFI/claude-hfi`  (expected location)
   - `<marlin-auto>/bin/claude-hfi`  (legacy)
   - `~/marlin-tools/claude-hfi`
   - `~/Downloads/claude-hfi` (and common `linux-amd64` download names)
   - Windows Downloads / Desktop paths under `/mnt/c/Users/...`
   - `~/.local/bin/claude-hfi`
   - `/usr/local/bin/claude-hfi`

4. If it cant find the binary, it will show an error panel with instructions.
   You can also manually copy it after setup:

   ```bash
   cp /path/to/marlin-auto/HFI/claude-hfi /path/to/cloned/repo/claude-hfi
   chmod +x /path/to/cloned/repo/claude-hfi
   ```

Without `claude-hfi`, the automation still generates prompts and feedback
answers, but you wont be able to run the HFI sessions.

### Step 5: Verify WSL is working (Windows only)

```powershell
wsl echo "WSL is working"
```

If this fails, install WSL: `wsl --install`

---

## Running

Run from **WSL** (recommended):

```bash
python3 marlin_auto.py
```

### Pass API key directly (overrides .env)

```bash
python3 marlin_auto.py --gemini YOUR_KEY
python3 marlin_auto.py --openai YOUR_KEY
python3 marlin_auto.py --anthropic YOUR_KEY
```

### Skip Phase 1 (PR ranking)

At the Phase 1 prompt, press **S** to skip ranking and provide a single PR
URL directly. Or use the CLI flag:

```bash
python3 marlin_auto.py --pr https://github.com/owner/repo/pull/123 --skip-ranking
```

### Skip environment setup

```bash
python3 marlin_auto.py --skip-setup
```

### Resume a task

If the automation crashes or you need to come back later, resume from a
specific turn:

```bash
python3 marlin_auto.py --resume apache_kafka_10438 --turn 2
```

This skips Phases 1-3 and jumps straight into the turn loop. It reads
`task_state.env`, `checklist.json`, and existing diffs/prompts from the
task directory.

### Combine options

```bash
python3 marlin_auto.py --pr https://github.com/owner/repo/pull/123 --skip-setup --gemini KEY
```

---

## What Happens When You Run It

### Phase 1: PR Selection

You paste GitHub PR URLs (bulk paste supported). The system fetches metadata
and ranks them by Marlin suitability. Press **S** to skip ranking and use a
single PR directly.

### Phase 2: Analysis and Prompt Generation

Generates a full `phase2.md` document:
- **Repo Definition** - what the repo does
- **PR Definition** - what this change does and why
- **PR Change Checklist** - 6-8 concrete atomic changes extracted from the diff
- **Turn Strategy** - which checklist items go in which turn
- **Edge Cases** - 4-6 specific scenarios that could break
- **Acceptance Criteria** - 5-7 testable outcomes
- **Turn 1 Prompt** - covers checklist items 1-5 vaguely (problem-oriented)

You can accept, edit, or regenerate the prompt.

### Phase 3: Automatic Environment Setup

Live progress bar shows: clone repo, install deps, generate CLAUDE.md,
locate and copy claude-hfi binary. Skip with `--skip-setup` if already done.

### Turn Loop (3 turns)

For each turn:

1. **Prompt** - Turn 1 uses the generated prompt. Turns 2-3 are **adaptive**:
   the system compares model diffs against the checklist to see what was
   completed, then decides whether to push new items or re-prompt for gaps.

2. **HFI Navigation Guide** - shows you the tmux window map:
   - `Ctrl+B then 0` → Feedback form (paste answers here)
   - `Ctrl+B then 1` → Model A trajectory
   - `Ctrl+B then 2` → Model B trajectory

3. **Auto-detection** - polls for result files to detect when both
   trajectories finish.

4. **Checklist Progress** - shows which PR items are DONE vs PENDING before
   generating the next prompt.

5. **Feedback Generation** - generates all 21 HFI feedback fields via LLM,
   then humanizes every answer through a multi-pass pipeline to stay under
   30% AI detection. Each answer gets a different writing "persona" so they
   dont pattern-match across questions.

6. **Copy-paste** - answers are displayed with per-field AI scores and saved
   to `FEEDBACK_ANSWERS_TURNx.md`. You paste them into the HFI feedback form.

7. **Per-field regeneration** - if a specific answer is bad, press `n` to
   regenerate just that one field instead of all 21. See below.

### Between Turns

The automation shows a step-by-step checklist:

1. Paste feedback into HFI (`Ctrl+B then 0` to reach the form)
2. Select "Continue conversation"
3. **Press Ctrl+C twice** to exit HFI
4. Relaunch with `./claude-hfi --tmux --continue`
5. Type `/clear` and press Enter to reset context
6. Paste the next turns prompt

The Ctrl+C + /clear cycle is **critical** -- without it, context accumulates
and model performance degrades by Turn 3.

### Adaptive Prompt Strategy

```
Turn 1: "Heres the problem..." (vaguely covers checklist items 1-5)
  → models run → diffs analyzed against checklist

Turn 2: Did models finish items 1-5?
  YES → "While reviewing, I also noticed..." (items 6-7)
  NO  → "The changes look good but theres gaps in..." (re-prompt 1-5)
  → models run → diffs analyzed again

Turn 3: Push all remaining items + cleanup/tests to match golden PR
```

### Done

After Turn 3: select "Finish conversation" in HFI, fill the post-thread
survey, and submit on Snorkel.

---

## Feedback Review Menu

After feedback is generated, you see this menu:

```
[C]ontinue to next step  [V]iew full file  [R]egenerate all  [N]umber to regen one field
```

| Key | Action |
|-----|--------|
| `c` | Accept answers and move to the next step |
| `v` | Print the full markdown file to the terminal |
| `r` | Regenerate **all** 21 fields from scratch |
| `n` | Regenerate a **single** field by number |

### Per-field regeneration

Press `n` and you get a prompt:

```
Field numbers: 1=Senior engineer expectations, 2=Model A solution quality,
3=Model A agency, 4=Model A communication, 5=Model B solution quality,
6=Model B agency, 7=Model B communication, 21=Overall justification

Enter field number [1-7 or 21]:
```

Type a number (e.g. `5`) and the system regenerates only that field,
rewrites `FEEDBACK_ANSWERS_TURNx.md`, and re-displays all answers.
You can repeat this as many times as needed before pressing `c`.

---

## Humanization Pipeline

Every generated answer goes through:

1. **Regex cleanup** - strips 40+ AI vocabulary words ("leverage" → "use",
   "robust" → "solid"), removes filler phrases, drops apostrophes in
   contractions (dont, its, wont), fixes em dashes, removes backticks
2. **Structural pass** - varies sentence openers, merges short sentences for
   length variation, compacts technical lists
3. **AI score check** (9 weighted signals, target < 30%)
4. **LLM rewrite** - full-text rewrite preserving all technical content but
   varying sentence structure and word choice
5. **Sentence-level fixes** - targets the most "AI-like" sentences for
   individual rewrites
6. **Final cleanup** - strips any remaining backticks, ensures no trailing
   period

Each of the 7 textarea answers gets a **different voice modifier** so
cross-answer patterns are broken. All backticks and markdown formatting
are stripped since HFI text fields are plain text.

---

## File Structure

```
marlin-auto/
  .env.example              # API key template
  .env                      # Your actual keys (gitignored)
  requirements.txt          # Python dependencies
  config.py                 # Paths, LLM settings, scoring thresholds
  utils.py                  # WSL command execution, file I/O
  llm_client.py             # Unified Gemini/OpenAI/Claude client
  ai_scorer.py              # 9-signal AI detection scorer
  humanizer.py              # Multi-pass humanization pipeline
  pr_fetcher.py             # GitHub API PR fetching and ranking
  prompt_generator.py       # Phase 2 analysis + checklist + Turn 1 prompt
  turn_prompt_generator.py  # Adaptive Turn 2/3 prompts (checklist-aware)
  feedback_generator.py     # All 21 HFI feedback field answers
  hfi_watcher.py            # HFI session polling and diff extraction
  tui.py                    # Rich terminal UI components
  marlin_auto.py            # Main orchestrator (entry point)
  HFI/                      # Put claude-hfi here (binary gitignored; folder tracked)
  tasks/                    # Per-task artifacts (gitignored)
    <owner>_<repo>_<pr>/
      phase2.md             # Full analysis document
      checklist.json        # PR change checklist (6-8 items)
      completed_items.json  # Checklist progress tracker
      turn1_prompt.txt      # Turn 1 prompt
      turn2_prompt.txt      # Turn 2 prompt (adaptive)
      turn3_prompt.txt      # Turn 3 prompt (adaptive)
      turn1_diffs.txt       # Model diffs from Turn 1
      FEEDBACK_ANSWERS_TURN1.md  # 21 humanized answers
      FEEDBACK_ANSWERS_TURN2.md
      FEEDBACK_ANSWERS_TURN3.md
```

---

## Troubleshooting

**"No LLM API key found"** - Make sure `.env` exists with a valid key.
Check: `python3 -c "from llm_client import provider_info; print(provider_info())"`

**WSL commands failing** - Check WSL is running: `wsl --status`.
If using a non-Ubuntu distro, set `WSL_DISTRO=YourDistro` in `.env`.

**GitHub rate limit** - Add `GITHUB_TOKEN` to `.env` (see Step 3 above).

**HFI not detected** - Make sure HFI writes results to `/tmp/claude-hfi/`.
You can always press Enter to proceed manually when prompted.

**claude-hfi binary not found** - Put it at `marlin-auto/HFI/claude-hfi`
and run `chmod +x HFI/claude-hfi`, or use a fallback path from the README
search list (e.g. `~/marlin-tools/claude-hfi`).

**AI score too high** - The system retries up to 6 passes. If scores are
still above 30%, press `r` to regenerate all answers, or `n` to regenerate
a specific field by number.

**tmux "sessions should be nested"** - You tried to start tmux inside tmux.
Run `unset TMUX` first, or use `Ctrl+B :switch-client`.
