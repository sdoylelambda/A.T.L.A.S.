"""
test_brain_dynamic_ctx.py

PURPOSE
-------
Find the minimum ctx tokens Mistral needs to return a complete, valid JSON plan
for each category of command. Calls brain.create_plan() directly with
num_ctx_override — full system prompt, JSON instruction, and examples all apply,
exactly as in production. phi3 is bypassed entirely.

Each command is tested at every ctx level in CTX_LEVELS. The tuning table at
the end shows pass/fail per ctx level so you can read off the minimum ctx needed
for each command and tune _get_num_ctx accordingly.

Run with:
    pytest tests/test_brain_dynamic_ctx.py -v -s

    -s is required to see live output and the final tuning table.

READING THE TUNING TABLE
-------------------------
Each row is one command. Columns are ctx levels. Symbols:
    ✅  = complete valid JSON returned
    ❌  = truncated or structurally invalid JSON

The minimum ctx for a command is the leftmost ✅ where all columns to its
right are also ✅. Use this to set return values in _get_num_ctx.

KNOWN _get_num_ctx BUGS (fix in brain.py)
------------------------------------------
  BUG-1: "backend"/"frontend" in code_keywords misclassifies plain-English
          commands as code → over-allocates ctx.
          Fix: remove "backend"/"frontend" from code_keywords.

  BUG-2: threshold is `words > 15` (strict), so a 15-word code command hits
          code_simple (4096) instead of code_complex (8192). The lightbulb
          command that caused real prod truncation is exactly 15 words.
          Fix: change to `words >= 15`.

Author: Sean Doyle / A.T.L.A.S. Project
"""

import json
import re
import time
import yaml
import pytest
from datetime import datetime
from pathlib import Path

from modules.brain import Brain

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
ERROR_DIR   = Path.home() / ".atlas" / "errors" / "ctx_test_errors"

# ctx levels to test every command at — ordered ascending.
# These map directly to the bucket values used in _get_num_ctx.
CTX_LEVELS = [512, 1024, 2048, 4096, 8192]

# Required top-level keys in every valid Mistral JSON plan
REQUIRED_KEYS = {"summary", "route", "steps"}
VALID_ROUTES  = {"local", "claude", "gemini"}

# ---------------------------------------------------------------------------
# Test commands
# ---------------------------------------------------------------------------
#
# Format: (label, current_bucket, command, notes)
#
#   label          : short snake_case identifier
#   current_bucket : what _get_num_ctx currently assigns — what we are validating
#   command        : the raw voice/text command
#   notes          : "" or "BUG-N" for known _get_num_ctx misclassifications

TEST_COMMANDS = [

    # ---- simple (target bucket: 1024) ----
    ("simple_open_app",
     1024,
     "open the terminal",
     ""),

    ("simple_reminder",
     1024,
     "remind me to drink water",
     ""),

    ("simple_time",
     1024,
     "what time is it",
     ""),

    # ---- medium (target bucket: 2048) ----
    ("medium_calendar",
     2048,
     "add a meeting with the design team on Friday at two pm",
     ""),

    ("medium_email_read",
     2048,
     "read my latest emails and tell me if anything is urgent",
     ""),

    # NOTE: "Python" triggers is_code → actual bucket 4096, correct behaviour
    ("medium_search",
     4096,
     "search the web for the best Python async libraries in 2024",
     ""),

    # ---- long non-code (target bucket: 4096) ----
    # BUG-1: "backend" in code_keywords → actual bucket 8192
    ("long_email_draft",
     8192,
     "draft an email to my manager explaining that the deployment was delayed "
     "because of an unexpected database migration issue and we expect to be back "
     "on track by Thursday morning",
     "BUG-1"),

    ("long_summary_task",
     4096,
     "summarize everything I need to do today based on my calendar events my unread "
     "emails and any reminders I have set so I can plan my morning effectively",
     ""),

    # ---- multi-step ----
    ("multi_medium_files",
     2048,
     "create a folder called reports and also open the file manager",
     ""),

    ("multi_long_workflow",
     4096,
     "check my emails and then draft a reply to the most recent one from the client "
     "and also add a follow up task to my calendar for next Monday with a reminder",
     ""),

    # ---- code simple (target bucket: 4096) ----
    ("code_simple_script",
     4096,
     "write a python script to read a csv file",
     ""),

    ("code_simple_function",
     4096,
     "create a function to validate an email address",
     ""),

    ("code_simple_html",
     4096,
     "create an html page with a login form",
     ""),

    # ---- code complex (target bucket: 8192) ----
    # BUG-2: exactly 15 words → hits code_simple (4096) not code_complex (8192)
    # This is the command that caused real prod truncation.
    ("code_complex_lightbulb",
     4096,
     "create a new project called lightbulbs with Python API for controlling smart lights",
     "BUG-2"),

    ("code_complex_flask_api",
     8192,
     "create a flask api with authentication and a database for managing user accounts "
     "and return json responses for all endpoints",
     ""),

    ("code_complex_react",
     8192,
     "build a react component for a dashboard with a sidebar navigation and a main content "
     "area that loads data from a backend api using typescript",
     ""),

    ("code_complex_django",
     8192,
     "create a django authentication module with login logout and password reset views "
     "using class based views and a postgresql database backend",
     ""),
]

# Flat parametrize list: one entry per (command × ctx_level)
PARAMS = [
    (label, bucket, command, notes, ctx)
    for (label, bucket, command, notes) in TEST_COMMANDS
    for ctx in CTX_LEVELS
]

# ---------------------------------------------------------------------------
# Result store  { (label, ctx) -> result_dict }
# ---------------------------------------------------------------------------

_results: dict[tuple[str, int], dict] = {}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def ensure_error_dir():
    ERROR_DIR.mkdir(parents=True, exist_ok=True)


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", text.lower())


def save_error(label: str, command: str, ctx: int, raw: dict | str):
    ensure_error_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = ERROR_DIR / f"{slugify(label)}_{ctx}ctx_{timestamp}.txt"
    raw_str   = json.dumps(raw, indent=2) if isinstance(raw, dict) else str(raw)
    with open(filename, "w") as f:
        f.write(f"Label:   {label}\n")
        f.write(f"Command: {command}\n")
        f.write(f"CTX:     {ctx}\n")
        f.write(f"Time:    {timestamp}\n")
        f.write("=" * 60 + "\n")
        f.write(raw_str)
    print(f"    [error log] → {filename}")


def validate_plan(plan: dict | str) -> tuple[dict | None, str]:
    """
    Validate a Mistral plan response.
    create_plan() returns a dict — but may also return a fallback dict
    with empty steps if JSON parsing failed internally.

    Returns (data, status) where status is one of:
        'ok'         — valid plan, all required keys, steps is a list
        'truncated'  — string that failed json.loads (raw string returned)
        'incomplete' — parsed but structurally wrong
        'fallback'   — create_plan() returned its error fallback
                       {"summary": <prose>, "steps": [], "route": "local"}
                       meaning Mistral did not return valid JSON at this ctx
    """
    # create_plan() always returns a dict
    if isinstance(plan, str):
        # Should not happen when using create_plan(), but handle defensively
        text = plan.strip()
        text = re.sub(r"^```json\s*", "", text)
        text = re.sub(r"^```\s*",     "", text)
        text = re.sub(r"\s*```$",     "", text)
        try:
            plan = json.loads(text)
        except json.JSONDecodeError:
            return None, "truncated"

    # Detect create_plan()'s internal fallback:
    # {"summary": <raw mistral prose>, "steps": [], "route": "local"}
    # The giveaway is steps=[] combined with a summary that looks like
    # prose rather than a one-sentence plan description.
    steps = plan.get("steps", None)
    summary = plan.get("summary", "")
    route = plan.get("route", "")

    if steps == [] and route == "local" and len(summary) > 120:
        # Long summary + empty steps = Mistral returned prose, create_plan
        # couldn't parse it, fell back to wrapping it in a minimal dict.
        return plan, "fallback"

    missing = REQUIRED_KEYS - set(plan.keys())
    if missing:
        return plan, f"incomplete (missing: {', '.join(sorted(missing))})"

    if route not in VALID_ROUTES:
        return plan, f"incomplete (bad route: '{route}')"

    if not isinstance(steps, list):
        return plan, "incomplete (steps not a list)"

    return plan, "ok"


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def brain():
    config = load_config()
    return Brain(config)


# ---------------------------------------------------------------------------
# Test — one command × one ctx level
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,bucket,command,notes,ctx", PARAMS)
def test_mistral_ctx_tuning(label, bucket, command, notes, ctx, brain):
    """
    Call brain.create_plan() with num_ctx_override=ctx.
    Uses the full production system prompt and JSON examples.
    phi3 is bypassed — this goes straight to Mistral.

    Does NOT hard-fail on bad JSON — collects all results so the full
    matrix is visible in the tuning table. Only hard-fails on exceptions
    (Ollama unreachable etc.).
    """
    t0   = time.time()
    plan = brain.create_plan(command, num_ctx_override=ctx)
    elapsed = round(time.time() - t0, 2)

    _, status = validate_plan(plan)
    passed    = status == "ok"

    _results[(label, ctx)] = {
        "label":     label,
        "bucket":    bucket,
        "command":   command,
        "notes":     notes,
        "ctx":       ctx,
        "elapsed_s": elapsed,
        "status":    status,
        "passed":    passed,
        "words":     len(command.split()),
    }

    if not passed:
        save_error(f"{label}_{ctx}ctx", command, ctx, plan)

    # Live one-liner visible with -s
    icon = "✅" if passed else "❌"
    bug  = f" ⚠{notes}" if notes else ""
    print(
        f"  {icon} {label+bug:<36} ctx={ctx:<5} "
        f"{elapsed:>6.2f}s  {status}"
    )

    # Only hard-fail on exception — bad JSON goes in the table, not as a
    # pytest failure, so we collect data at every ctx level
    assert plan is not None, (
        f"[{label}] ctx={ctx}: create_plan() returned None"
    )


# ---------------------------------------------------------------------------
# Tuning table
# ---------------------------------------------------------------------------

def _min_ctx_for(label: str) -> int | None:
    """
    Return the lowest ctx where the command passed AND all higher levels
    also passed (stability check). Returns None if failed at every level.
    """
    for ctx in CTX_LEVELS:
        key = (label, ctx)
        if key in _results and _results[key]["passed"]:
            higher_all_pass = all(
                _results.get((label, c), {}).get("passed", False)
                for c in CTX_LEVELS if c > ctx
            )
            if higher_all_pass:
                return ctx
    return None


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not _results:
        return

    W = terminalreporter.write_line

    lw = 32   # label column width
    bw = 6    # bucket column width
    cw = 13   # per-ctx column width

    # ------------------------------------------------------------------ #
    # SECTION 1 — Full matrix
    # ------------------------------------------------------------------ #
    terminalreporter.write_sep("=", "A.T.L.A.S. Mistral CTX Tuning Results")
    W("")
    W("MATRIX  ✅ = complete JSON   ❌ = truncated/incomplete/fallback   (seconds shown)")
    W("")

    ctx_headers = "".join(f"  ctx={c:<5}" for c in CTX_LEVELS)
    W(f"  {'Label':<{lw}} {'Bucket':>{bw}}  {ctx_headers}")
    W("  " + "-" * (lw + bw + 4 + len(CTX_LEVELS) * (cw + 2)))

    for label, bucket, command, notes in TEST_COMMANDS:
        bug_marker = f" ⚠{notes}" if notes else ""
        row_label  = f"{label}{bug_marker}"
        row        = f"  {row_label:<{lw}} {bucket:>{bw}}  "
        for ctx in CTX_LEVELS:
            r = _results.get((label, ctx))
            if r is None:
                cell = f"{'--':<{cw}}"
            else:
                icon = "✅" if r["passed"] else "❌"
                cell = f"{icon} {r['elapsed_s']:>5.2f}s"
                cell = f"{cell:<{cw}}"
            row += f"  {cell}"
        W(row)

    W("")

    # ------------------------------------------------------------------ #
    # SECTION 2 — Minimum ctx per command
    # ------------------------------------------------------------------ #
    terminalreporter.write_sep("-", "Minimum CTX per Command")
    W("")
    W(f"  {'Label':<{lw}} {'Words':>5}  {'CurrentBucket':>13}  {'MinNeeded':>9}  Verdict")
    W("  " + "-" * 85)

    over_allocated  = []
    under_allocated = []
    correct         = []

    for label, bucket, command, notes in TEST_COMMANDS:
        words   = len(command.split())
        min_ctx = _min_ctx_for(label)

        if min_ctx is None:
            verdict = "❌ FAILED ALL — needs > 8192 or Mistral system prompt fix"
            under_allocated.append(label)
        elif min_ctx < bucket:
            verdict = f"⬇  over-allocated  (could lower to {min_ctx})"
            over_allocated.append(label)
        elif min_ctx > bucket:
            verdict = f"⬆  UNDER-ALLOCATED (needs {min_ctx}, currently {bucket})"
            under_allocated.append(label)
        else:
            verdict = "✅ correct"
            correct.append(label)

        bug_marker = f" ⚠{notes}" if notes else ""
        W(
            f"  {label+bug_marker:<{lw}} {words:>5}  {bucket:>13}  "
            f"{str(min_ctx) if min_ctx is not None else 'NONE':>9}  {verdict}"
        )

    W("")

    # ------------------------------------------------------------------ #
    # SECTION 3 — Suggested _get_num_ctx changes
    # ------------------------------------------------------------------ #
    terminalreporter.write_sep("-", "Suggested _get_num_ctx Changes")
    W("")

    changes = []
    for label, bucket, command, notes in TEST_COMMANDS:
        min_ctx = _min_ctx_for(label)
        if min_ctx is not None and min_ctx != bucket:
            direction = "REDUCE  " if min_ctx < bucket else "INCREASE"
            changes.append((label, bucket, min_ctx, direction, notes))

    if changes:
        W(f"  {'Label':<{lw}} {'Change':<10} {'From':>6} → {'To':<6}  Notes")
        W("  " + "-" * 70)
        for label, bucket, min_ctx, direction, notes in changes:
            bug = f"⚠ {notes}" if notes else ""
            W(f"  {label:<{lw}} {direction:<10} {bucket:>6} → {min_ctx:<6}  {bug}")
        W("")
        W("  Apply these changes to _get_num_ctx in modules/brain.py.")
    else:
        W("  All buckets correctly sized — no changes needed.")

    W("")

    # ------------------------------------------------------------------ #
    # SECTION 4 — Summary totals
    # ------------------------------------------------------------------ #
    terminalreporter.write_sep("-", "Summary")
    W("")
    W(f"  Commands tested    : {len(TEST_COMMANDS)}")
    W(f"  CTX levels tested  : {CTX_LEVELS}")
    W(f"  Total Mistral calls: {len(PARAMS)}")
    W(f"  ✅ Correctly sized  : {len(correct)}")
    W(f"  ⬇  Over-allocated   : {len(over_allocated)}   {over_allocated}")
    W(f"  ⬆  Under-allocated  : {len(under_allocated)}  {under_allocated}")
    W("")

    # ------------------------------------------------------------------ #
    # SECTION 5 — Known bugs
    # ------------------------------------------------------------------ #
    bugs = [(l, n) for l, b, c, n in TEST_COMMANDS if n]
    if bugs:
        terminalreporter.write_sep("-", "Known _get_num_ctx Bugs to Fix in brain.py")
        W("")
        W("  BUG-1: 'backend'/'frontend' in code_keywords misclassifies plain-English")
        W("         commands as code and over-allocates ctx.")
        W("         Fix: remove 'backend' and 'frontend' from the code_keywords list.")
        W("")
        W("  BUG-2: `words > 15` strict threshold misses exactly-15-word code commands.")
        W("         The lightbulb command (15 words) caused real prod truncation because")
        W("         it landed in code_simple (4096) instead of code_complex (8192).")
        W("         Fix: change `words > 15` → `words >= 15` in _get_num_ctx.")
        W("")

    if under_allocated or any(
        not _results.get((label, ctx), {}).get("passed", True)
        for label, _, _, _ in TEST_COMMANDS
        for ctx in CTX_LEVELS
    ):
        W(f"  ⚠  Error logs saved to: {ERROR_DIR}")
        W("")
