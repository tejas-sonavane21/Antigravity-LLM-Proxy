"""
patcher.py — Antigravity IDE Patcher
======================================
Standalone CLI script. Patches the Antigravity IDE JS files to redirect
API traffic to our local proxy, and restores them from backups.

Usage:
    python patcher.py patch     — Apply patch to IDE files
    python patcher.py unpatch   — Restore files from .js.bak backups
    python patcher.py status    — Show current patch state
    python patcher.py           — Show this help

This script is STANDALONE — it uses only Python built-ins (json, re, sys,
shutil, pathlib). No imports from src/, no pip packages required.

Design ported from reference project:
    patch.rs apply_patch()       → do_patch()    [L23-L107]
    patch.rs remove_patch()      → do_unpatch()  [L110-L130]
    patch.rs check_patch_status()→ do_status()   [L132-L165]
    constants.rs INJECT_CODE     → TLS_INJECT    [L26]

URL Analysis (verified against actual installed IDE):
    Only main.js contains the 3 hardcoded cloudcode URLs.
    The other 3 files resolve URLs via IPC — they only need TLS injection.
    See Phase 3 implementation plan for full analysis.
"""

import json
import re
import shutil
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 4 target files — exact list from patch.rs L7-L19
# Order matches reference project (extensionHost files first, then main, then cli)
TARGET_FILES = [
    "main.js",
    "vs/workbench/api/node/extensionHostProcess.js",
    "vs/workbench/api/worker/extensionHostWorkerMain.js",
    "vs/code/node/cliProcessMain.js",
]

# TLS bypass injection string — from constants.rs L26
# Injected at the very start of each JS file.
TLS_INJECT = "process.env.NODE_TLS_REJECT_UNAUTHORIZED='0';"

# URL matching regex — ported from patch.rs L35
# Matches ALL cloudcode googleapis URLs AND previously-patched 127.0.0.1 URLs.
# The 127.0.0.1:\d+ branch enables idempotent re-patching with a different port.
# Deliberately does NOT match OAuth, telemetry, feedback, or other googleapis URLs.
URL_PATTERN = re.compile(
    r"https://([a-zA-Z0-9.\-]*cloudcode[a-zA-Z0-9.\-]*\.googleapis\.com"
    r"|127\.0\.0\.1:\d+)"
)

# Regex for reading the current patched target from status check
PATCHED_URL_PATTERN = re.compile(r"https?://127\.0\.0\.1:(\d+)")


# ---------------------------------------------------------------------------
# Config Loader
# ---------------------------------------------------------------------------

def load_patcher_config() -> tuple:
    """
    Read config.json (from same directory as this script) and return
    (ide_path: str, target_url: str).

    Performs validation:
    - config.json must exist
    - patcher.ide_path must be set (not optional for patcher)
    - ide_path/main.js must exist (verifies the IDE path is correct)

    Returns:
        tuple: (ide_path, target_url)

    Raises:
        SystemExit: On any config or path error.
    """
    config_path = Path(__file__).parent / "config.json"

    if not config_path.exists():
        _error(
            f"config.json not found at: {config_path}\n"
            f"Run 'python src/main.py' first to generate the template config."
        )

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except json.JSONDecodeError as e:
        _error(f"Invalid JSON in config.json: {e}")

    patcher_cfg = cfg.get("patcher", {})

    # target_url
    target_url = patcher_cfg.get("target_url", "").strip().rstrip("/")
    if not target_url:
        _error("patcher.target_url is missing or empty in config.json")

    # ide_path — mandatory for patcher (unlike src/config.py where it's optional)
    ide_path_raw = patcher_cfg.get("ide_path", "").strip()
    if not ide_path_raw:
        _error(
            "patcher.ide_path is not set in config.json\n"
            "Set it to your IDE's 'out' directory, e.g.:\n"
            '  "ide_path": "D:\\\\Anti_Gravity\\\\Antigravity\\\\resources\\\\app\\\\out"'
        )

    ide_path = Path(ide_path_raw)

    # Verify the path is valid by checking main.js exists
    main_js = ide_path / "main.js"
    if not main_js.exists():
        _error(
            f"IDE not found at: {ide_path}\n"
            f"Expected main.js at: {main_js}\n"
            f"Check patcher.ide_path in config.json."
        )

    return str(ide_path), target_url


# ---------------------------------------------------------------------------
# Patch
# ---------------------------------------------------------------------------

def do_patch(ide_path: str, target_url: str) -> None:
    """
    Apply patch to all TARGET_FILES:
    1. Create .js.bak backup (only if no backup exists yet)
    2. Replace cloudcode URLs → target_url using URL_PATTERN regex
    3. Inject TLS bypass at start of file (if not already present)
    4. Write patched content only if it changed

    Ported from patch.rs apply_patch() L23-L107.
    """
    base = Path(ide_path)
    patched_count = 0
    skipped_count = 0
    errors = []

    print(f"\n[PATCH] Target URL : {target_url}")
    print(f"[PATCH] IDE path   : {ide_path}")
    print()

    for relative_path in TARGET_FILES:
        file_path = base / relative_path
        filename = file_path.name

        # Skip if the file doesn't exist in this IDE build (soft fail)
        if not file_path.exists():
            print(f"  [SKIP]    {relative_path}  (file not found)")
            continue

        # --- Backup FIRST --- (patch.rs L55-L61)
        # Create byte-perfect backup BEFORE reading content for patching.
        # Uses shutil.copy2() to preserve exact bytes, permissions, and timestamps.
        # Only create backup if one doesn't exist — preserves the truly original file.
        backup_path = file_path.with_suffix(".js.bak")
        backup_created = False
        if not backup_path.exists():
            try:
                shutil.copy2(str(file_path), str(backup_path))
                backup_created = True
            except OSError as e:
                errors.append(f"{relative_path}: backup failed — {e}")
                print(f"  [ERROR]   {relative_path}  (backup failed: {e})")
                continue

        # Read content for patching.
        # newline="" prevents Python from converting \n <-> \r\n on Windows,
        # which would corrupt the file (JS files use Unix \n line endings).
        try:
            with open(file_path, "r", encoding="utf-8", newline="") as f:
                original_content = f.read()
        except OSError as e:
            errors.append(f"{relative_path}: read failed — {e}")
            print(f"  [ERROR]   {relative_path}  (read failed: {e})")
            continue

        # --- URL Replacement --- (patch.rs L63-L65)
        url_replace_count = len(URL_PATTERN.findall(original_content))
        new_content = URL_PATTERN.sub(target_url, original_content)

        # --- TLS Injection --- (patch.rs L67-L70)
        tls_injected = False
        if TLS_INJECT not in new_content:
            new_content = TLS_INJECT + new_content
            tls_injected = True

        # --- Write only if changed --- (patch.rs L72-L83)
        if new_content == original_content:
            # File is already correctly patched
            skipped_count += 1
            print(f"  [ALREADY] {relative_path}  (no changes needed)")
            continue

        try:
            with open(file_path, "w", encoding="utf-8", newline="") as f:
                f.write(new_content)
        except OSError as e:
            errors.append(f"{relative_path}: write failed — {e}")
            print(
                f"  [ERROR]   {relative_path}  (write failed: {e})\n"
                f"            → Make sure Antigravity IDE is fully closed and retry."
            )
            continue

        patched_count += 1

        # Build status description for log
        parts = []
        if url_replace_count > 0:
            parts.append(f"{url_replace_count} URL replacement{'s' if url_replace_count > 1 else ''}")
        else:
            parts.append("0 URL replacements")
        if tls_injected:
            parts.append("TLS injected")
        else:
            parts.append("TLS already present")
        bak_note = " [new backup]" if backup_created else " [backup existed]"

        print(f"  [PATCHED] {relative_path}  ({', '.join(parts)}){bak_note}")

    # --- Summary ---
    print()
    total = len(TARGET_FILES)
    if errors:
        print(f"[PATCH] Completed with errors: {patched_count} patched, "
              f"{skipped_count} already patched, {len(errors)} failed")
    elif patched_count == 0 and skipped_count > 0:
        print(f"[PATCH] All files are already patched — nothing to do.")
    else:
        print(f"[PATCH] Done: {patched_count} patched, {skipped_count} already patched")


# ---------------------------------------------------------------------------
# Unpatch
# ---------------------------------------------------------------------------

def do_unpatch(ide_path: str) -> None:
    """
    Restore all TARGET_FILES from their .js.bak backups.
    Deletes the .js.bak file after restoring (patch.rs L120: fs::remove_file).

    Ported from patch.rs remove_patch() L110-L130.
    """
    base = Path(ide_path)
    restored_count = 0
    missing_count = 0

    print(f"\n[UNPATCH] IDE path: {ide_path}")
    print()

    for relative_path in TARGET_FILES:
        file_path = base / relative_path
        backup_path = file_path.with_suffix(".js.bak")

        if not backup_path.exists():
            print(f"  [SKIP]     {relative_path}  (no .js.bak found)")
            missing_count += 1
            continue

        try:
            # Restore: copy backup over the patched file
            shutil.copy2(str(backup_path), str(file_path))
            # Delete the backup after restoring — clean state for next patch cycle
            backup_path.unlink()
            restored_count += 1
            print(f"  [RESTORED] {relative_path}")
        except OSError as e:
            print(
                f"  [ERROR]    {relative_path}  (restore failed: {e})\n"
                f"             → Make sure Antigravity IDE is fully closed and retry."
            )

    # --- Summary ---
    print()
    if restored_count == 0:
        print("[UNPATCH] No backup files found — nothing to restore.")
        print("          (If you patched manually, backups may have been renamed.)")
    else:
        print(f"[UNPATCH] Done: {restored_count} file(s) restored.")
        if missing_count > 0:
            print(f"          {missing_count} file(s) had no backup (may not have been patched).")


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def do_status(ide_path: str) -> None:
    """
    Read main.js and report current patch state.
    Also lists which .js.bak backup files exist.

    Ported from patch.rs check_patch_status() L132-L165.
    """
    base = Path(ide_path)
    main_js = base / "main.js"

    print(f"\n[STATUS] IDE path: {ide_path}")
    print()

    # Read main.js to determine patch status
    try:
        with open(main_js, "r", encoding="utf-8", newline="") as f:
            content = f.read()
    except OSError as e:
        print(f"  [ERROR] Could not read main.js: {e}")
        return

    # Check patch state via TLS marker (patch.rs L146)
    if TLS_INJECT in content:
        # Patched — extract the target URL
        match = PATCHED_URL_PATTERN.search(content)
        if match:
            patched_to = f"https://127.0.0.1:{match.group(1)}"
            print(f"  Patch state : PATCHED")
            print(f"  Target      : {patched_to}")
        else:
            print(f"  Patch state : PATCHED (could not extract target URL)")
    else:
        print(f"  Patch state : NOT PATCHED")

    # List backup files
    print()
    print("  Backup files (.js.bak):")
    any_bak = False
    for relative_path in TARGET_FILES:
        file_path = base / relative_path
        backup_path = file_path.with_suffix(".js.bak")
        if backup_path.exists():
            size_kb = round(backup_path.stat().st_size / 1024, 1)
            print(f"    EXISTS  {relative_path}.bak  ({size_kb} KB)")
            any_bak = True
        else:
            print(f"    MISSING {relative_path}.bak")

    if not any_bak:
        print()
        print("  No backups found. Run 'python patcher.py patch' to patch and create backups.")

    print()


# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

def print_help() -> None:
    print("""
Antigravity IDE Patcher
========================
Redirects IDE API traffic to the local proxy and restores original files.

Commands:
  python patcher.py patch     Apply patch to IDE JS files
  python patcher.py unpatch   Restore original files from .js.bak backups
  python patcher.py status    Show current patch state and backup info

Notes:
  - Close Antigravity IDE before running patch or unpatch.
  - Proxy (python src/main.py) must be running when using the patched IDE.
  - Backups are created automatically on first patch. They are deleted on unpatch.
  - Re-running 'patch' on already-patched files is safe (idempotent).
""")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _error(message: str) -> None:
    """Print an error message and exit with code 1."""
    print(f"\n[ERROR] {message}\n", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print_help()
        sys.exit(0)

    command = sys.argv[1].lower()

    if command in ("patch", "unpatch", "status"):
        ide_path, target_url = load_patcher_config()
    elif command in ("-h", "--help", "help"):
        print_help()
        sys.exit(0)
    else:
        print(f"\n[ERROR] Unknown command: '{sys.argv[1]}'", file=sys.stderr)
        print_help()
        sys.exit(1)

    if command == "patch":
        do_patch(ide_path, target_url)
    elif command == "unpatch":
        do_unpatch(ide_path)
    elif command == "status":
        do_status(ide_path)


if __name__ == "__main__":
    main()
