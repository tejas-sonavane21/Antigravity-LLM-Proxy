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

Design ported from reference project:
    patch.rs apply_patch()       → do_patch()    [L23-L107]
    patch.rs remove_patch()      → do_unpatch()  [L110-L130]
    patch.rs check_patch_status()→ do_status()   [L132-L165]
    constants.rs INJECT_CODE     → TLS_INJECT    [L26]

URL Analysis (verified against actual installed IDE):
    Only main.js contains the 3 hardcoded cloudcode URLs.
    The other 3 files resolve URLs via IPC — they only need TLS injection.

Binary TLS Trust (language_server_windows_x64.exe):
    The language server binary is a Go executable. It verifies outgoing
    HTTPS connections using the Windows system cert store (CryptoAPI) by
    default, OR via SSL_CERT_FILE env var (Go 1.19+).

    We inject process.env.SSL_CERT_FILE into main.js pointing to a
    combined CA bundle (certifi standard CAs + our proxy CA). The binary
    inherits this env var via child_process.spawn and trusts our proxy cert.

    NOTE: cert.pem at dist/languageServer/cert.pem is the binary's LOCAL
    SERVER certificate (for its gRPC server), NOT a CA trust bundle.
    Modifying that file has no effect on outgoing TLS verification.
"""

import json
import re
import shutil
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 4 target JS files — exact list from patch.rs L7-L19
# Order matches reference project (extensionHost files first, then main, then cli)
TARGET_FILES = [
    "main.js",
    "vs/workbench/api/node/extensionHostProcess.js",
    "vs/workbench/api/worker/extensionHostWorkerMain.js",
    "vs/code/node/cliProcessMain.js",
]

# TLS bypass injection — from constants.rs L26.
# Injected at the very start of each JS file (Node.js processes).
TLS_INJECT = "process.env.NODE_TLS_REJECT_UNAUTHORIZED='0';"

# URL matching regex — ported from patch.rs L35
URL_PATTERN = re.compile(
    r"https://([a-zA-Z0-9.\-]*cloudcode[a-zA-Z0-9.\-]*\.googleapis\.com"
    r"|127\.0\.0\.1:\d+)"
)

# Regex for reading the current patched target from status check
PATCHED_URL_PATTERN = re.compile(r"https?://127\.0\.0\.1:(\d+)")

# ---------------------------------------------------------------------------
# Pool Trigger Patch
# ---------------------------------------------------------------------------

# Exact string that ends AntigravityAuthMainService.M() in main.js.
# Confirmed by grep: single occurrence at char offset ~11,561,180.
# The method body is fully minified onto one line; this tail is unique.
#
# CRITICAL: We target the INTERIOR of M() — specifically the last statement
# before the closing `}`. The IIFE must be injected INSIDE M(), not after it.
#
# WHY: The `}` closes the `M()` method. After a method in a class body, the
# parser expects only another method definition or the class closing `}`. An
# expression statement like `;(()=>{})()` appended AFTER the `}` would be
# placed inside the class body — which is illegal ES syntax and causes:
#   SyntaxError: Unexpected token '('
#
# CORRECT injection point: replace the closing `}` of M() with:
#   ;(()=>{...IIFE...})()\ }    <-- IIFE runs inside M(), then M() closes
#
# This also fixes the `this` context: as an arrow function IIFE inside M(),
# it captures M()'s `this` (the AntigravityAuthMainService instance).
# The request handler `async(req,res)=>{}` is also an arrow, so it inherits
# the same `this` — meaning `this.n` and `this.refreshUserStatus` work.
POOL_TRIGGER_TARGET = "this.r=setInterval(t,xTa)}"

# IIFE injected INSIDE M() before its closing `}`. Replaces the full target
# string (including the `}`) with: <setInterval>;(()=>{...IIFE...})()}
# So the replacement = IIFE body + M()'s closing }
POOL_TRIGGER_INJECT = ";(()=>{if(!global.__agProxyTriggerServer){const _h=require('http'),_s=_h.createServer(async(req,res)=>{if(req.url==='/refresh-models'&&req.method==='GET'){try{const n=(await this.n).get(),a=R6(n);if(a)await this.refreshUserStatus(a);res.writeHead(200);res.end('ok')}catch(e){res.writeHead(500);res.end(e.message)}}else{res.writeHead(404);res.end()}});_s.listen(9528,'127.0.0.1');global.__agProxyTriggerServer=_s}})()"

# The replacement string: POOL_TRIGGER_TARGET (minus its closing `}`) + IIFE + `}`
# i.e.: 'this.r=setInterval(t,xTa)' + IIFE + '}'
# This keeps the IIFE INSIDE M() and the `}` still closes M() correctly.
POOL_TRIGGER_REPLACEMENT = (
    POOL_TRIGGER_TARGET[:-1]  # strip the closing }
    + POOL_TRIGGER_INJECT     # IIFE as last statement in M()
    + "}"                     # M()'s closing } — now after the IIFE
)

# Sentinel to detect whether the pool trigger is already applied
POOL_TRIGGER_SENTINEL = "__agProxyTriggerServer"


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
    if errors:
        print(f"[PATCH] Completed with errors: {patched_count} patched, "
              f"{skipped_count} already patched, {len(errors)} failed")
    elif patched_count == 0 and skipped_count > 0:
        print(f"[PATCH] All JS files are already patched.")
    else:
        print(f"[PATCH] JS done: {patched_count} patched, {skipped_count} already patched")
    print()
    print("[PATCH] All done. Run 'python patcher.py pool-trigger-patch' to also inject the HTTP refresh trigger.")


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
        print("[UNPATCH] No JS backup files found — nothing to restore.")
        print("          (If you patched manually, backups may have been renamed.)")
    else:
        print(f"[UNPATCH] JS done: {restored_count} file(s) restored.")
        if missing_count > 0:
            print(f"          {missing_count} file(s) had no backup.")

    print()
    print("[UNPATCH] All done. Note: pool-trigger-patch must be manually reverted by running 'python patcher.py unpatch' (restores from .js.bak).")


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

    # Check pool trigger state
    print()
    if POOL_TRIGGER_SENTINEL in content:
        print("  Pool trigger: INJECTED (port 9528 HTTP server present)")
    else:
        print("  Pool trigger: NOT INJECTED (run 'python patcher.py pool-trigger-patch')")

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
# Pool Trigger Patch
# ---------------------------------------------------------------------------

def do_pool_trigger_patch(ide_path: str) -> None:
    """
    Inject the HTTP refresh trigger into AntigravityAuthMainService.M() in main.js.

    The trigger is a self-invoking anonymous function appended to the end of M(),
    right after 'this.r=setInterval(t,xTa)'. It starts a minimal Node.js HTTP server
    on 127.0.0.1:9528 that, when called with GET /refresh-models, invokes
    refreshUserStatus() — causing Antigravity to make a fresh fetchAvailableModels
    request that our proxy can intercept and patch with the next pool entry's context.

    Target method (minified, single line in main.js):
        M(){this.r&&clearInterval(this.r);const t=async()=>{...};t(),this.r=setInterval(t,xTa)}

    Confirmed unique occurrence at ~char 11,561,180.
    """
    main_js = Path(ide_path) / "main.js"

    print(f"\n[POOL-TRIGGER] IDE path: {ide_path}")
    print(f"[POOL-TRIGGER] Target  : {main_js}")
    print()

    if not main_js.exists():
        _error(f"main.js not found at: {main_js}")

    try:
        with open(main_js, "r", encoding="utf-8", newline="") as f:
            content = f.read()
    except OSError as e:
        _error(f"Could not read main.js: {e}")

    # Idempotency check — don't inject twice
    if POOL_TRIGGER_SENTINEL in content:
        print("  [ALREADY] Pool trigger is already injected. Nothing to do.")
        return

    # Verify target string is present
    if POOL_TRIGGER_TARGET not in content:
        _error(
            f"Injection target not found in main.js.\n"
            f"Expected: {POOL_TRIGGER_TARGET!r}\n"
            f"The IDE may have been updated. Inspect main.js around 'refreshUserStatus' "
            f"and update POOL_TRIGGER_TARGET in patcher.py."
        )

    # Count occurrences (must be exactly 1 for safe injection)
    count = content.count(POOL_TRIGGER_TARGET)
    if count != 1:
        _error(
            f"Expected exactly 1 occurrence of the injection target, found {count}.\n"
            f"Target: {POOL_TRIGGER_TARGET!r}\n"
            f"Manual inspection of main.js required."
        )

    # Inject: strip the closing `}` from the target, append the IIFE as the
    # last statement inside M(), then re-add the closing `}` that closes M().
    #
    # Result structure:
    #   M(){...this.r=setInterval(t,xTa);(()=>{...IIFE...})()}  <-- valid
    #                                                       ^^^^
    #                              IIFE runs inside M()  /  M() closes
    #
    # If we had appended AFTER the `}` instead, the IIFE would land in the
    # ES class body — which only allows method definitions, not expression
    # statements — producing SyntaxError: Unexpected token '('.
    patched = content.replace(
        POOL_TRIGGER_TARGET,
        POOL_TRIGGER_REPLACEMENT,
        1,  # replace only the first (and only) occurrence
    )

    # Write
    try:
        with open(main_js, "w", encoding="utf-8", newline="") as f:
            f.write(patched)
    except OSError as e:
        _error(
            f"Write failed: {e}\n"
            f"Make sure Antigravity IDE is fully closed and retry."
        )

    print("  [INJECTED] Pool trigger HTTP server injected into main.js.")
    print("             Restart Antigravity IDE for the patch to take effect.")
    print()
    print("  Trigger endpoint: GET http://127.0.0.1:9528/refresh-models")
    print("  What it does    : Forces Antigravity to call refreshUserStatus(),")
    print("                    which triggers a fresh fetchAvailableModels request.")
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
  python patcher.py patch               Apply URL redirect + TLS patch to IDE JS files
  python patcher.py unpatch             Restore original files from .js.bak backups
  python patcher.py status              Show current patch state and backup info
  python patcher.py pool-trigger-patch  Inject the HTTP refresh trigger into main.js
                                        (enables on-demand fetchAvailableModels refresh
                                         via GET http://127.0.0.1:9528/refresh-models)

Notes:
  - Close Antigravity IDE before running patch or unpatch.
  - Proxy (python src/main.py) must be running when using the patched IDE.
  - Backups are created automatically on first patch. They are deleted on unpatch.
  - Re-running 'patch' on already-patched files is safe (idempotent).
  - pool-trigger-patch is also idempotent and safe to run multiple times.
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

    if command in ("patch", "unpatch", "status", "pool-trigger-patch"):
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
    elif command == "pool-trigger-patch":
        do_pool_trigger_patch(ide_path)


if __name__ == "__main__":
    main()
