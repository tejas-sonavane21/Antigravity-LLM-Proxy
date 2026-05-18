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


# Exact string that ends AntigravityAuthMainService.M() in main.js.
# Confirmed unique occurrence at ~char 11,561,180.
# IIFE is injected INSIDE M() so arrow function captures `this` (the service).
POOL_TRIGGER_TARGET = "this.r=setInterval(t,xTa)}"

# Flag file path placeholder — replaced by do_pool_trigger_patch() at patch time.
# The actual path (e.g. D:\\...\\scratchpad\\ag_proxy_refresh.flag) is embedded
# as a JS string literal so the Node.js watcher knows exactly where to look.
_FLAG_PATH_PLACEHOLDER = "__FLAG_PATH_PLACEHOLDER__"

# fs.watch()-based IIFE injected inside M():
# - Debounce guard (_b) prevents double-trigger from NTFS multiple change events
# - Reads "1" → writes "0" back → calls refreshUserStatus() (same pattern as
#   the existing t() interval function inside M())
# - global.__agProxyWatcher sentinel prevents re-registration if M() reruns
POOL_TRIGGER_INJECT = (
    ";(()=>{if(!global.__agProxyWatcher){"
    "const _fs=require('fs'),_fp=" + repr(_FLAG_PATH_PLACEHOLDER) + ",_b={v:false};"
    "try{const _w=_fs.watch(_fp,async(evt)=>{"
    "if(evt!=='change'||_b.v)return;"
    "_b.v=true;"
    "try{"
    "const _v=_fs.readFileSync(_fp,'utf8').trim();"
    "if(_v==='1'){"
    "_fs.writeFileSync(_fp,'0');"
    "const _n=(await this.n).get(),_a=R6(_n);"
    "if(_a)await this.refreshUserStatus(_a);"
    "}"
    "}catch(_e){}"
    "finally{_b.v=false;}"
    "});"
    "_w.on('error',()=>{global.__agProxyWatcher=null;});"
    "global.__agProxyWatcher=_w;"
    "}catch(_e){}}"
    "})()"
)

# Sentinel to detect whether the pool trigger is already applied
POOL_TRIGGER_SENTINEL = "__agProxyWatcher"




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
    Inject the fs.watch()-based refresh trigger into AntigravityAuthMainService.M()
    in main.js.

    The trigger is an arrow-function IIFE appended to the end of M(), right after
    'this.r=setInterval(t,xTa)'. It sets up a Node.js fs.watch() on the shared
    ag_proxy_refresh.flag file. When the proxy writes "1" to that file, the watcher
    fires, reads "1", writes "0" back, and calls refreshUserStatus() — causing
    Antigravity to make a fresh fetchAvailableModels request that our proxy
    intercepts and patches with the next pool entry's context window.

    The flag file path is read from config.json (patcher.flag_file) and embedded
    as a string literal in the injected JS code at patch time.

    Target method (minified, single line in main.js):
        M(){this.r&&clearInterval(this.r);const t=async()=>{...};t(),this.r=setInterval(t,xTa)}

    Confirmed unique occurrence at ~char 11,561,180.
    """
    main_js = Path(ide_path) / "main.js"

    # Read flag_file path from config.json (required for this patch)
    config_path = Path(__file__).parent / "config.json"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            _cfg = json.load(f)
        flag_file = _cfg.get("patcher", {}).get("flag_file", "")
    except (OSError, json.JSONDecodeError) as e:
        _error(f"Could not read config.json: {e}")
        return  # unreachable — _error exits

    if not flag_file:
        _error(
            "patcher.flag_file is not set in config.json.\n"
            "Add a 'flag_file' entry under the 'patcher' section, e.g.:\n"
            '  "flag_file": "D:\\\\Anti_Projects\\\\...\\\\scratchpad\\\\ag_proxy_refresh.flag"'
        )

    print(f"\n[POOL-TRIGGER] IDE path : {ide_path}")
    print(f"[POOL-TRIGGER] Target   : {main_js}")
    print(f"[POOL-TRIGGER] Flag file: {flag_file}")
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

    # Build the final injection with the real flag file path embedded.
    # flag_file from config.json has real single backslashes: D:\path\to\file
    # In a JS single-quoted string literal each \ must be written as \\
    # POOL_TRIGGER_INJECT contains the placeholder already quoted as:
    #   repr(_FLAG_PATH_PLACEHOLDER) == "'__FLAG_PATH_PLACEHOLDER__'"
    # We replace that exact token with a properly JS-escaped single-quoted path.
    js_flag_path = "'" + flag_file.replace("\\", "\\\\") + "'"
    final_inject = POOL_TRIGGER_INJECT.replace(repr(_FLAG_PATH_PLACEHOLDER), js_flag_path)

    # Build replacement: target without closing `}` + IIFE + `}` closes M()
    pool_trigger_replacement = (
        POOL_TRIGGER_TARGET[:-1]   # strip the closing } of M()
        + final_inject             # IIFE as last statement inside M()
        + "}"                      # M()'s closing } — after the IIFE
    )

    patched = content.replace(
        POOL_TRIGGER_TARGET,
        pool_trigger_replacement,
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

    print("  [INJECTED] Pool trigger (fs.watch) injected into main.js.")
    print("             Restart Antigravity IDE for the patch to take effect.")
    print()
    print(f"  Signal file : {flag_file}")
    print(f"  How it works: Python proxy writes '1' to flag file")
    print(f"                → fs.watch fires in main.js (<10ms)")
    print(f"                → main.js calls refreshUserStatus()")
    print(f"                → IDE makes fetchAvailableModels request")
    print(f"                → Proxy intercepts, serves cached+patched response")
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
