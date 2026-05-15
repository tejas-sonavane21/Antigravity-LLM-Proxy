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

# SSL_CERT_FILE injection — makes the language server Go binary trust our proxy CA.
# Go 1.19+ respects SSL_CERT_FILE on Windows. The binary inherits env from
# the Electron main process via child_process.spawn.
# Value is machine-specific (absolute path) — computed at patch time.
# Marker is constant so we can detect/remove it on unpatch.
SSL_CERT_MARKER = "process.env.SSL_CERT_FILE="

# URL matching regex — ported from patch.rs L35
URL_PATTERN = re.compile(
    r"https://([a-zA-Z0-9.\-]*cloudcode[a-zA-Z0-9.\-]*\.googleapis\.com"
    r"|127\.0\.0\.1:\d+)"
)

# Regex for reading the current patched target from status check
PATCHED_URL_PATTERN = re.compile(r"https?://127\.0\.0\.1:(\d+)")

# Path to our proxy CA cert, relative to this script.
PROXY_CA_RELATIVE = Path("ag_proxy") / "proxy-ca.crt"

# Combined CA bundle path (certifi standard CAs + our proxy CA).
# Created at patch time. Pointed to by SSL_CERT_FILE env var.
COMBINED_CA_RELATIVE = Path("ag_proxy") / "combined-ca.pem"


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
# Combined CA Bundle + SSL_CERT_FILE injection
# ---------------------------------------------------------------------------

def build_combined_ca(proxy_ca_path: Path, combined_path: Path) -> bool:
    """
    Create ag_proxy/combined-ca.pem = certifi standard CAs + our proxy CA.

    The language server Go binary uses SSL_CERT_FILE env var (Go 1.19+).
    SSL_CERT_FILE REPLACES the system cert pool, so we must include the
    standard CAs (for Google) plus our proxy CA (for our proxy).

    Uses certifi (bundled with httpx in our venv) for the standard CA bundle.
    Falls back to a minimal-but-correct bundle if certifi is unavailable.

    Returns True on success, False on failure.
    """
    if not proxy_ca_path.exists():
        print(f"  [SKIP] proxy CA not found at: {proxy_ca_path}")
        print(f"         Run 'python src/main.py' first to generate TLS certs.")
        return False

    # Find certifi CA bundle
    certifi_bundle: str | None = None
    try:
        import certifi  # type: ignore
        certifi_bundle = certifi.where()
    except ImportError:
        pass

    # Read standard CA bundle
    standard_cas = ""
    if certifi_bundle and Path(certifi_bundle).exists():
        try:
            with open(certifi_bundle, "r", encoding="utf-8") as f:
                standard_cas = f.read()
        except OSError as e:
            print(f"  [WARN] Could not read certifi bundle: {e}")
    else:
        print(f"  [WARN] certifi not found — combined bundle will only contain proxy CA.")
        print(f"         Install certifi: pip install certifi")

    # Read our proxy CA
    try:
        with open(proxy_ca_path, "r", encoding="utf-8") as f:
            proxy_ca = f.read().strip()
    except OSError as e:
        print(f"  [ERROR] proxy CA read failed: {e}")
        return False

    # Write combined bundle
    marker = "# Antigravity Proxy CA — injected by patcher.py"
    combined = f"{standard_cas.rstrip()}\n\n{marker}\n{proxy_ca}\n"
    try:
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        with open(combined_path, "w", encoding="utf-8") as f:
            f.write(combined)
    except OSError as e:
        print(f"  [ERROR] combined-ca.pem write failed: {e}")
        return False

    ca_count = standard_cas.count("-----BEGIN CERTIFICATE-----")
    print(f"  [OK] combined-ca.pem created ({ca_count} standard CAs + proxy CA)")
    return True


def make_cert_inject(combined_ca_path: Path) -> str:
    """
    Build the SSL_CERT_FILE injection line for a JS file.
    Uses forward slashes for the path (safe for JS string literals on Windows).
    """
    # Forward slashes work fine in Node.js / Go on Windows
    path_fwd = str(combined_ca_path).replace("\\", "/")
    return f"process.env.SSL_CERT_FILE='{path_fwd}';"





# ---------------------------------------------------------------------------
# Patch
# ---------------------------------------------------------------------------

def do_patch(ide_path: str, target_url: str) -> None:
    """
    Apply patch to all TARGET_FILES:
    1. Build combined CA bundle (certifi + proxy CA) for SSL_CERT_FILE
    2. Create .js.bak backup (only if no backup exists yet)
    3. Replace cloudcode URLs → target_url using URL_PATTERN regex
    4. Inject TLS + SSL_CERT_FILE at start of file
    5. Write patched content only if it changed

    Ported from patch.rs apply_patch() L23-L107.
    """
    base = Path(ide_path)
    patched_count = 0
    skipped_count = 0
    errors = []

    print(f"\n[PATCH] Target URL : {target_url}")
    print(f"[PATCH] IDE path   : {ide_path}")
    print()

    # --- Build combined CA bundle FIRST ---
    print("[PATCH] Building combined CA bundle for language server binary...")
    proxy_ca_path = (Path(__file__).parent / PROXY_CA_RELATIVE).resolve()
    combined_ca_path = (Path(__file__).parent / COMBINED_CA_RELATIVE).resolve()
    ca_ok = build_combined_ca(proxy_ca_path, combined_ca_path)
    if not ca_ok:
        print("  [WARN] Continuing without SSL_CERT_FILE injection.")
        print("         Binary TLS trust will rely on Windows cert store.")
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

        # --- TLS + SSL_CERT_FILE Injection ---
        tls_injected = False
        cert_injected = False

        if TLS_INJECT not in new_content:
            new_content = TLS_INJECT + new_content
            tls_injected = True

        # SSL_CERT_FILE injection: Go binary trusts proxy cert via env var.
        # Idempotent: strip old SSL_CERT_FILE line first, then re-inject.
        if ca_ok:
            lines = new_content.splitlines(keepends=True)
            lines_no_cert = [l for l in lines if not l.startswith(SSL_CERT_MARKER)]
            if lines_no_cert != lines:
                new_content = "".join(lines_no_cert)

            cert_inject_line = make_cert_inject(combined_ca_path)
            if cert_inject_line not in new_content:
                new_content = cert_inject_line + new_content
                cert_injected = True

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
        parts.append("TLS injected" if tls_injected else "TLS present")
        parts.append("SSL_CERT_FILE injected" if cert_injected else "SSL_CERT_FILE present")
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
    print("[PATCH] All done.")


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

    # --- Delete combined CA bundle ---
    combined_ca_path = (Path(__file__).parent / COMBINED_CA_RELATIVE).resolve()
    if combined_ca_path.exists():
        try:
            combined_ca_path.unlink()
            print(f"  [REMOVED] combined-ca.pem")
        except OSError as e:
            print(f"  [WARN] Could not remove combined-ca.pem: {e}")
    print()
    print("[UNPATCH] All done.")


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

    # Check SSL_CERT_FILE injection state
    ssl_injected = SSL_CERT_MARKER in content
    combined_ca_path = (Path(__file__).parent / COMBINED_CA_RELATIVE).resolve()
    print()
    print("  Language server binary TLS trust:")
    if ssl_injected:
        print(f"    SSL_CERT_FILE : INJECTED into main.js")
        print(f"    combined-ca.pem : {'EXISTS' if combined_ca_path.exists() else 'MISSING (re-run patch)'}")
    else:
        print(f"    SSL_CERT_FILE : NOT injected")
        print(f"    combined-ca.pem : {'EXISTS' if combined_ca_path.exists() else 'not created'}")

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
