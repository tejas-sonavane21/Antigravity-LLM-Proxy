#!/usr/bin/env python3
"""
keys_db.py — API Key Database Manager
========================================
Standalone CLI tool for managing API keys in scratchpad/keys.db.
Keeps keys out of config.json so the config stays clean and manageable.

All operations use only Python stdlib — no third-party packages needed.

Quick start:
    python keys_db.py init
    python keys_db.py import --from config.json
    python keys_db.py list
    python keys_db.py stats

For full help:
    python keys_db.py --help
    python keys_db.py add --help
    python keys_db.py import --help
"""

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DB_PATH = Path("scratchpad/keys.db")

# Built-in provider -> prefix mapping.
# The prefix is used to generate stable key IDs like "cl-key-7".
# New providers not listed here get a prefix auto-derived or provided via --prefix.
_KNOWN_PREFIXES: dict[str, str] = {
    "clod":        "cl",
    "openrouter":  "or",
    "siliconflow": "sf",
    "opencode":    "oc",
}


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS pool_keys (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    key_id      TEXT    NOT NULL UNIQUE,
    provider_id TEXT    NOT NULL,
    api_key     TEXT    NOT NULL UNIQUE,
    enabled     INTEGER NOT NULL DEFAULT 1,
    label       TEXT    DEFAULT NULL,
    notes       TEXT    DEFAULT NULL,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now', 'utc')),
    updated_at  TEXT    NOT NULL DEFAULT (datetime('now', 'utc'))
);

CREATE INDEX IF NOT EXISTS idx_pk_provider ON pool_keys (provider_id);
CREATE INDEX IF NOT EXISTS idx_pk_enabled  ON pool_keys (provider_id, enabled);

CREATE TABLE IF NOT EXISTS provider_key_seq (
    provider_id TEXT    PRIMARY KEY,
    key_prefix  TEXT    NOT NULL UNIQUE,
    next_seq    INTEGER NOT NULL DEFAULT 1
);
"""


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Open a WAL-mode SQLite connection. Row factory set to sqlite3.Row."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _require_db() -> None:
    """Exit with a helpful message if keys.db doesn't exist yet."""
    if not DB_PATH.exists():
        print(
            f"\n[X] Database not found: {DB_PATH}\n"
            f"  Run first:  python keys_db.py init\n",
            file=sys.stderr,
        )
        sys.exit(1)


def _find_by_api_key(conn: sqlite3.Connection, api_key: str):
    return conn.execute(
        "SELECT * FROM pool_keys WHERE api_key = ?", (api_key,)
    ).fetchone()


def _find_by_key_id(conn: sqlite3.Connection, key_id: str):
    return conn.execute(
        "SELECT * FROM pool_keys WHERE key_id = ?", (key_id,)
    ).fetchone()


def _get_all_prefixes(conn: sqlite3.Connection) -> dict[str, str]:
    """Return {prefix: provider_id} for all registered providers."""
    rows = conn.execute(
        "SELECT provider_id, key_prefix FROM provider_key_seq"
    ).fetchall()
    return {r["key_prefix"]: r["provider_id"] for r in rows}


# ---------------------------------------------------------------------------
# Prefix / provider registration
# ---------------------------------------------------------------------------

def _derive_prefix(provider_id: str) -> str:
    """
    Auto-derive a 2-char prefix from provider_id.
      "my-new-provider" -> "mn"  (first letter of first two words)
      "openrouter"      -> "op"  (single word -> first two chars)
    """
    parts = re.split(r"[-_.]", provider_id)
    parts = [p for p in parts if p]
    if len(parts) >= 2:
        return (parts[0][0] + parts[1][0]).lower()
    return provider_id[:2].lower()


def _prefix_conflict_error(candidate: str, all_prefixes: dict[str, str], provider_id: str) -> str:
    table = "\n".join(
        f"    {pfx:<6} -> {pid}"
        for pfx, pid in sorted(all_prefixes.items())
    )
    return (
        f"\n[X] Prefix '{candidate}' conflicts with existing provider "
        f"'{all_prefixes[candidate]}'.\n\n"
        f"  Existing prefixes:\n{table}\n\n"
        f"  Provide a unique prefix manually:\n"
        f"    python keys_db.py add --provider {provider_id} --prefix <your-prefix> --key \"...\"\n"
    )


def _ensure_provider(conn: sqlite3.Connection, provider_id: str, prefix: str | None = None) -> str:
    """
    Ensure provider exists in provider_key_seq. Returns the key_prefix.
    If provider is new, registers it — auto-deriving or using the supplied prefix.
    Raises SystemExit on prefix conflict.
    """
    row = conn.execute(
        "SELECT key_prefix FROM provider_key_seq WHERE provider_id = ?",
        (provider_id,)
    ).fetchone()
    if row:
        return row["key_prefix"]

    # Provider not registered yet — register now
    all_prefixes = _get_all_prefixes(conn)

    if prefix is None:
        # Check built-in table first
        if provider_id in _KNOWN_PREFIXES:
            prefix = _KNOWN_PREFIXES[provider_id]
        else:
            prefix = _derive_prefix(provider_id)

    prefix = prefix.lower()

    if prefix in all_prefixes and all_prefixes[prefix] != provider_id:
        print(_prefix_conflict_error(prefix, all_prefixes, provider_id), file=sys.stderr)
        sys.exit(1)

    conn.execute(
        "INSERT INTO provider_key_seq (provider_id, key_prefix, next_seq) VALUES (?, ?, 1)",
        (provider_id, prefix)
    )
    return prefix


def _advance_seq(conn: sqlite3.Connection, provider_id: str) -> str:
    """
    Atomically consume one sequence number and return the generated key_id.
    Must be called within an active transaction.
    """
    row = conn.execute(
        "SELECT key_prefix, next_seq FROM provider_key_seq WHERE provider_id = ?",
        (provider_id,)
    ).fetchone()
    if row is None:
        print(f"[X] Provider '{provider_id}' not registered.", file=sys.stderr)
        sys.exit(1)
    key_id = f"{row['key_prefix']}-key-{row['next_seq']}"
    conn.execute(
        "UPDATE provider_key_seq SET next_seq = next_seq + 1 WHERE provider_id = ?",
        (provider_id,)
    )
    return key_id


# ---------------------------------------------------------------------------
# Core insert
# ---------------------------------------------------------------------------

def _insert_one(
    conn: sqlite3.Connection,
    provider_id: str,
    api_key: str,
    prefix: str | None = None,
    label: str | None = None,
    notes: str | None = None,
    explicit_key_id: str | None = None,
) -> tuple[bool, str]:
    """
    Insert one key. Returns (success, key_id_or_conflict_id).

    If explicit_key_id is given (import path), uses it directly.
    Otherwise auto-generates from the provider sequence.

    Prints its own error on duplicate; does NOT sys.exit so callers can
    keep looping (bulk mode).
    """
    # Dedup by actual api_key value
    existing = _find_by_api_key(conn, api_key)
    if existing:
        print(
            f"  [X] Skipped  (api_key already exists as '{existing['key_id']}' "
            f"in provider '{existing['provider_id']}')"
        )
        return False, existing["key_id"]

    if explicit_key_id:
        # Also check by key_id for the import path
        existing_id = _find_by_key_id(conn, explicit_key_id)
        if existing_id:
            print(f"  [X] Skipped  (key_id '{explicit_key_id}' already exists in DB)")
            return False, explicit_key_id
        key_id = explicit_key_id
    else:
        _ensure_provider(conn, provider_id, prefix)
        key_id = _advance_seq(conn, provider_id)

    try:
        conn.execute(
            "INSERT INTO pool_keys (key_id, provider_id, api_key, label, notes) "
            "VALUES (?, ?, ?, ?, ?)",
            (key_id, provider_id, api_key, label, notes)
        )
        return True, key_id
    except sqlite3.IntegrityError as e:
        print(f"  [X] DB error for {key_id}: {e}")
        return False, key_id


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_init(args) -> None:
    """Create keys.db and tables. Idempotent."""
    with _connect() as conn:
        conn.executescript(_SCHEMA_SQL)
        # Seed built-in known prefixes (skip if already present)
        all_prefixes = _get_all_prefixes(conn)
        seeded = []
        for pid, pfx in _KNOWN_PREFIXES.items():
            exists = conn.execute(
                "SELECT 1 FROM provider_key_seq WHERE provider_id = ?", (pid,)
            ).fetchone()
            if exists is None and pfx not in all_prefixes:
                conn.execute(
                    "INSERT INTO provider_key_seq (provider_id, key_prefix, next_seq) VALUES (?, ?, 1)",
                    (pid, pfx)
                )
                seeded.append(pid)

    print(f"\n[OK] Database ready: {DB_PATH}")
    print(f"  Tables   : pool_keys, provider_key_seq")
    if seeded:
        print(f"  Providers: {', '.join(seeded)} (pre-registered)")
    print(
        f"\n  Next steps:\n"
        f"    python keys_db.py import --from config.json   <- migrate existing keys\n"
        f"    python keys_db.py add --provider <id> --key \"...\"  <- add a new key\n"
    )


def cmd_list(args) -> None:
    """List keys, optionally filtered by provider."""
    _require_db()
    with _connect() as conn:
        if args.provider_id:
            rows = conn.execute(
                "SELECT * FROM pool_keys WHERE provider_id = ? ORDER BY id",
                (args.provider_id,)
            ).fetchall()
            provider_ids = [args.provider_id] if rows else []
        else:
            rows = conn.execute(
                "SELECT * FROM pool_keys ORDER BY provider_id, id"
            ).fetchall()
            provider_ids = list(dict.fromkeys(r["provider_id"] for r in rows))

    if not rows:
        if args.provider_id:
            print(f"\n  No keys found for provider '{args.provider_id}'.")
        else:
            print(
                f"\n  Database is empty.\n"
                f"  Add a key:  python keys_db.py add --provider <id> --key \"...\"\n"
                f"  Or import:  python keys_db.py import --from config.json\n"
            )
        return

    W = 70
    print(f"\n  Key Pool  -  {DB_PATH}\n")
    for pid in provider_ids:
        prows = [r for r in rows if r["provider_id"] == pid]
        total  = len(prows)
        active = sum(1 for r in prows if r["enabled"])
        print(f"  +{'-' * (W - 2)}+")
        print(f"  |  Provider: {pid:<20}  [{active} active / {total} total]{' ' * (W - 46 - len(pid))}|")
        print(f"  +{'-'*4}+{'-'*14}+{'-'*12}+{'-'*(W - 35)}+")
        print(f"  | {'#':<3}| {'key_id':<13}| {'status':<11}| {'label':<{W - 36}}|")
        print(f"  +{'-'*4}+{'-'*14}+{'-'*12}+{'-'*(W - 35)}+")
        for i, r in enumerate(prows, 1):
            status = "[OK] active  " if r["enabled"] else "[X] disabled"
            lbl = (r["label"] or "—")[:W - 37]
            print(f"  | {i:<3}| {r['key_id']:<13}| {status:<11}| {lbl:<{W - 36}}|")
        print(f"  +{'-'*4}+{'-'*14}+{'-'*12}+{'-'*(W - 35)}+\n")


def cmd_add(args) -> None:
    """Add one key or bulk-add from file."""
    _require_db()

    if args.from_file:
        _add_from_file(args)
        return

    if not args.key:
        print("[X] Error: --key is required (or use --from-file for bulk add)", file=sys.stderr)
        sys.exit(1)

    with _connect() as conn:
        _ensure_provider(conn, args.provider, args.prefix)
        ok, key_id = _insert_one(
            conn, args.provider, args.key,
            prefix=args.prefix, label=args.label, notes=args.notes
        )
        if ok:
            count = conn.execute(
                "SELECT COUNT(*) FROM pool_keys WHERE provider_id = ? AND enabled = 1",
                (args.provider,)
            ).fetchone()[0]
            print(f"\n[OK] Added {key_id} to provider '{args.provider}'  ({count} active key(s) total)")
            print(f"  Restart the proxy to include this key in the pool.\n")
        else:
            print(
                f"\n  To update an existing key's value, use:\n"
                f"    python keys_db.py rotate {key_id} --new-key \"...\"\n"
            )


def _add_from_file(args) -> None:
    """Bulk-add keys from a .txt or .json file."""
    path = Path(args.from_file)
    if not path.exists():
        print(f"[X] File not found: {path}", file=sys.stderr)
        sys.exit(1)

    entries: list[dict] = []
    try:
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                print("[X] JSON file must be a top-level array: [{\"api_key\": \"...\"}, ...]",
                      file=sys.stderr)
                sys.exit(1)
            for item in data:
                if "api_key" not in item:
                    print(f"[X] JSON entry missing 'api_key': {item}", file=sys.stderr)
                    sys.exit(1)
                entries.append({
                    "api_key": item["api_key"],
                    "label":   item.get("label", args.label),
                    "notes":   item.get("notes", args.notes),
                })
        else:
            # Plain text: one key per line, # = comment
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                entries.append({"api_key": line, "label": args.label, "notes": args.notes})
    except Exception as exc:
        print(f"[X] Error reading file: {exc}", file=sys.stderr)
        sys.exit(1)

    if not entries:
        print(f"[X] No keys found in {path}")
        return

    print(f"\nBulk-add from '{path.name}' -> provider '{args.provider}'  ({len(entries)} entries)\n")
    added = skipped = 0

    with _connect() as conn:
        _ensure_provider(conn, args.provider, args.prefix)
        for entry in entries:
            ok, kid = _insert_one(
                conn, args.provider, entry["api_key"],
                label=entry.get("label"), notes=entry.get("notes")
            )
            if ok:
                print(f"  [OK] {kid}  (added)")
                added += 1
            else:
                skipped += 1

    print(f"\n  Result: {added} added, {skipped} skipped.\n")
    if added:
        print("  Restart the proxy to include new keys in the pool.\n")


def cmd_delete(args) -> None:
    """Permanently delete a key by key_id."""
    _require_db()
    with _connect() as conn:
        row = _find_by_key_id(conn, args.key_id)
        if row is None:
            print(f"\n[X] Key not found: {args.key_id}\n")
            sys.exit(1)

        print(f"\n  Key to delete:")
        print(f"    key_id   : {row['key_id']}")
        print(f"    provider : {row['provider_id']}")
        print(f"    status   : {'active' if row['enabled'] else 'disabled'}")
        print(f"    label    : {row['label'] or '—'}\n")

        if not args.yes:
            confirm = input("  Permanently delete? [y/N] ").strip().lower()
            if confirm != "y":
                print("  Cancelled.\n")
                return

        conn.execute("DELETE FROM pool_keys WHERE key_id = ?", (args.key_id,))
        print(f"\n[OK] Deleted {args.key_id}")
        print(f"  Any cooldown entry for this key_id in cooldowns.json will be ignored on next proxy start.\n")


def cmd_disable(args) -> None:
    """Soft-disable a key (pool skips it, key stays in DB)."""
    _require_db()
    with _connect() as conn:
        row = _find_by_key_id(conn, args.key_id)
        if row is None:
            print(f"\n[X] Key not found: {args.key_id}\n"); sys.exit(1)
        if not row["enabled"]:
            print(f"\n  {args.key_id} is already disabled.\n"); return
        conn.execute(
            "UPDATE pool_keys SET enabled=0, updated_at=datetime('now','utc') WHERE key_id=?",
            (args.key_id,)
        )
        print(f"\n[OK] Disabled {args.key_id}  (proxy will skip it on next start)\n")


def cmd_enable(args) -> None:
    """Re-enable a soft-disabled key."""
    _require_db()
    with _connect() as conn:
        row = _find_by_key_id(conn, args.key_id)
        if row is None:
            print(f"\n[X] Key not found: {args.key_id}\n"); sys.exit(1)
        if row["enabled"]:
            print(f"\n  {args.key_id} is already active.\n"); return
        conn.execute(
            "UPDATE pool_keys SET enabled=1, updated_at=datetime('now','utc') WHERE key_id=?",
            (args.key_id,)
        )
        print(f"\n[OK] Enabled {args.key_id}  (restart proxy to activate)\n")


def cmd_rotate(args) -> None:
    """Replace the api_key value for an existing key_id."""
    _require_db()
    with _connect() as conn:
        row = _find_by_key_id(conn, args.key_id)
        if row is None:
            print(f"\n[X] Key not found: {args.key_id}\n"); sys.exit(1)

        # Check new value not already in use by a DIFFERENT entry
        existing = _find_by_api_key(conn, args.new_key)
        if existing and existing["key_id"] != args.key_id:
            print(
                f"\n[X] New key value already registered as '{existing['key_id']}' "
                f"(provider: {existing['provider_id']}).\n"
                f"  Cannot rotate — would create a duplicate.\n"
            )
            sys.exit(1)

        try:
            conn.execute(
                "UPDATE pool_keys SET api_key=?, updated_at=datetime('now','utc') WHERE key_id=?",
                (args.new_key, args.key_id)
            )
            print(f"\n[OK] Rotated api_key for {args.key_id}")
            print(f"  Restart the proxy to use the new key value.\n")
        except sqlite3.IntegrityError as e:
            print(f"\n[X] DB error: {e}\n", file=sys.stderr); sys.exit(1)


def cmd_label(args) -> None:
    """Set or clear the label for a key."""
    _require_db()
    with _connect() as conn:
        row = _find_by_key_id(conn, args.key_id)
        if row is None:
            print(f"\n[X] Key not found: {args.key_id}\n"); sys.exit(1)
        new_label = args.text or None
        conn.execute(
            "UPDATE pool_keys SET label=?, updated_at=datetime('now','utc') WHERE key_id=?",
            (new_label, args.key_id)
        )
        if new_label:
            print(f"\n[OK] Label for {args.key_id} set to: {new_label!r}\n")
        else:
            print(f"\n[OK] Label for {args.key_id} cleared.\n")


def cmd_stats(args) -> None:
    """Summary counts per provider."""
    _require_db()
    with _connect() as conn:
        providers = conn.execute(
            "SELECT provider_id, key_prefix, next_seq FROM provider_key_seq ORDER BY provider_id"
        ).fetchall()

        if not providers:
            print("\n  No providers registered. Run: python keys_db.py init\n"); return

        print(f"\n  Key Pool Stats  -  {DB_PATH}\n")
        print(f"  {'Provider':<18}  {'Prefix':<8}  {'Total':<7}  {'Active':<8}  {'Disabled':<10}  next_seq")
        print(f"  {'-' * 68}")
        gt = ga = gd = 0
        for p in providers:
            pid = p["provider_id"]
            total   = conn.execute(
                "SELECT COUNT(*) FROM pool_keys WHERE provider_id=?", (pid,)
            ).fetchone()[0]
            active  = conn.execute(
                "SELECT COUNT(*) FROM pool_keys WHERE provider_id=? AND enabled=1", (pid,)
            ).fetchone()[0]
            disabled = total - active
            gt += total; ga += active; gd += disabled
            print(f"  {pid:<18}  {p['key_prefix']:<8}  {total:<7}  {active:<8}  {disabled:<10}  {p['next_seq']}")
        print(f"  {'-' * 68}")
        print(f"  {'TOTAL':<18}  {'':8}  {gt:<7}  {ga:<8}  {gd:<10}\n")


def cmd_providers(args) -> None:
    """List registered providers and their prefixes."""
    _require_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT p.provider_id, p.key_prefix, p.next_seq, COUNT(k.id) AS key_count "
            "FROM provider_key_seq p "
            "LEFT JOIN pool_keys k ON k.provider_id = p.provider_id "
            "GROUP BY p.provider_id ORDER BY p.provider_id"
        ).fetchall()

    if not rows:
        print("\n  No providers. Run: python keys_db.py init\n"); return
    print(f"\n  Registered Providers  -  {DB_PATH}\n")
    print(f"  {'provider_id':<20}  {'prefix':<8}  {'total_keys':<12}  next_seq")
    print(f"  {'-' * 56}")
    for r in rows:
        print(f"  {r['provider_id']:<20}  {r['key_prefix']:<8}  {r['key_count']:<12}  {r['next_seq']}")
    print()


def cmd_export(args) -> None:
    """Export keys to JSON."""
    _require_db()
    with _connect() as conn:
        if args.provider_id:
            rows = conn.execute(
                "SELECT * FROM pool_keys WHERE provider_id=? ORDER BY id",
                (args.provider_id,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM pool_keys ORDER BY provider_id, id"
            ).fetchall()

    data = [
        {
            "key_id":      r["key_id"],
            "provider_id": r["provider_id"],
            "api_key":     r["api_key"],
            "enabled":     bool(r["enabled"]),
            "label":       r["label"],
            "notes":       r["notes"],
            "created_at":  r["created_at"],
        }
        for r in rows
    ]
    output = json.dumps(data, indent=2, ensure_ascii=False)

    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
        print(f"\n[OK] Exported {len(data)} key(s) to {args.out}\n")
    else:
        print(output)


def cmd_import(args) -> None:
    """Import keys from config.json or config.json.bak."""
    path = Path(args.from_file)
    if not path.exists():
        print(f"\n[X] File not found: {path}\n", file=sys.stderr); sys.exit(1)

    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"\n[X] JSON parse error: {e}\n", file=sys.stderr); sys.exit(1)

    providers = cfg.get("model_pool", {}).get("providers", [])
    if not providers:
        print(f"\n[X] No providers found in model_pool.providers in {path}\n"); sys.exit(1)

    # Ensure DB is initialized
    with _connect() as conn:
        conn.executescript(_SCHEMA_SQL)

    total_added = total_skipped = total_errors = 0
    print(f"\nImporting from '{path.name}'...\n")

    for provider in providers:
        pid  = provider.get("id", "")
        keys = provider.get("keys", [])
        if not pid or not keys:
            continue

        print(f"  Provider '{pid}': {len(keys)} key(s) found")
        added = skipped = errors = 0
        max_seq = 0

        with _connect() as conn:
            for entry in keys:
                key_id  = entry.get("id", "")
                api_key = entry.get("api_key", "")

                if not key_id or not api_key:
                    print(f"    [X] Skipped (missing id or api_key): {entry}")
                    errors += 1; continue

                # Extract sequence number to track max_seq
                m = re.search(r"-key-(\d+)$", key_id)
                if m:
                    max_seq = max(max_seq, int(m.group(1)))

                # Determine prefix from key_id (e.g. "cl" from "cl-key-7")
                extracted_pfx = key_id.rsplit("-key-", 1)[0] if "-key-" in key_id else None

                # Ensure provider registered using extracted prefix if available
                prov_row = conn.execute(
                    "SELECT key_prefix FROM provider_key_seq WHERE provider_id=?", (pid,)
                ).fetchone()
                if prov_row is None:
                    pfx_to_use = extracted_pfx or _KNOWN_PREFIXES.get(pid)
                    if pfx_to_use:
                        # Check no conflict
                        all_pfx = _get_all_prefixes(conn)
                        if pfx_to_use in all_pfx and all_pfx[pfx_to_use] != pid:
                            print(
                                f"    [X] Prefix conflict for '{pfx_to_use}' "
                                f"(already used by '{all_pfx[pfx_to_use]}'). "
                                f"Skipping all {len(keys)} keys for provider '{pid}'."
                            )
                            errors += len(keys); break
                        conn.execute(
                            "INSERT OR IGNORE INTO provider_key_seq "
                            "(provider_id, key_prefix, next_seq) VALUES (?, ?, 1)",
                            (pid, pfx_to_use)
                        )
                    else:
                        try:
                            _ensure_provider(conn, pid)
                        except SystemExit:
                            errors += len(keys); break

                ok, kid = _insert_one(conn, pid, api_key, explicit_key_id=key_id)
                if ok:
                    print(f"    [OK] {kid}  (imported)")
                    added += 1
                else:
                    skipped += 1

            # Sync next_seq to max seen + 1
            if max_seq > 0:
                conn.execute(
                    "UPDATE provider_key_seq SET next_seq = MAX(next_seq, ?) WHERE provider_id=?",
                    (max_seq + 1, pid)
                )
                ns = conn.execute(
                    "SELECT next_seq FROM provider_key_seq WHERE provider_id=?", (pid,)
                ).fetchone()
                if ns:
                    print(f"    provider_key_seq.next_seq -> {ns['next_seq']}")

        print(f"    -> {added} imported, {skipped} skipped, {errors} errors\n")
        total_added += added; total_skipped += skipped; total_errors += errors

    print("  " + "-" * 58)
    print(f"  Import complete: {total_added} imported, {total_skipped} skipped, {total_errors} errors\n")

    if total_added > 0:
        print(
            "  Next step: remove the 'keys' arrays from config.json providers\n"
            "  to stop the startup warning about keys still being in config.json.\n"
        )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="keys_db.py",
        description=(
            "API Key Database Manager — manages pool API keys in scratchpad/keys.db.\n"
            "Keeps keys separate from config.json for clean config and easy CRUD.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "commands:\n"
            "  init       Create/init the DB (idempotent)\n"
            "  list       List keys (all or by provider)\n"
            "  add        Add one key or bulk-add from file\n"
            "  delete     Permanently remove a key\n"
            "  disable    Soft-disable a key (pool skips it)\n"
            "  enable     Re-enable a disabled key\n"
            "  rotate     Replace api_key value for an existing key\n"
            "  label      Set or clear a key's label\n"
            "  import     Import keys from config.json / config.json.bak\n"
            "  export     Export all keys to JSON\n"
            "  stats      Summary counts per provider\n"
            "  providers  List registered providers and prefixes\n\n"
            "examples:\n"
            "  python keys_db.py init\n"
            "  python keys_db.py import --from config.json\n"
            "  python keys_db.py list\n"
            "  python keys_db.py add --provider clod --key \"eyJhbGci...\"\n"
            "  python keys_db.py add --provider clod --from-file keys.txt\n"
            "  python keys_db.py delete cl-key-3\n"
            "  python keys_db.py rotate cl-key-5 --new-key \"eyJhbGci...\"\n"
        ),
    )
    sub = p.add_subparsers(dest="command", metavar="command")

    # init
    sub.add_parser("init", help="Create/initialize keys.db (idempotent)")

    # list
    pl = sub.add_parser("list", help="List keys")
    pl.add_argument("provider_id", nargs="?", help="Filter to this provider_id")

    # add
    pa = sub.add_parser(
        "add", help="Add one key or bulk-add from file",
        description=(
            "Add a single key or bulk-add from a file.\n\n"
            "single key:\n"
            "  python keys_db.py add --provider clod --key \"eyJhbGci...\"\n"
            "  python keys_db.py add --provider clod --key \"eyJhbGci...\" --label \"Account #13\"\n\n"
            "bulk from text file (one key per line, # = comment):\n"
            "  python keys_db.py add --provider clod --from-file keys.txt\n\n"
            "bulk from JSON file:\n"
            "  python keys_db.py add --provider clod --from-file keys.json\n"
            "  JSON format: [{\"api_key\": \"...\", \"label\": \"optional\", \"notes\": \"optional\"}, ...]\n\n"
            "new provider (first-time, prefix auto-derived or explicit):\n"
            "  python keys_db.py add --provider my-provider --prefix mp --key \"sk-...\"\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    pa.add_argument("--provider",   required=True, help="Provider ID (matches config.json provider 'id')")
    pa.add_argument("--key",        help="API key value (single key mode)")
    pa.add_argument("--from-file",  metavar="FILE", help="Bulk-add from .txt or .json file")
    pa.add_argument("--prefix",     help="Key prefix for new provider (auto-derived if omitted)")
    pa.add_argument("--label",      help="Optional human-readable label")
    pa.add_argument("--notes",      help="Optional free-form notes")

    # delete
    pd = sub.add_parser("delete", help="Permanently delete a key")
    pd.add_argument("key_id", help="Key ID to delete (e.g. cl-key-3)")
    pd.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")

    # disable
    pdi = sub.add_parser("disable", help="Soft-disable a key (pool skips it)")
    pdi.add_argument("key_id", help="Key ID to disable")

    # enable
    pen = sub.add_parser("enable", help="Re-enable a disabled key")
    pen.add_argument("key_id", help="Key ID to enable")

    # rotate
    pro = sub.add_parser(
        "rotate", help="Replace the api_key value for a key",
        description=(
            "Replace the api_key value for an existing key_id.\n"
            "The key_id stays the same — only the secret changes.\n\n"
            "example:\n"
            "  python keys_db.py rotate cl-key-3 --new-key \"eyJhbGci...\"\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    pro.add_argument("key_id",    help="Key ID to rotate")
    pro.add_argument("--new-key", required=True, metavar="API_KEY", help="New api_key value")

    # label
    plb = sub.add_parser("label", help="Set or clear a key's label")
    plb.add_argument("key_id", help="Key ID")
    plb.add_argument("--text",  help="New label text (omit to clear)")

    # import
    pim = sub.add_parser(
        "import", help="Import keys from config.json or config.json.bak",
        description=(
            "Import all keys from config.json model_pool.providers[].keys[] arrays.\n"
            "Duplicate detection uses actual api_key value — existing keys are skipped.\n\n"
            "examples:\n"
            "  python keys_db.py import --from config.json\n"
            "  python keys_db.py import --from config.json.bak\n\n"
            "after import:\n"
            "  1. python keys_db.py list          <- verify\n"
            "  2. Remove 'keys' arrays from config.json providers\n"
            "  3. Restart proxy\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    pim.add_argument("--from", dest="from_file", required=True, metavar="FILE",
                     help="Path to config.json or config.json.bak")

    # export
    pex = sub.add_parser("export", help="Export keys to JSON")
    pex.add_argument("provider_id", nargs="?", help="Filter to this provider (default: all)")
    pex.add_argument("--out", metavar="FILE", help="Output file (default: stdout)")

    # stats
    sub.add_parser("stats", help="Summary counts per provider")

    # providers
    sub.add_parser("providers", help="List registered providers and prefixes")

    return p


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_CMD = {
    "init":      cmd_init,
    "list":      cmd_list,
    "add":       cmd_add,
    "delete":    cmd_delete,
    "disable":   cmd_disable,
    "enable":    cmd_enable,
    "rotate":    cmd_rotate,
    "label":     cmd_label,
    "import":    cmd_import,
    "export":    cmd_export,
    "stats":     cmd_stats,
    "providers": cmd_providers,
}

if __name__ == "__main__":
    parser = _build_parser()
    args   = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    _CMD[args.command](args)
