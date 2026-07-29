import sqlite3
import json
from pathlib import Path
from datetime import datetime, timezone

REGISTRY_DB_PATH = Path(__file__).parent / "file_registry.db"


def _get_conn():
    conn = sqlite3.connect(REGISTRY_DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS uploaded_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            upload_date TEXT NOT NULL,
            storage_type TEXT NOT NULL,
            alias TEXT,
            db_path TEXT,
            tables_created TEXT,
            chunk_count INTEGER,
            status TEXT NOT NULL DEFAULT 'processed',
            error_message TEXT
        )
        """
    )
    # Backfill for existing DBs created before alias/db_path existed.
    cols = {row[1] for row in conn.execute("PRAGMA table_info(uploaded_files)").fetchall()}
    if "alias" not in cols:
        conn.execute("ALTER TABLE uploaded_files ADD COLUMN alias TEXT")
    if "db_path" not in cols:
        conn.execute("ALTER TABLE uploaded_files ADD COLUMN db_path TEXT")
    return conn


def register_file(
    session_id: str,
    name: str,
    type: str,
    size_bytes: int,
    storage_type: str,
    alias: str | None = None,
    db_path: str | None = None,
    tables_created: list[str] | None = None,
    chunk_count: int | None = None,
    status: str = "processed",
    error_message: str | None = None,
) -> int:
    conn = _get_conn()
    cur = conn.execute(
        """INSERT INTO uploaded_files
           (session_id, name, type, size_bytes, upload_date,
            storage_type, alias, db_path, tables_created, chunk_count,
            status, error_message)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            session_id,
            name,
            type,
            size_bytes,
            datetime.now(timezone.utc).isoformat(),
            storage_type,
            alias,
            db_path,
            json.dumps(tables_created) if tables_created else None,
            chunk_count,
            status,
            error_message,
        ),
    )
    conn.commit()
    id = cur.lastrowid
    conn.close()
    return id


def list_files(session_id: str) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        """SELECT id, name, type, size_bytes, upload_date,
                  storage_type, alias, db_path, tables_created, chunk_count,
                  status, error_message
           FROM uploaded_files WHERE session_id = ?
           ORDER BY upload_date ASC""",
        (session_id,),
    ).fetchall()
    conn.close()
    cols = [
        "id", "name", "type", "size_bytes", "upload_date",
        "storage_type", "alias", "db_path", "tables_created", "chunk_count",
        "status", "error_message",
    ]
    out = []
    for row in rows:
        d = dict(zip(cols, row))
        d["tables_created"] = json.loads(d["tables_created"]) if d["tables_created"] else []
        out.append(d)
    return out


def list_sql_schemas(session_id: str) -> list[dict]:
    """Returns [{alias, db_path, filename, tables}] for every successfully
    ingested Excel/CSV file in this session -- used to build the ATTACH
    statements for the agent's read-only connection."""
    files = list_files(session_id)
    return [
        {
            "alias": f["alias"],
            "db_path": f["db_path"],
            "name": f["name"],
            "tables": f["tables_created"],
        }
        for f in files
        if f["storage_type"] == "sql" and f["status"] == "processed" and f["alias"]
    ]


def already_processed(session_id: str, name: str, size_bytes: int) -> bool:
    """Guard against re-ingesting the same file on a Streamlit rerun."""
    conn = _get_conn()
    row = conn.execute(
        """SELECT 1 FROM uploaded_files
           WHERE session_id = ? AND name = ? AND size_bytes = ? AND status = 'processed'""",
        (session_id, name, size_bytes),
    ).fetchone()
    conn.close()
    return row is not None