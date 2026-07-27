import re
from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.pool import NullPool
from langchain_core.tools import tool

BLOCKED_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
    "TRUNCATE", "ATTACH", "DETACH", "REPLACE", "VACUUM",
    "PRAGMA", "GRANT", "REVOKE",
]

MAX_ROWS = 200

_ALIAS_RE = re.compile(r"^[a-zA-Z0-9_]+$")


class MultiSchemaDB:
    """
    Minimal drop-in replacement for langchain's SQLDatabase, but built for a
    single connection with MULTIPLE attached sqlite databases (one per
    uploaded file, each attached under its own alias/schema name). Exposes
    the same shape (`get_usable_table_names`, `get_table_info`, `run`) so
    the rest of the code (rag.py's auto_generate_schema_docs, agent.py's
    tools) doesn't need to know the difference.

    Table names throughout are qualified as "alias.table", e.g.
    "example_1.sheet1".
    """

    def __init__(self, engine):
        self.engine = engine

    def _schemas(self) -> list[str]:
        insp = inspect(self.engine)
        # 'main' and 'temp' are sqlite's built-in schema names, not
        # user-uploaded files.
        return [s for s in insp.get_schema_names() if s not in ("main", "temp")]

    def get_usable_table_names(self) -> list[str]:
        insp = inspect(self.engine)
        names = []
        for schema in self._schemas():
            for t in insp.get_table_names(schema=schema):
                names.append(f"{schema}.{t}")
        return names

    def get_table_info(self, table_names: list[str] | None = None) -> str:
        insp = inspect(self.engine)
        targets = table_names or self.get_usable_table_names()
        parts = []
        with self.engine.connect() as conn:
            for qualified in targets:
                if "." not in qualified:
                    parts.append(f"-- skipped '{qualified}': expected 'alias.table' format")
                    continue
                schema, table = qualified.split(".", 1)
                try:
                    cols = insp.get_columns(table, schema=schema)
                except Exception as e:
                    parts.append(f"-- could not inspect {qualified}: {e}")
                    continue
                col_defs = ", ".join(f"{c['name']} {c['type']}" for c in cols)
                ddl = f"CREATE TABLE {qualified} ({col_defs});"
                try:
                    sample = conn.execute(text(f"SELECT * FROM {qualified} LIMIT 3")).fetchall()
                    sample_str = "\n".join(str(tuple(row)) for row in sample)
                    block = f"{ddl}\n/*\n{len(sample)} sample rows from {qualified}:\n{sample_str}\n*/"
                except Exception:
                    block = ddl
                parts.append(block)
        return "\n\n".join(parts)

    def run(self, query: str):
        with self.engine.connect() as conn:
            result = conn.execute(text(query))
            if result.returns_rows:
                rows = result.fetchall()
                return str([tuple(r) for r in rows])
            return ""


def build_readonly_db(schemas: list[dict]) -> MultiSchemaDB:
    """
    schemas: list of {"alias": str, "db_path": str} for every uploaded
    Excel/CSV file in this session (see file_registry.list_sql_schemas).

    Builds one base in-memory engine and ATTACHes every file's own .db
    under its alias on every new physical connection (NullPool forces a
    fresh connection each checkout, so newly-uploaded files become visible
    without needing to reconstruct the engine).
    """
    for s in schemas:
        if not _ALIAS_RE.match(s["alias"]):
            raise ValueError(f"Unsafe schema alias: {s['alias']!r}")

    engine = create_engine("sqlite:///:memory:", poolclass=NullPool)

    @event.listens_for(engine, "connect")
    def _set_up_connection(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        for s in schemas:
            # alias is regex-validated above; db_path is bound as a param.
            cursor.execute(f"ATTACH DATABASE ? AS {s['alias']}", (s["db_path"],))
        cursor.execute("PRAGMA query_only = ON;")
        cursor.close()

    return MultiSchemaDB(engine)


def is_safe_query(query: str) -> tuple[bool, str]:
    upper = query.upper()
    for kw in BLOCKED_KEYWORDS:
        if re.search(rf"\b{kw}\b", upper):
            return False, (
                f"Rejected: '{kw}' is not allowed. Only read (SELECT) "
                f"queries are permitted -- please rewrite using SELECT."
            )
    if not re.match(r"^\s*(SELECT|WITH)\b", upper):
        return False, (
            "Rejected: only read (SELECT) queries are allowed. "
            "Please rewrite your query using SELECT."
        )
    return True, ""


def enforce_row_limit(query: str, max_rows: int = MAX_ROWS) -> str:
    if re.search(r"\bLIMIT\s+\d+\b", query, re.IGNORECASE):
        return query
    return query.rstrip().rstrip(";") + f" LIMIT {max_rows};"


def make_safe_query_tool(db: MultiSchemaDB):
    @tool
    def query_data_tables(query: str) -> str:
        """Execute a read-only SQL SELECT query against the user's uploaded
        Excel/CSV data and return the result. Tables must be qualified as
        `alias.table`, e.g. `example_1.sheet1` (alias = the uploaded file's
        name). Only SELECT (or WITH ... SELECT) statements are permitted --
        anything else is rejected with an explanation, not executed."""
        safe, reason = is_safe_query(query)
        if not safe:
            return reason

        limited_query = enforce_row_limit(query)
        try:
            return db.run(limited_query)
        except Exception as e:
            return f"ERROR executing query: {e}"

    return query_data_tables


def make_schema_tools(db: MultiSchemaDB):
    """Replacement for SQLDatabaseToolkit's list/schema tools, since that
    toolkit only understands a single schema and we have one per file."""

    @tool
    def list_data_tables() -> str:
        """List every table currently available to query, qualified as
        `alias.table` (alias = the uploaded file's name). Call this before
        writing a query if you're unsure what tables exist."""
        tables = db.get_usable_table_names()
        return "\n".join(tables) if tables else "No tables available yet -- no Excel/CSV files uploaded."

    @tool
    def describe_table_schema(table_names: str) -> str:
        """Get the CREATE TABLE statement and a few sample rows for one or
        more tables, so you know the exact column names before writing a
        query. Input: comma-separated qualified table names, e.g.
        'example_1.sheet1, example_1.sheet2'."""
        names = [t.strip() for t in table_names.split(",") if t.strip()]
        return db.get_table_info(names)

    return [list_data_tables, describe_table_schema]