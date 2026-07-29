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
    A single connection with MULTIPLE attached sqlite databases (one per
    uploaded Excel/CSV file, each attached under its own alias/schema
    name), so tables are queryable as `alias.table`, e.g. `example_1.sheet1`.
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

    def get_compact_schema(self) -> str:
        """
        Compact `schema.table: col1 (TYPE), col2 (TYPE)` listing, one line
        per table. Meant to be injected directly into the system prompt so
        the agent already knows every table and column before it ever
        writes a query -- no separate list-tables/describe-schema tool
        call needed. If the agent wants to see actual sample values, it can
        just run `SELECT ... LIMIT 5` itself through sql_tool.
        """
        insp = inspect(self.engine)
        lines = []
        for schema in self._schemas():
            for t in insp.get_table_names(schema=schema):
                try:
                    cols = insp.get_columns(t, schema=schema)
                    col_list = ", ".join(f"{c['name']} ({c['type']})" for c in cols)
                except Exception as e:
                    col_list = f"(could not inspect: {e})"
                lines.append(f"{schema}.{t}: {col_list}")
        return "\n".join(lines) if lines else "(no Excel/CSV files uploaded yet -- no tables exist)"

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


def make_sql_tool(db: MultiSchemaDB):
    """
    The one and only SQL tool. The agent already knows the schema (it's
    injected into the system prompt at build time), so this tool just
    takes a raw query -- no separate list-tables/describe-schema tools.
    The agent can run its own `SELECT ... LIMIT n` introspection queries
    if it wants sample values; it knows SQL, it doesn't need a wrapper for
    that.
    """

    @tool
    def sql_tool(query: str) -> str:
        """Execute a read-only SQL SELECT (or WITH ... SELECT) query
        against the user's uploaded Excel/CSV data and return the result.
        Tables must be qualified as `schema.table`, e.g. `example_1.sheet1`
        (schema = the uploaded file's name). Only SELECT/WITH statements
        are permitted -- anything else is rejected with an explanation,
        not executed."""
        safe, reason = is_safe_query(query)
        if not safe:
            return reason

        limited_query = enforce_row_limit(query)
        try:
            return db.run(limited_query)
        except Exception as e:
            return f"ERROR executing query: {e}"

    return sql_tool