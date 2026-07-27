import re
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from langchain_core.documents import Document

from file_registry import register_file, list_sql_schemas
from mistral_OCR import ocr_and_chunk

SUPPORTED_TABULAR = {".csv", ".xlsx", ".xls"}
SUPPORTED_DOCUMENT = {".pdf", ".docx"}


def _sanitize(name: str) -> str:
    """Lowercase, alphanumeric + underscore only. Used for both schema
    aliases (filenames) and table names (sheet names)."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", name).strip("_").lower()


def _unique_alias(base_alias: str, existing_aliases: set[str]) -> str:
    """Two files that sanitize to the same alias (e.g. 'Example-1.xlsx' and
    'Example_1.xlsx') would otherwise collide on the same schema name --
    append _2, _3, ... until it's unique."""
    if base_alias not in existing_aliases:
        return base_alias
    i = 2
    while f"{base_alias}_{i}" in existing_aliases:
        i += 1
    return f"{base_alias}_{i}"


def _ingest_tabular(
    file_path: Path,
    original_filename: str,
    schemas_dir: Path,
    session_id: str,
) -> dict:
    """
    Writes the file to its OWN physical sqlite database under schemas_dir,
    e.g. schemas_dir/example_1.db, with each sheet as a table inside it.
    This file is later ATTACHed under its alias so the agent can query
    `example_1.sheet1` directly.
    """
    schemas_dir.mkdir(parents=True, exist_ok=True)

    existing_aliases = {s["alias"] for s in list_sql_schemas(session_id)}
    base_alias = _sanitize(Path(original_filename).stem)
    alias = _unique_alias(base_alias, existing_aliases)

    db_path = schemas_dir / f"{alias}.db"
    write_engine = create_engine(f"sqlite:///{db_path}")
    tables_created = []

    try:
        if file_path.suffix.lower() == ".csv":
            df = pd.read_csv(file_path)
            table_name = "data"
            df.to_sql(table_name, write_engine, if_exists="replace", index=False)
            tables_created.append(table_name)
        else:
            sheets = pd.read_excel(file_path, sheet_name=None)
            for sheet_name, df in sheets.items():
                table_name = _sanitize(sheet_name) or "sheet1"
                df.to_sql(table_name, write_engine, if_exists="replace", index=False)
                tables_created.append(table_name)
    finally:
        write_engine.dispose()

    register_file(
        session_id=session_id,
        filename=original_filename,
        filetype=file_path.suffix.lower().lstrip("."),
        size_bytes=file_path.stat().st_size,
        storage_type="sql",
        alias=alias,
        db_path=str(db_path),
        tables_created=tables_created,
    )
    return {"alias": alias, "tables": tables_created}


def _ingest_document(file_path: Path, original_filename: str, vectorstore, session_id: str) -> int:
    """
    Runs Mistral OCR (markdown per page) then hierarchically chunks along
    headings/tables (see mistral_ocr.py) instead of blindly slicing every
    N characters -- this keeps each chunk semantically whole and preserves
    table structure.
    """
    chunks = ocr_and_chunk(file_path)
    docs = [
        Document(
            page_content=c["content"],
            metadata={
                "type": "user_document",
                "filename": original_filename,
                "chunk_index": i,
                "section": c["breadcrumb"],
                "page_start": c["page_start"],
                "page_end": c["page_end"],
                "session_id": session_id,
            },
        )
        for i, c in enumerate(chunks)
    ]
    if docs:
        vectorstore.add_documents(docs)

    register_file(
        session_id=session_id,
        filename=original_filename,
        filetype=file_path.suffix.lower().lstrip("."),
        size_bytes=file_path.stat().st_size,
        storage_type="vector",
        chunk_count=len(docs),
    )
    return len(docs)


def ingest_file(
    file_path: Path,
    original_filename: str,
    schemas_dir: Path,
    vectorstore,
    session_id: str,
) -> dict:
    """
    Returns a small result dict the UI can show as a confirmation message,
    e.g. {"type": "sql", "detail": {"alias": "example_1", "tables": [...]}}
    """
    ext = file_path.suffix.lower()
    try:
        if ext in SUPPORTED_TABULAR:
            result = _ingest_tabular(file_path, original_filename, schemas_dir, session_id)
            return {"filename": original_filename, "type": "sql", "detail": result}
        elif ext in SUPPORTED_DOCUMENT:
            n_chunks = _ingest_document(file_path, original_filename, vectorstore, session_id)
            return {"filename": original_filename, "type": "vector", "detail": f"{n_chunks} chunks"}
        else:
            register_file(
                session_id=session_id,
                filename=original_filename,
                filetype=ext.lstrip("."),
                size_bytes=file_path.stat().st_size,
                storage_type="unsupported",
                status="error",
                error_message="Unsupported file type",
            )
            return {"filename": original_filename, "type": "error", "detail": "Unsupported file type"}
    except Exception as e:
        register_file(
            session_id=session_id,
            filename=original_filename,
            filetype=ext.lstrip("."),
            size_bytes=file_path.stat().st_size,
            storage_type="error",
            status="error",
            error_message=str(e),
        )
        return {"filename": original_filename, "type": "error", "detail": str(e)}