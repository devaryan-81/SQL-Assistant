import hashlib
from pathlib import Path
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_PERSIST_DIR = Path(__file__).parent / "chroma_stores"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _store_dir_for_session(session_id: str) -> Path:
    """Unique, stable folder per session so different sessions don't share embeddings."""
    key = hashlib.sha256(session_id.encode()).hexdigest()[:16]
    return BASE_PERSIST_DIR / key


def auto_generate_schema_docs(db) -> list[Document]:
    """
    Build one Document per table using the DB's own metadata + sample rows.
    Works with anything exposing get_usable_table_names()/get_table_info()
    (e.g. guardrails.MultiSchemaDB) -- no manual per-schema writing required.
    """
    docs = []
    for table_name in db.get_usable_table_names():
        info = db.get_table_info([table_name])
        docs.append(
            Document(
                page_content=info,
                metadata={"table": table_name, "type": "auto_schema"},
            )
        )
    return docs


def glossary_docs_from_text(text: str, source_name: str = "user_glossary") -> list[Document]:
    """
    Optional: chunk user-supplied business-rules / glossary text into
    Documents. Use this when the user uploads a README, data dictionary,
    or business-rules doc alongside their database.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    return [
        Document(page_content=chunk, metadata={"type": "business_rule", "source": source_name})
        for chunk in chunks
    ]


def has_uploaded_documents(vectorstore: Chroma) -> bool:
    """True if at least one PDF/Word chunk has been ingested this session.
    Used to decide whether search_uploaded_documents is worth calling at
    all -- schema/business-rule docs don't count."""
    try:
        got = vectorstore.get(where={"type": "user_document"}, limit=1)
        return bool(got and got.get("ids"))
    except Exception:
        return False


def search_documents(
    vectorstore: Chroma,
    query: str,
    num_results: int = 5,
    files: Optional[list[str]] = None,
) -> str:
    """
    Searches only actual uploaded PDF/Word chunks (never the auto-generated
    schema docs -- those are covered by the SQL schema tools instead).
    If `files` is given, restricts the search to those filenames; otherwise
    searches across every uploaded document in the session.
    """
    if files:
        where = {"$and": [{"type": "user_document"}, {"filename": {"$in": files}}]}
    else:
        where = {"type": "user_document"}

    results = vectorstore.similarity_search(query, k=num_results, filter=where)
    if not results:
        scope = f" in {', '.join(files)}" if files else ""
        return f"No matching content found{scope}. Check list_uploaded_files to confirm the filename(s)."

    parts = []
    for i, doc in enumerate(results, start=1):
        fname = doc.metadata.get("filename", "unknown file")
        section = doc.metadata.get("section")
        page_start = doc.metadata.get("page_start")
        page_end = doc.metadata.get("page_end")
        loc_bits = [fname]
        if section:
            loc_bits.append(f"section: {section}")
        if page_start is not None:
            page_str = f"page {page_start}" if page_start == page_end else f"pages {page_start}-{page_end}"
            loc_bits.append(page_str)
        parts.append(f"[{i}] from {' | '.join(loc_bits)}:\n{doc.page_content}")
    return "\n\n".join(parts)


def build_vectorstore(
    db,
    session_id: str,
    glossary_text: Optional[str] = None,
) -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    documents = auto_generate_schema_docs(db)
    if glossary_text:
        documents += glossary_docs_from_text(glossary_text)

    persist_dir = _store_dir_for_session(session_id)

    # A brand-new session starts with no tables and no glossary, so there
    # may be zero documents -- Chroma.from_documents([]) errors on an empty
    # list. Initialize an empty collection instead and add docs if present.
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
        collection_name="schema_docs",
    )
    if documents:
        vectorstore.add_documents(documents)
    return vectorstore