import hashlib
from pathlib import Path
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

BASE_PERSIST_DIR = Path(__file__).parent / "chroma_stores"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _store_dir_for_session(session_id: str) -> Path:
    """Unique, stable folder per session so different sessions don't share embeddings."""
    key = hashlib.sha256(session_id.encode()).hexdigest()[:16]
    return BASE_PERSIST_DIR / key


def has_uploaded_documents(vectorstore: Chroma) -> bool:
    """True if at least one PDF/Word chunk has been ingested this session.
    Used to tell the agent (via the system prompt) whether
    search_document_context has anything to search at all."""
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
    Searches uploaded PDF/Word chunks. If `files` is given, restricts the
    search to those filenames; otherwise searches every uploaded document
    in the session.
    """
    if files:
        where = {"$and": [{"type": "user_document"}, {"filename": {"$in": files}}]}
    else:
        where = {"type": "user_document"}

    results = vectorstore.similarity_search(query, k=num_results, filter=where)
    if not results:
        scope = f" in {', '.join(files)}" if files else ""
        return f"No matching content found{scope}. Check get_user_info to confirm the filename(s)."

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


def build_vectorstore(session_id: str) -> Chroma:
    """
    Opens (or creates) this session's persisted Chroma collection. Starts
    empty -- documents are added as they're uploaded via
    Ingestion._ingest_document -- so there's nothing to bootstrap here.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    persist_dir = _store_dir_for_session(session_id)
    return Chroma(
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
        collection_name="user_documents",
    )