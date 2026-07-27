import os
from typing import Optional

from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from prompts import SYSTEM_PROMPT
from rag import build_vectorstore, search_documents
from guardrails import build_readonly_db, make_safe_query_tool, make_schema_tools
from file_registry import list_files, list_sql_schemas

GROQ_MODEL = "openai/gpt-oss-120b"


def _make_file_info_tool(session_id: str):
    @tool
    def list_uploaded_files() -> str:
        """List every file the user has uploaded this session: name, type,
        when it was uploaded, and how it was stored. Excel/CSV files show
        their schema alias and sheet/table names; PDF/Word files show their
        chunk count. Use this when the user asks what they've uploaded,
        what data is available, or when you're unsure whether a question
        is about tabular data or a document."""
        files = list_files(session_id)
        if not files:
            return "No files have been uploaded yet in this session."
        lines = []
        for f in files:
            header = f"- {f['filename']} ({f['filetype']}, uploaded {f['upload_date']})"
            if f["status"] != "processed":
                lines.append(f"{header}: FAILED ({f['error_message']})")
            elif f["storage_type"] == "sql":
                lines.append(
                    f"{header} -> schema `{f['alias']}`, "
                    f"{len(f['tables_created'])} table(s): {', '.join(f['tables_created'])}"
                )
            elif f["storage_type"] == "vector":
                lines.append(f"{header} -> {f['chunk_count']} chunk(s) in the document search index")
        return "\n".join(lines)

    return list_uploaded_files


def build_agent(
    session_id: str,
    glossary_text: Optional[str] = None,
    checkpointer: Optional[InMemorySaver] = None,
):
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY not set. Copy .env.example to .env and add your key."
        )

    model = ChatGroq(model=GROQ_MODEL, temperature=0)

    # Every uploaded Excel/CSV file lives in its own .db file; attach all of
    # them under their alias so tables are queryable as `alias.table`.
    schemas = list_sql_schemas(session_id)
    db = build_readonly_db(schemas)

    sql_tools = make_schema_tools(db) + [make_safe_query_tool(db)]

    vectorstore = build_vectorstore(db, session_id, glossary_text=glossary_text)

    @tool
    def search_uploaded_documents(
        query: str,
        num_results: int = 5,
        files: Optional[list[str]] = None,
    ) -> str:
        """Search the content of uploaded PDF/Word documents (NOT Excel/CSV
        data -- use the SQL tools for that). Only call this if the user has
        actually uploaded a PDF/Word file this session (check
        list_uploaded_files if unsure) -- there is nothing to search
        otherwise.

        Args:
            query: what to search for.
            num_results: how many chunks to return (default 5).
            files: optional list of filenames to restrict the search to,
                e.g. ["policy.pdf"]. Omit to search all uploaded documents.
        """
        return search_documents(vectorstore, query, num_results=num_results, files=files)

    file_info_tool = _make_file_info_tool(session_id)

    all_tools = sql_tools + [search_uploaded_documents, file_info_tool]

    # Reuse the same checkpointer across rebuilds so conversation memory
    # survives when a new file is uploaded mid-chat.
    checkpointer = checkpointer or InMemorySaver()
    agent = create_agent(
        model,
        all_tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
    return {
        "agent": agent,
        "db": db,
        "vectorstore": vectorstore,
        "checkpointer": checkpointer,
        "tables": db.get_usable_table_names(),
    }


def run_agent(agent, question: str, config: dict):
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
    )

    # Build a name -> output map from ToolMessages, keyed by tool_call_id,
    # so each call can be paired with its result.
    outputs_by_call_id = {}
    for msg in result["messages"]:
        if getattr(msg, "type", None) == "tool":
            outputs_by_call_id[msg.tool_call_id] = msg.content

    tool_calls = []
    executed_queries = []
    used_rag = False
    for msg in result["messages"]:
        calls = getattr(msg, "tool_calls", None)
        if not calls:
            continue
        for tc in calls:
            tool_calls.append(
                {
                    "name": tc["name"],
                    "input": tc["args"],
                    "output": outputs_by_call_id.get(tc["id"], ""),
                }
            )
            if tc["name"] == "query_data_tables":
                executed_queries.append(tc["args"]["query"])
            if tc["name"] == "search_uploaded_documents":
                used_rag = True

    return result["messages"][-1].content, executed_queries, used_rag, tool_calls


def stream_agent(agent, question: str, config: dict):
    """
    Generator that yields live progress events while the agent runs, so the
    UI can render each step as it happens instead of waiting for the whole
    turn to finish.

    Yields dicts:
      {"type": "tool_call",   "name": str, "input": dict}
      {"type": "tool_result", "name": str, "output": str}
      {"type": "final", "content": str, "tool_calls": [...],
       "executed_queries": [...], "used_rag": bool}
    """
    calls_by_id: dict = {}
    ordered_calls: list = []
    executed_queries: list = []
    used_rag = False
    final_content = ""

    for step in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
        stream_mode="updates",
    ):
        for node_output in step.values():
            messages = node_output.get("messages", []) if isinstance(node_output, dict) else []
            for msg in messages:
                calls = getattr(msg, "tool_calls", None)
                if calls:
                    for tc in calls:
                        record = {"name": tc["name"], "input": tc["args"], "output": ""}
                        calls_by_id[tc["id"]] = record
                        ordered_calls.append(record)
                        yield {"type": "tool_call", "name": tc["name"], "input": tc["args"]}
                        if tc["name"] == "query_data_tables":
                            executed_queries.append(tc["args"]["query"])
                        if tc["name"] == "search_uploaded_documents":
                            used_rag = True
                elif getattr(msg, "type", None) == "tool":
                    record = calls_by_id.get(msg.tool_call_id)
                    name = record["name"] if record else "tool"
                    if record:
                        record["output"] = msg.content
                    yield {"type": "tool_result", "name": name, "output": msg.content}
                elif getattr(msg, "type", None) == "ai":
                    # Overwritten each time; the last AI message with no
                    # tool_calls is the final answer.
                    final_content = msg.content

    yield {
        "type": "final",
        "content": final_content,
        "tool_calls": ordered_calls,
        "executed_queries": executed_queries,
        "used_rag": used_rag,
    }