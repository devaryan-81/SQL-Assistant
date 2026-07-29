import json
import os
import tempfile
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agent import build_agent, run_agent, stream_agent
from Ingestion import ingest_file
from guardrails import is_safe_query, enforce_row_limit
from file_registry import list_files

load_dotenv()

st.set_page_config(page_title="Ask Your Data", page_icon="✳️", layout="wide")

# ---------------------------------------------------------------------------
# Theme: a small set of color tokens per mode, so dark/light stay in sync
# from one CSS template instead of two hand-maintained stylesheets. The
# toggle lives at the very top so its value is set before the CSS below
# reads it.
# ---------------------------------------------------------------------------
THEMES = {
    "dark": {
        "bg": "#18191A",
        "sidebar_bg": "#1E1F20",
        "border": "#2C2D2E",
        "text": "#ECECEC",
        "muted": "#8F8F8F",
        "muted2": "#7A7A7A",
        "code_bg": "#131415",
        "chat_bg": "#1E1F20",
        "input_bg": "#1E1F20",
        "tool_card_bg": "#1B1C1D",
        "file_row_bg": "#18191A",
        "button_bg": "#1E1F20",
        "accent": "#2DD4BF",
        "pill_bg": "rgba(45,212,191,0.12)",
        "pill_border": "rgba(45,212,191,0.35)",
        "feature_grad": "linear-gradient(135deg, #1E8F82 0%, #10534C 100%)",
        "feature_text": "#F3FBFA",
        "feature_desc": "#DCF3F0",
    },
    "light": {
        "bg": "#FFFFFF",
        "sidebar_bg": "#FAFAFA",
        "border": "#EDEDED",
        "text": "#1B1B1B",
        "muted": "#8A8A8A",
        "muted2": "#9A9A9A",
        "code_bg": "#F5F5F5",
        "chat_bg": "#FFFFFF",
        "input_bg": "#FFFFFF",
        "tool_card_bg": "#FAFAFA",
        "file_row_bg": "#FFFFFF",
        "button_bg": "#FFFFFF",
        "accent": "#0E8E80",
        "pill_bg": "#F0FBFA",
        "pill_border": "#CDEFEA",
        "feature_grad": "linear-gradient(135deg, #1FAE9E 0%, #0E7A6E 100%)",
        "feature_text": "#FFFFFF",
        "feature_desc": "#EAFBF8",
    },
}

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

_is_light = st.sidebar.toggle(
    "☀️ Light mode",
    value=(st.session_state.theme == "light"),
    key="theme_toggle",
)
st.session_state.theme = "light" if _is_light else "dark"
T = THEMES[st.session_state.theme]

# ---------------------------------------------------------------------------
# One place to describe each tool: icon, accent color, human label. Used
# for both the live "thinking" trace and the tool-call cards, so every
# tool is rendered the same way everywhere.
# ---------------------------------------------------------------------------
TOOL_META = {
    "get_user_info": {"icon": "📁", "color": "#7FB0E0", "bg": "rgba(127,176,224,0.14)", "label": "Files"},
    "sql_tool": {"icon": "◆", "color": "#2DD4BF", "bg": "rgba(45,212,191,0.14)", "label": "SQL Query"},
    "search_document_context": {"icon": "🔎", "color": "#E3A857", "bg": "rgba(227,168,87,0.14)", "label": "Document Search"},
}
DEFAULT_TOOL_META = {"icon": "🔧", "color": "#9A9A9A", "bg": "rgba(154,154,154,0.14)", "label": "Tool"}


def _tool_meta(name: str) -> dict:
    return TOOL_META.get(name, DEFAULT_TOOL_META)


def _pretty(value):
    """Best-effort pretty-print, exactly as received: dicts/lists become
    indented JSON; strings that happen to parse as JSON get reformatted as
    JSON; anything else is returned completely unmodified. Returns
    (text, language) for st.code."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, indent=2), "json"
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("{", "[")):
            try:
                return json.dumps(json.loads(stripped), indent=2), "json"
            except (json.JSONDecodeError, TypeError):
                pass
        return value, None
    return str(value), None


# ---------------------------------------------------------------------------
# Theme CSS, built from the token dict above -- matched to Perplexity's
# layout/style (serif hero heading, gradient feature cards, pill input),
# with a light variant added on top of the original dark match. (Built to
# match the visual style/layout from the reference screenshot -- not
# extracted from their actual code or brand assets.)
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {{ font-family: 'Inter', -apple-system, sans-serif; }}

    .stApp {{ background: {T['bg']}; color: {T['text']}; }}

    .block-container {{ max-width: 820px; padding-top: 2.5rem; }}

    h1, h2, h3 {{ font-weight: 600 !important; letter-spacing: -0.01em; color: {T['text']}; }}

    section[data-testid="stSidebar"] {{
        background: {T['sidebar_bg']};
        border-right: 1px solid {T['border']};
    }}
    section[data-testid="stSidebar"] .block-container {{ padding-top: 1.5rem; }}
    section[data-testid="stSidebar"] * {{ color: {T['text']}; }}

    [data-testid="stChatMessage"] {{
        background: {T['chat_bg']};
        border: 1px solid {T['border']};
        border-radius: 16px;
        padding: 0.85rem 1.1rem !important;
        margin-bottom: 0.6rem;
        color: {T['text']};
    }}

    [data-testid="stChatInput"] {{
        border-radius: 999px;
        background: {T['input_bg']};
        border: 1px solid {T['border']};
    }}
    [data-testid="stChatInput"] textarea {{
        border-radius: 999px !important;
        color: {T['text']} !important;
    }}

    .stButton>button, .stDownloadButton>button {{
        border-radius: 999px;
        border: 1px solid {T['border']};
        background: {T['button_bg']};
        color: {T['text']};
        font-weight: 500;
    }}
    .stButton>button:hover, .stDownloadButton>button:hover {{
        border-color: {T['accent']};
        color: {T['accent']};
    }}

    [data-testid="stExpander"] {{
        background: {T['chat_bg']};
        border: 1px solid {T['border']} !important;
        border-radius: 12px;
    }}

    [data-testid="stTextArea"] textarea, [data-testid="stSelectbox"] div, .stSlider {{
        color: {T['text']};
    }}

    code, pre, .stCode {{ font-family: 'JetBrains Mono', monospace !important; }}
    [data-testid="stCodeBlock"] {{ background: {T['code_bg']} !important; }}

    .hero-eyebrow {{
        font-size: 0.85rem;
        color: {T['muted']};
        margin-bottom: 0.35rem;
    }}
    .hero-heading {{
        font-family: 'Fraunces', serif;
        font-weight: 500;
        font-size: 2.4rem;
        color: {T['text']};
        margin-bottom: 1.6rem;
        line-height: 1.2;
    }}

    .feature-card {{
        background: {T['feature_grad']};
        border-radius: 16px;
        padding: 1.1rem 1.3rem;
        color: {T['feature_text']};
        height: 100%;
    }}
    .feature-card .fc-title {{
        font-weight: 600;
        font-size: 1.02rem;
        margin-bottom: 0.35rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    .feature-card .fc-desc {{
        font-size: 0.85rem;
        color: {T['feature_desc']};
        line-height: 1.4;
    }}

    .pill {{
        display: inline-block;
        background: {T['pill_bg']};
        border: 1px solid {T['pill_border']};
        color: {T['accent']};
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.76rem;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        margin: 0.15rem 0.3rem 0.15rem 0;
    }}

    .file-row {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: {T['file_row_bg']};
        border: 1px solid {T['border']};
        padding: 0.45rem 0.7rem;
        border-radius: 10px;
        margin-bottom: 0.35rem;
        font-size: 0.83rem;
        color: {T['text']};
    }}

    .tool-card {{
        border-left: 3px solid var(--tc, {T['accent']});
        background: {T['tool_card_bg']};
        border-radius: 0 10px 10px 0;
        padding: 0.45rem 0.85rem;
        margin: 0.5rem 0 0.25rem 0;
        display: flex;
        align-items: center;
        gap: 0.55rem;
    }}
    .tool-badge {{
        font-size: 0.78rem;
        font-weight: 600;
        padding: 0.15rem 0.65rem;
        border-radius: 999px;
        background: var(--tbg, {T['pill_bg']});
        color: var(--tc, {T['accent']});
    }}
    .tool-name {{ font-size: 0.8rem; color: {T['muted']}; }}
    .tool-name code {{ background: transparent; color: {T['muted']}; }}
    .tool-index {{ margin-left: auto; font-size: 0.75rem; color: {T['muted2']}; }}

    .section-label {{
        font-size: 0.72rem;
        font-weight: 600;
        color: {T['muted']};
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin: 0.35rem 0 0.15rem 0;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY not set. Add it to your .env file and restart.")
    st.stop()

# ---------------------------------------------------------------------------
# Session setup: each uploaded Excel/CSV file gets its own physical .db file
# under schemas_dir (e.g. schemas_dir/example_1.db), attached under its own
# alias at query time -- so uploads never mutate each other or bleed across
# sessions, and each file is queryable as `alias.table`.
# ---------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    tmp_dir = Path(tempfile.mkdtemp())
    st.session_state.upload_dir = tmp_dir
    st.session_state.schemas_dir = tmp_dir / "schemas"
    st.session_state.processed_filenames = set()  # (name, size) tuples
    st.session_state.messages = []
    st.session_state.bundle = None  # holds agent/db/vectorstore/checkpointer

session_id = st.session_state.session_id


def get_or_build_bundle():
    if st.session_state.bundle is None:
        with st.spinner("Setting up..."):
            st.session_state.bundle = build_agent(session_id=session_id)
    return st.session_state.bundle


def rebuild_bundle_preserving_memory():
    """Rebuild the agent so its system prompt picks up the latest schema
    and document status, but reuse the same checkpointer so conversation
    history is not lost."""
    old = st.session_state.bundle
    with st.spinner("Updating with new data..."):
        st.session_state.bundle = build_agent(
            session_id=session_id,
            checkpointer=old["checkpointer"] if old else None,
        )


# ---------------------------------------------------------------------------
# Sidebar: uploader + file list + Data Inspector.
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("#### Your files")
    st.caption("Excel/CSV becomes queryable tables. PDF/Word becomes searchable text.")

    uploaded = st.file_uploader(
        "Add file(s)",
        type=["csv", "xlsx", "xls", "pdf", "docx"],
        accept_multiple_files=True,
        key="uploader",
        label_visibility="collapsed",
    )

    if st.button("Reset session", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Process any newly-added files (dedup by name+size so reruns don't re-ingest)
new_results = []
if uploaded:
    bundle = get_or_build_bundle()
    for f in uploaded:
        marker = (f.name, f.size)
        if marker in st.session_state.processed_filenames:
            continue
        dest = st.session_state.upload_dir / f.name
        dest.write_bytes(f.getbuffer())
        result = ingest_file(
            dest,
            original_filename=f.name,
            schemas_dir=st.session_state.schemas_dir,
            vectorstore=bundle["vectorstore"],
            session_id=session_id,
        )
        new_results.append(result)
        st.session_state.processed_filenames.add(marker)

if new_results:
    # Rebuild every time (SQL or document) so the system prompt's schema
    # block and document-availability note always reflect the latest state.
    rebuild_bundle_preserving_memory()

    for r in new_results:
        if r["type"] == "sql":
            alias = r["detail"]["alias"]
            qualified = [f"{alias}.{t}" for t in r["detail"]["tables"]]
            note = f"📎 Added **{r['name']}** → schema `{alias}`, tables: {', '.join(qualified)}"
        elif r["type"] == "vector":
            note = f"📎 Added **{r['name']}** → {r['detail']} in the document search index"
        else:
            note = f"⚠️ Could not process **{r['name']}**: {r['detail']}"
        st.session_state.messages.append({"role": "assistant", "content": note})

bundle = get_or_build_bundle()
agent = bundle["agent"]
db = bundle["db"]
tables = bundle["tables"]

with st.sidebar:
    files_list = list_files(session_id)
    if files_list:
        st.markdown("---")
        for f in files_list:
            status_dot = "🟢" if f["status"] == "processed" else "🔴"
            kind = "table" if f["storage_type"] == "sql" else "document"
            st.markdown(
                f'<div class="file-row"><span>{status_dot} {f["name"]}</span>'
                f'<span style="color:{T["muted"]};font-size:0.75rem;">{kind}</span></div>',
                unsafe_allow_html=True,
            )

    with st.expander("🔍 Data Inspector", expanded=False):
        if not tables:
            st.caption("Upload an Excel/CSV file to browse its tables here.")
        else:
            st.caption("Browse a table directly, or run your own read-only SQL -- separate from the chat.")
            selected_table = st.selectbox("Table", tables, key="inspector_table")
            preview_rows = st.slider("Rows to preview", 5, 200, 25, key="inspector_rows")
            if st.button("Preview table", key="inspector_preview_btn"):
                try:
                    df = pd.read_sql(f"SELECT * FROM {selected_table} LIMIT {preview_rows}", db.engine)
                    st.dataframe(df, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not read {selected_table}: {e}")

            st.divider()
            custom_sql = st.text_area(
                "Custom SQL (read-only)",
                placeholder="SELECT * FROM example_1.sheet1 WHERE ...",
                key="inspector_sql",
            )
            if st.button("Run query", key="inspector_run_btn"):
                safe, reason = is_safe_query(custom_sql) if custom_sql.strip() else (False, "Enter a query first.")
                if not safe:
                    st.error(reason)
                else:
                    try:
                        df = pd.read_sql(enforce_row_limit(custom_sql), db.engine)
                        st.dataframe(df, use_container_width=True)
                    except Exception as e:
                        st.error(f"Query failed: {e}")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
if not st.session_state.messages:
    st.markdown('<div class="hero-eyebrow">Ask</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-heading">What do you want to know?</div>', unsafe_allow_html=True)

    fc1, fc2 = st.columns(2)
    with fc1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="fc-title">🗄️ Query your tables</div>
                <div class="fc-desc">Upload Excel or CSV files and ask questions in plain English —
                answered with real, read-only SQL.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with fc2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="fc-title">📄 Search your documents</div>
                <div class="fc-desc">Upload PDFs or Word docs and ask about their content —
                searched section by section, page-accurate.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.markdown("#### Ask your data")
    if tables:
        st.markdown("".join(f'<span class="pill">{t}</span>' for t in tables), unsafe_allow_html=True)

config = {"configurable": {"thread_id": session_id}}


def _render_tool_calls(tool_calls: list[dict], key_prefix: str):
    """
    One common card per tool call, identical structure for every tool:
    a colored badge naming the tool, then the EXACT input it was called
    with and the EXACT output it received, both pretty-printed as JSON
    where possible.
    """
    with st.expander(f"🔧 {len(tool_calls)} tool call{'s' if len(tool_calls) != 1 else ''}", expanded=False):
        for i, tc in enumerate(tool_calls, start=1):
            meta = _tool_meta(tc["name"])
            st.markdown(
                f"""
                <div class="tool-card" style="--tc:{meta['color']};--tbg:{meta['bg']}">
                    <span class="tool-badge">{meta['icon']} {meta['label']}</span>
                    <span class="tool-name"><code>{tc['name']}</code></span>
                    <span class="tool-index">#{i}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            input_str, input_lang = _pretty(tc["input"])
            st.markdown('<div class="section-label">Input</div>', unsafe_allow_html=True)
            st.code(input_str, language=input_lang or "json")

            output_str, output_lang = _pretty(tc["output"])
            st.markdown('<div class="section-label">Output</div>', unsafe_allow_html=True)
            if len(output_str) > 2000:
                show_full = st.toggle("Show full output", key=f"{key_prefix}-{i}", value=False)
                st.code(output_str if show_full else output_str[:2000] + " …", language=output_lang)
            else:
                st.code(output_str, language=output_lang)

            if i < len(tool_calls):
                st.markdown(f"<hr style='margin:0.6rem 0;border-color:{T['border']}'>", unsafe_allow_html=True)


for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("tool_calls"):
            _render_tool_calls(msg["tool_calls"], key_prefix=f"hist-{idx}")

question = st.chat_input("Ask anything about your data...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        status_box = st.status("Thinking...", expanded=True)
        answer, tool_calls = "", []
        for event in stream_agent(agent, question, config):
            if event["type"] == "tool_call":
                meta = _tool_meta(event["name"])
                status_box.write(f"{meta['icon']} Calling **{meta['label']}**")
            elif event["type"] == "tool_result":
                meta = _tool_meta(event["name"])
                status_box.write(f"✓ {meta['label']} returned a result")
            elif event["type"] == "final":
                answer = event["content"]
                tool_calls = event["tool_calls"]

        status_box.update(label="Done", state="complete", expanded=False)
        st.markdown(answer)
        if tool_calls:
            _render_tool_calls(tool_calls, key_prefix=f"live-{len(st.session_state.messages)}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "tool_calls": tool_calls}
    )