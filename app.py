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

# Modern, spacious dashboard configuration
st.set_page_config(
    page_title="Data Console • AI Analytics", 
    page_icon="⚡", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# ENTERPRISE MATTE UI CORE STYLE
# ---------------------------------------------------------------------------
TOOL_STYLES = {
    "list_uploaded_files": {"icon": "📁", "color": "#6366F1", "label": "Files Index"},
    "list_data_tables": {"icon": "🗂️", "color": "#3B82F6", "label": "Tables Catalog"},
    "describe_table_schema": {"icon": "🔎", "color": "#8B5CF6", "label": "Schema Insight"},
    "query_data_tables": {"icon": "⚡", "color": "#10B981", "label": "Engine execution"},
    "search_uploaded_documents": {"icon": "📄", "color": "#F59E0B", "label": "Semantic Search"},
}
DEFAULT_TOOL_STYLE = {"icon": "🔧", "color": "#6B7280", "label": "System Utility"}
ERROR_MARKERS = ("ERROR", "Rejected:")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global Typography Reset */
    html, body, [class*="css"] { 
        font-family: 'Plus Jakarta Sans', sans-serif; 
    }
    .stApp { 
        background-color: #0F172A; 
        color: #F1F5F9; 
    }

    /* Elegant Structural Headers */
    h1, h2, h3, h4 { 
        font-family: 'Plus Jakarta Sans', sans-serif !important; 
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
    }
    
    /* Matte Sidebar Restyling */
    section[data-testid="stSidebar"] {
        background-color: #0B0F19 !important;
        border-right: 1px solid #1E293B !important;
    }

    /* Advanced Message Bubble Architecture */
    div[data-testid="stChatMessage"] {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        padding: 1rem !important;
        margin-bottom: 1rem;
    }
    div[data-testid="stChatMessage"] p {
        font-size: 0.95rem !important;
        line-height: 1.6 !important;
    }

    /* Fluent UI Components */
    .stButton > button, .stDownloadButton > button {
        background: #1E293B;
        color: #F1F5F9;
        border: 1px solid #334155;
        border-radius: 8px;
        font-weight: 500;
        padding: 0.5rem 1rem;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .stButton > button:hover {
        border-color: #38BDF8;
        color: #38BDF8;
        background: #1E293B;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.15);
    }

    /* Monospace Data Terminal Styling */
    div[data-testid="stJson"], .stCode, code, pre {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
    }
    div[data-testid="stJson"] {
        background-color: #020617 !important;
        border: 1px solid #1E293B !important;
        border-radius: 8px !important;
    }

    /* High-Fidelity Custom Layout Widgets */
    .metric-card {
        background: #131A2C;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        text-align: center;
    }
    .metric-val {
        font-size: 1.2rem;
        font-weight: 700;
        color: #38BDF8;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-lbl {
        font-size: 0.7rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .tool-badge-card {
        border: 1px solid #334155;
        border-left: 4px solid var(--tc, #3B82F6);
        background: #1E293B;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.5rem;
    }
    .badge-meta {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .badge-title {
        font-weight: 600;
        font-size: 0.9rem;
    }
    .status-indicator {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        text-transform: uppercase;
        font-weight: 500;
    }
    .status-ok { background: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .status-err { background: rgba(239, 68, 68, 0.15); color: #F87171; border: 1px solid rgba(239, 68, 68, 0.3); }

    .file-pill {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #131A2C;
        border: 1px solid #1E293B;
        padding: 0.5rem 0.75rem;
        border-radius: 6px;
        margin-bottom: 0.4rem;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Core Engine Session Handlers
# ---------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    tmp_dir = Path(tempfile.mkdtemp())
    st.session_state.upload_dir = tmp_dir
    st.session_state.schemas_dir = tmp_dir / "schemas"
    st.session_state.processed_filenames = set()
    st.session_state.messages = []
    st.session_state.bundle = None

session_id = st.session_state.session_id

def get_or_build_bundle(glossary_text=None):
    if st.session_state.bundle is None:
        with st.spinner("Initializing AI Core Context..."):
            st.session_state.bundle = build_agent(session_id=session_id, glossary_text=glossary_text)
    return st.session_state.bundle

def rebuild_bundle_preserving_memory():
    old = st.session_state.bundle
    with st.spinner("Refreshed schema parameters verified. Hot reloading agent..."):
        st.session_state.bundle = build_agent(
            session_id=session_id,
            checkpointer=old["checkpointer"] if old else None,
        )

# ---------------------------------------------------------------------------
# Sidebar Panel Context
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("📁 Data Asset Manager")
    st.caption("Upload source spreadsheets (CSV/XLSX) or dynamic documents (PDF/DOCX) to append schema profiles to the workspace session.")
    
    uploaded = st.file_uploader(
        "Ingest Documents",
        type=["csv", "xlsx", "xls", "pdf", "docx"],
        accept_multiple_files=True,
        key="uploader",
        label_visibility="collapsed"
    )
    
    # Process Files Cleanly
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
        if any(r["type"] == "sql" for r in new_results):
            rebuild_bundle_preserving_memory()
        for r in new_results:
            if r["type"] == "sql":
                alias = r["detail"]["alias"]
                qualified = [f"{alias}.{t}" for t in r["detail"]["tables"]]
                note = f"📎 **Structured Asset Ingested**: File `{r['filename']}` committed to workspace routing alias `{alias}`. Found tables: {', '.join(qualified)}."
            elif r["type"] == "vector":
                note = f"📎 **Unstructured Document Appended**: Ingested `{r['filename']}` containing {r['detail']} partitioned embedding segments."
            else:
                note = f"⚠️ **Ingestion Fault**: Matrix build engine parsing failure on `{r['filename']}`: {r['detail']}"
            st.session_state.messages.append({"role": "assistant", "content": note})

    st.markdown("---")
    
    # Inline Quick Asset Metric Grid
    files_list = list_files(session_id)
    sql_files_count = sum(1 for f in files_list if f["storage_type"] == "sql")
    vec_files_count = sum(1 for f in files_list if f["storage_type"] == "vector")
    
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{sql_files_count}</div><div class="metric-lbl">DB Schemas</div></div>', unsafe_allow_html=True)
    with m_col2:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{vec_files_count}</div><div class="metric-lbl">Vector Docs</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if files_list:
        with st.expander("📝 Attached Assets Bin", expanded=True):
            for f in files_list:
                status_icon = "🟢" if f["status"] == "processed" else "🔴"
                type_lbl = "Relational" if f["storage_type"] == "sql" else "Vector"
                st.markdown(
                    f'<div class="file-pill"><span>{status_icon} <b>{f["filename"]}</b></span>'
                    f'<span style="color:#64748B; font-size:0.75rem;">{type_lbl}</span></div>',
                    unsafe_allow_html=True
                )
                
    if st.button("↺ Flush Workspace Session", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ---------------------------------------------------------------------------
# Workspace Viewport Layout Setup
# ---------------------------------------------------------------------------
bundle = get_or_build_bundle()
agent = bundle["agent"]
db = bundle["db"]
tables = bundle["tables"]

# Modern App Header Elements
h_left, h_right = st.columns([3, 1])
with h_left:
    st.title("⚡ Enterprise Data Workspace")
    st.caption("Perform multi-modal conversational orchestration over raw files. Pipeline execution details trace natively below.")
with h_right:
    st.markdown(
        f'<div style="text-align:right; margin-top:1.5rem; font-family:\'JetBrains Mono\'; font-size:0.8rem; color:#64748B;">'
        f'SESSION TRACKING ID: <span style="color:#38BDF8;">{session_id[:8].upper()}</span></div>',
        unsafe_allow_html=True
    )

# Establish Primary Tab Viewports
tab_chat, tab_inspector = st.tabs(["💬 Dynamic Engine Agent", "🔍 System Schema Inspector"])

# ---------------------------------------------------------------------------
# Tab Viewport 1: Conversations & Traces
# ---------------------------------------------------------------------------
def _parse_output(output):
    if isinstance(output, (dict, list)):
        return output, True
    text = str(output)
    if text.strip().startswith(("{", "[")):
        try:
            return json.loads(text.strip()), True
        except Exception:
            pass
    return text, False

def render_ui_tool_cards(tool_calls: list[dict], iteration_key: str):
    if not tool_calls:
        return
    
    with st.expander(f"🛠️ Pipeline Subroutine Logs ({len(tool_calls)} Tool Actions Evaluated)", expanded=False):
        for idx, call in enumerate(tool_calls, start=1):
            style = TOOL_STYLES.get(call["name"], DEFAULT_TOOL_STYLE)
            out_str = str(call["output"])
            is_err = any(err in out_str for err in ERROR_MARKERS)
            badge_status = f'<span class="status-indicator status-err">Fault</span>' if is_err else f'<span class="status-indicator status-ok">Success</span>'
            
            st.markdown(
                f"""
                <div class="tool-badge-card" style="--tc: {style['color']}">
                    <div class="badge-meta">
                        <span style="font-size:1.1rem;">{style['icon']}</span>
                        <span class="badge-title">Subroutine call: <code style="color:#38BDF8;">{call['name']}</code></span>
                        <span style="color:#64748B; font-size:0.75rem; font-family:'JetBrains Mono';">#{idx}</span>
                    </div>
                    {badge_status}
                </div>
                """,
                unsafe_allow_html=True
            )
            
            req_col, res_col = st.columns(2)
            with req_col:
                st.markdown("<small style='color:#64748B; font-family:\"JetBrains Mono\";'>ROUTING INPUT BLOCK</small>", unsafe_allow_html=True)
                st.json(call["input"], expanded=True)
            with res_col:
                st.markdown("<small style='color:#64748B; font-family:\"JetBrains Mono\";'>PIPELINE RESPONSE STREAM</small>", unsafe_allow_html=True)
                parsed, is_json = _parse_output(call["output"])
                if is_json:
                    st.json(parsed, expanded=False)
                else:
                    if len(out_str) > 1500:
                        toggle_state = st.toggle("Expand Complete Payload", key=f"tgl-{iteration_key}-{idx}", value=False)
                        st.code(out_str if toggle_state else out_str[:1500] + "\n\n... [Truncated for visibility. Click above to view complete log stream] ...", language=None)
                    else:
                        st.code(out_str, language=None)
            if idx < len(tool_calls):
                st.markdown("<hr style='margin:0.5rem 0; border-color:#1E293B;' />", unsafe_allow_html=True)

with tab_chat:
    if not tables and not st.session_state.messages:
        st.info("👋 Ambient context empty. Upload tabular schemas or text corpuses via the left sidebar controls to instantiate context threads.")
        
    for idx, msg in enumerate(st.session_state.messages):
        avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg.get("tool_calls"):
                render_ui_tool_cards(msg["tool_calls"], iteration_key=f"hist-{idx}")

    question = st.chat_input("Query aggregated runtime knowledge layers...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(question)

        with st.chat_message("assistant", avatar="🤖"):
            status_box = st.status("Initializing Execution Subroutines...", expanded=True)
            answer, tool_calls = "", []
            
            for event in stream_agent(agent, question, config={"configurable": {"thread_id": session_id}}):
                if event["type"] == "tool_call":
                    style = TOOL_STYLES.get(event["name"], DEFAULT_TOOL_STYLE)
                    status_box.write(f"{style['icon']} Invoking sub-process allocation: `{event['name']}`")
                elif event["type"] == "tool_result":
                    status_box.write(f"✅ Evaluated context callback from `{event['name']}`.")
                elif event["type"] == "final":
                    answer = event["content"]
                    tool_calls = event["tool_calls"]

            status_box.update(label="Subroutines Cleared", state="complete", expanded=False)
            st.markdown(answer)
            if tool_calls:
                render_ui_tool_cards(tool_calls, iteration_key=f"live-{len(st.session_state.messages)}")

        st.session_state.messages.append({"role": "assistant", "content": answer, "tool_calls": tool_calls})

# ---------------------------------------------------------------------------
# Tab Viewport 2: Expanded Inspector View
# ---------------------------------------------------------------------------
with tab_inspector:
    if not tables:
        st.info("No active relational targets deployed. Upload relational sources to review live physical layout definitions.")
    else:
        st.subheader("📊 Deployed Table Schema Layout Navigator")
        
        c1, c2 = st.columns([1, 2])
        with c1:
            selected_table = st.selectbox("Select Active Physical Target Table Target", tables, key="inspect_tab_select")
            preview_rows = st.slider("Preview Rows Allocation Size", 5, 100, 20, key="inspect_tab_slider")
            
            st.markdown("---")
            st.markdown("### 🦺 Isolated Virtual Query Console")
            st.caption("Submit custom read-only queries directly down to the memory matrix pool without affecting the agent context.")
            custom_sql = st.text_area("SQL Statements Sandbox Input (Read-Only)", placeholder="SELECT * FROM file_alias.table_name LIMIT 10;", height=150)
            execute_custom = st.button("Execute Sandboxed Query Statement", use_container_width=True)
            
        with c2:
            st.markdown(f"#### Target Profiler Frame: `{selected_table}`")
            try:
                df_preview = pd.read_sql(f"SELECT * FROM {selected_table} LIMIT {preview_rows}", db.engine)
                st.dataframe(df_preview, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Failed to generate structured interface preview dataframes for target matrix: {e}")
                
            if execute_custom and custom_sql.strip():
                st.markdown("#### Sandbox Output Execution Matrix")
                safe, reason = is_safe_query(custom_sql)
                if not safe:
                    st.error(reason)
                else:
                    try:
                        df_custom = pd.read_sql(enforce_row_limit(custom_sql), db.engine)
                        st.dataframe(df_custom, use_container_width=True, hide_index=True)
                    except Exception as e:
                        st.error(f"Syntax validation error or execution exception encountered: {e}")
