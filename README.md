# ⚡ AI SQL Assistant & Document RAG Console

An enterprise-grade, high-fidelity AI-powered SQL Assistant and Document RAG application. This system integrates tabular data analysis and semantic document search inside a modern Streamlit interface, utilizing LangChain/LangGraph, Groq LLMs, Mistral OCR, and Chroma vector stores.

---

## 🚀 Key Features

*   **Multi-Schema SQL Database Engine**: Automatically ingests uploaded Excel (`.xlsx`, `.xls`) or CSV files into dedicated SQLite databases and dynamically attaches them under custom schemas (e.g. `schema.table`), allowing the LLM to write multi-table joins across completely different uploaded files.
*   **Hierarchical Document RAG**: Processes uploaded PDF and Word documents (`.pdf`, `.docx`) using **Mistral OCR** (retaining headers, list hierarchies, and tables in Markdown) and chunks them contextually before indexing into a local **Chroma DB** with **Sentence Transformers** (`all-MiniLM-L6-v2`).
*   **Strict SQL Guardrails**: Protects data integrity by blocking unsafe keywords (`INSERT`, `DROP`, `UPDATE`, etc.) and enforcing dynamic row limits on results.
*   **Stateful Conversational Agent**: Employs **LangGraph** memory checkpointers so conversation history is preserved during agent updates or when adding files mid-chat.
*   **Modern Matte Dashboard UI**: A fully styled, responsive dark-themed dashboard featuring custom indicators, file indexes, metrics cards, and a real-time execution log.

---

## 🛠️ Tech Stack

*   **UI Framework**: Streamlit
*   **Orchestration**: LangChain & LangGraph
*   **Reasoning LLM**: Groq API
*   **Semantic Search**: Chroma DB & HuggingFace Embeddings
*   **OCR Engine**: Mistral OCR API
*   **Data Layer**: SQLAlchemy & Pandas

---

## 📦 Directory Structure

```text
SQL Assistant/
├── Ingestion.py          # Tabular SQLite injection and Document OCR/chunking
├── agent.py              # LangGraph agent definitions & tool initialization
├── app.py                # Streamlit UI, custom styles, and session loop
├── file_registry.py      # SQLite-backed session file index registry
├── guardrails.py         # Multi-database execution engine and safety guards
├── mistral_OCR.py        # Mistral OCR handler and Markdown-aware parsing
├── prompts.py            # System instructions and prompt templates
├── rag.py                # Chroma DB setup and semantic vector search
├── .gitignore            # Git exclusion patterns
├── .env.example          # Local configuration template
└── requirements.txt      # Project dependencies
```

---

## ⚙️ Setup & Installation

### 1. Clone & Navigate
```bash
git clone <your-repo-url>
cd "SQL Assistant"
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your API credentials:
```bash
cp .env.example .env
```
Open `.env` and configure:
```ini
GROQ_API_KEY=gsk_your_actual_key_here
MISTRAL_API_KEY=your_actual_key_here
```
> [!WARNING]
> Never commit your `.env` file to version control. The `.gitignore` is configured to prevent it.

### 3. Install Dependencies
Make sure your virtual environment is active, then run:
```bash
pip install -r requirements.txt
```

### 4. Run the Dashboard
Start the Streamlit application:
```bash
streamlit run app.py
```

---

## 🛡️ Security & Guardrails
*   **Read-Only Queries**: All SQL execution tools run with connection-level read-only configurations where possible, backed by a lexical scanner matching database modification keywords.
*   **Truncated Results**: Query results are capped at `200` rows by default to prevent LLM memory overflows and token exhaustion.
