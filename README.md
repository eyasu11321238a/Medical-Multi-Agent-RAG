# 🏥 Medical Research Assistant
### Multi-Agent System | LangGraph + Groq (Llama 3.1) + FAISS + SentenceTransformers

---

## 🏗️ Architecture

```
User Query (Streamlit UI)
        ↓
Supervisor Agent (LangGraph)
   ↓          ↓          ↓
Diagnosis  Treatment  Summarization
  Agent      Agent     /Q&A Agent
   ↓          ↓          ↓
      FAISS Vector Store (RAG)
      SentenceTransformers Embeddings
           ↓
      NCCN PDF Guidelines
```

---

## 🤖 Tech Stack

| Component | Technology |
|-----------|-----------|
| **LLM** | Groq API — `llama-3.1-8b-instant` |
| **Embeddings** | SentenceTransformers — `all-MiniLM-L6-v2` (local) |
| **Vector Store** | FAISS (local) |
| **Orchestration** | LangGraph |
| **UI** | Streamlit |

---

## 📁 Project Structure

```
medical_research_assistant/
├── .env                          # API keys & config
├── requirements.txt
├── app.py                        # ← Streamlit UI (run this)
├── main.py                       # CLI entry point
├── data/
│   ├── raw_pdfs/                 # Add NCCN PDFs here
│   └── faiss_index/              # Auto-generated
├── src/
│   ├── rag/
│   │   ├── pdf_ingestion.py      # PDF loading & chunking
│   │   └── vector_store.py       # FAISS + SentenceTransformers
│   ├── agents/
│   │   ├── supervisor.py         # Orchestrator
│   │   ├── diagnosis_agent.py    # Groq-powered diagnosis
│   │   ├── treatment_agent.py    # Groq-powered treatment
│   │   └── summarization_agent.py
│   ├── graph/
│   │   └── medical_graph.py      # LangGraph definition
│   └── utils/
│       └── helpers.py
└── tests/
    └── test_agents.py
```

---

## ⚙️ Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get a free Groq API key
👉 https://console.groq.com  
Add to `.env`:
```
GROQ_API_KEY=gsk_your_key_here
```

### 3. Add NCCN PDF guidelines
Download from: https://www.nccn.org/patientguidelines  
Place in `data/raw_pdfs/`:
```
data/raw_pdfs/
├── nccn_basal_cell_2026.pdf      ← already have this one
├── nccn_melanoma_2026.pdf
├── nccn_breast_cancer_2026.pdf
└── ...
```

### 4. Run

**Streamlit UI (recommended):**
```bash
streamlit run app.py
```

**CLI mode:**
```bash
python main.py
```

---

## 💬 Example Queries

```
"What are the signs and symptoms of basal cell skin cancer?"
"What surgery is used for high-risk BCC?"
"Compare melanoma vs basal cell cancer treatments"
"What is Mohs surgery?"
"Summarize the melanoma treatment guidelines"
```

---

## 🧪 Tests

```bash
python -m pytest tests/test_agents.py -v
```

---

## 🔧 Config (.env)

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | required | Get at console.groq.com |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embedding model |
| `CHUNK_SIZE` | `500` | PDF chunk size |
| `TOP_K_RESULTS` | `5` | Docs retrieved per query |
| `REBUILD_INDEX` | `false` | Force rebuild FAISS index |
