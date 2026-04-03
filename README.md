# Scribe — Chat with Your PDFs

> Upload any PDF. Ask anything. Get instant, cited answers — powered by Groq's LLaMA 3.3 70B.

<br>

## What it does
 
Scribe is a production-grade RAG (Retrieval-Augmented Generation) application that lets you have a real conversation with your documents. Upload one or more PDFs, and Scribe will:
 
- **Auto-summarise** every document the moment it's uploaded
- **Answer questions** grounded strictly in your document content — no hallucinations
- **Cite every answer** with the exact source file and page number
- **Stream responses** token-by-token like a real chat interface
- **Export your conversation** as a formatted Markdown file
<br>

## Features
 
| Feature | Details |
|---|---|
| ⚡ Fast inference | Groq's LPU delivers LLaMA 3.3 70B responses in ~2s |
| 📋 Auto-summary | Each uploaded PDF is summarised in 3–4 sentences before you ask anything |
| 📄 Page citations | Every answer shows `filename · page N` chips so you know exactly where the answer came from |
| 💬 Multi-turn memory | Conversation context is preserved across turns via `ConversationBufferMemory` |
| 🔍 Semantic search | FAISS vector store with `all-MiniLM-L6-v2` embeddings for high-precision chunk retrieval |
| ⬇️ Chat export | Download the full conversation with sources as a `.md` file |
| 🎨 Polished UI | Custom dark theme with DM Sans, streaming cursor, document cards, and status indicators |
<br>

## Tech Stack
 
| Layer | Technology |
|---|---|
| LLM | Groq · LLaMA 3.3 70B Versatile |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace |
| Vector Store | FAISS (in-memory, CPU) |
| RAG Framework | LangChain  `ConversationalRetrievalChain` |
| PDF Parsing | PyPDF2 with per-page metadata extraction |
| Frontend | Streamlit with custom CSS |
| Memory | `ConversationBufferMemory` |
<br>

## Architecture
 
```
PDF Upload
    │
    ▼
Per-page text extraction (PyPDF2)
    │
    ▼
Recursive chunking (1000 tokens, 150 overlap)
    │
    ▼
Embedding (all-MiniLM-L6-v2) → FAISS index
    │
    ▼
User question → Semantic retrieval (top-4 chunks)
    │
    ▼
LLaMA 3.3 70B via Groq (streaming)
    │
    ▼
Answer + page-level citations → Streamlit UI
```
<br>
 
## Getting Started
 
### 1. Clone the repo
 
```bash
git clone https://github.com/Sanya003/Scribe
cd Scribe
```
 
### 2. Install dependencies
 
```bash
pip install -r requirements.txt
```
 
### 3. Set up environment variables
 
Create a `.env` file in the root directory:
 
```env
GROQ_API_KEY=your_groq_api_key_here
```
 
### 4. Run the app
 
```bash
streamlit run app.py
```
<br>

## Usage
 
1. Upload one or more PDF files using the sidebar
2. Click **⚡ Analyse Documents** — Scribe will index and summarise your files
3. Ask any question in the chat input
4. Answers stream in real time with page-level source citations
5. Click **⬇️ Export Chat** to download your conversation
 
<br>
 
## Project Structure
 
```
Scribe/
├── app.py              # Main Streamlit app
├── requirements.txt    # Dependencies
├── .env                # API keys
└── README.md
```
<br>
 
 
## Roadmap
 
- [ ] Hybrid search (BM25 + semantic)
- [ ] Voice input via Whisper
- [ ] Highlight-level citations (jump to exact passage)
- [ ] Multi-language support
 
