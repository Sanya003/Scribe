import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.callbacks.base import BaseCallbackHandler
import datetime
import os

load_dotenv()

# Page config 
st.set_page_config(
    page_title="Scribe — Chat with your PDFs",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem 2rem; max-width: 100%; }
.stApp { background: #0d0f12; color: #e2e8f0; }

[data-testid="stSidebar"] {
    background: #111318 !important;
    border-right: 1px solid #1e2128;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }

.app-header { display: flex; align-items: center; gap: 12px; margin-bottom: 1.5rem; padding-bottom: 1rem; border-bottom: 1px solid #1e2128; }
.app-title { font-size: 1.4rem; font-weight: 600; color: #f1f5f9; letter-spacing: -0.02em; }
.app-subtitle { font-size: 0.78rem; color: #64748b; margin-top: 2px; font-family: 'DM Mono', monospace; }
.logo-badge { width: 38px; height: 38px; background: linear-gradient(135deg, #3b82f6, #8b5cf6); border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.1rem; flex-shrink: 0; }

.section-label { font-size: 0.68rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: #475569; margin-bottom: 0.6rem; font-family: 'DM Mono', monospace; }

.doc-card { background: #16191f; border: 1px solid #1e2128; border-radius: 10px; padding: 0.7rem 0.9rem; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 10px; }
.doc-name { font-size: 0.82rem; font-weight: 500; color: #cbd5e1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.doc-pages { font-size: 0.7rem; color: #475569; font-family: 'DM Mono', monospace; margin-top: 1px; }

.summary-card { background: #13161c; border: 1px solid #1e2128; border-left: 3px solid #3b82f6; border-radius: 10px; padding: 1rem 1.1rem; margin-bottom: 1rem; font-size: 0.83rem; color: #94a3b8; line-height: 1.6; }
.summary-title { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #3b82f6; margin-bottom: 0.5rem; font-family: 'DM Mono', monospace; }

.chat-container { max-width: 780px; margin: 0 auto; }

.msg-row { display: flex; gap: 12px; margin-bottom: 1.2rem; align-items: flex-start; }
.msg-row.user { flex-direction: row-reverse; }
.avatar { width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.85rem; flex-shrink: 0; }
.avatar.ai { background: linear-gradient(135deg, #3b82f6, #8b5cf6); }
.avatar.usr { background: #1e293b; border: 1px solid #334155; }
.bubble { max-width: 75%; padding: 0.85rem 1.1rem; border-radius: 14px; font-size: 0.88rem; line-height: 1.65; }
.bubble.ai { background: #16191f; border: 1px solid #1e2128; color: #cbd5e1; border-top-left-radius: 4px; }
.bubble.usr { background: #1d3a5f; border: 1px solid #2563eb33; color: #e2e8f0; border-top-right-radius: 4px; }

.citations { margin-top: 0.6rem; display: flex; flex-wrap: wrap; gap: 6px; }
.cite-chip { background: #0f172a; border: 1px solid #1e3a5f; border-radius: 20px; padding: 2px 10px; font-size: 0.7rem; color: #60a5fa; font-family: 'DM Mono', monospace; }

.empty-state { text-align: center; padding: 4rem 2rem; color: #334155; }
.empty-icon { font-size: 2.5rem; margin-bottom: 1rem; }
.empty-title { font-size: 1.1rem; font-weight: 500; color: #475569; margin-bottom: 0.4rem; }
.empty-sub { font-size: 0.82rem; color: #334155; }

.stTextInput input, .stChatInput textarea {
    background: #16191f !important; border: 1px solid #1e2128 !important;
    border-radius: 10px !important; color: #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 0.88rem !important;
}
[data-testid="stChatInput"] { background: #16191f !important; border: 1px solid #1e2128 !important; border-radius: 12px !important; }

.stButton button { background: #16191f !important; border: 1px solid #1e2128 !important; border-radius: 8px !important; color: #94a3b8 !important; font-family: 'DM Sans', sans-serif !important; font-size: 0.82rem !important; font-weight: 500 !important; }
.stButton button:hover { border-color: #3b82f6 !important; color: #60a5fa !important; }

.process-btn button { background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important; border: none !important; color: white !important; font-weight: 600 !important; width: 100%; }

.status-badge { display: inline-flex; align-items: center; gap: 5px; background: #0f2a1a; border: 1px solid #166534; border-radius: 20px; padding: 3px 10px; font-size: 0.7rem; color: #4ade80; font-family: 'DM Mono', monospace; }
.status-dot { width: 6px; height: 6px; background: #4ade80; border-radius: 50%; animation: pulse 2s infinite; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }

.divider { height: 1px; background: #1e2128; margin: 1.2rem 0; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: #1e2128; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# Streaming handler 
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")

    def on_llm_end(self, *args, **kwargs):
        self.container.markdown(self.text)


# Cached embeddings (loads once, reused across sessions)
@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def get_llm(streaming=False, handler=None):
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        streaming=streaming,
        callbacks=[handler] if handler else [],
        groq_api_key=os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
    )


# PDF helpers 
def extract_pages(pdf_docs):
    pages, meta = [], {}
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        meta[pdf.name] = len(reader.pages)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"text": text, "page": i + 1, "source": pdf.name})
    return pages, meta


def build_vectorstore(pages):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks, metadatas = [], []
    for p in pages:
        for chunk in splitter.split_text(p["text"]):
            chunks.append(chunk)
            metadatas.append({"source": p["source"], "page": p["page"]})
    return FAISS.from_texts(texts=chunks, embedding=load_embeddings(), metadatas=metadatas)


def summarize_doc(pages, filename):
    excerpt = " ".join(p["text"] for p in pages if p["source"] == filename)[:3000]
    response = get_llm().invoke(
        f"Summarize this document in 3-4 concise sentences. Be direct:\n\n{excerpt}"
    )
    return response.content


def build_chain(vectorstore):
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )
    return ConversationalRetrievalChain.from_llm(
        llm=get_llm(),
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        return_source_documents=True,
    )


def export_chat(messages):
    lines = [f"# Scribe Chat Export\n*{datetime.datetime.now().strftime('%d %b %Y, %H:%M')}*\n\n---\n"]
    for m in messages:
        role = "**You**" if m["role"] == "user" else "**Scribe**"
        lines.append(f"{role}\n{m['content']}\n")
        if m.get("sources"):
            lines.append("*Sources: " + ", ".join(m["sources"]) + "*\n")
        lines.append("\n")
    return "\n".join(lines)


# Session state
defaults = {"chain": None, "messages": [], "summaries": {}, "doc_meta": {}, "processed": False}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# Sidebar
with st.sidebar:
    st.markdown("""
    <div class="app-header">
        <div class="logo-badge">📄</div>
        <div>
            <div class="app-title">Scribe</div>
            <div class="app-subtitle">RAG · Groq · FAISS</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Upload Documents</div>', unsafe_allow_html=True)
    pdf_docs = st.file_uploader("Drop PDFs here", accept_multiple_files=True, type=["pdf"], label_visibility="collapsed")

    st.markdown("<div style='margin-top:0.8rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="process-btn">', unsafe_allow_html=True)
    process = st.button("⚡  Analyse Documents", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if process and pdf_docs:
        with st.spinner("Reading pages..."):
            pages, meta = extract_pages(pdf_docs)
            st.session_state.doc_meta = meta
        with st.spinner("Building vector index..."):
            vs = build_vectorstore(pages)
            st.session_state.chain = build_chain(vs)
        with st.spinner("Summarising documents..."):
            st.session_state.summaries = {pdf.name: summarize_doc(pages, pdf.name) for pdf in pdf_docs}
        st.session_state.messages = []
        st.session_state.processed = True
        st.rerun()

    if st.session_state.doc_meta:
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Loaded Documents</div>', unsafe_allow_html=True)
        st.markdown('<div class="status-badge"><div class="status-dot"></div>Ready</div>', unsafe_allow_html=True)
        st.markdown("<div style='margin-top:0.6rem'></div>", unsafe_allow_html=True)
        for name, n in st.session_state.doc_meta.items():
            short = name[:28] + "…" if len(name) > 30 else name
            st.markdown(f"""
            <div class="doc-card">
                <span style="font-size:1rem">📄</span>
                <div><div class="doc-name">{short}</div><div class="doc-pages">{n} pages</div></div>
            </div>""", unsafe_allow_html=True)

    if st.session_state.messages:
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.download_button(
            "⬇️  Export Chat", export_chat(st.session_state.messages),
            file_name=f"scribe_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown", use_container_width=True
        )

    if st.session_state.processed:
        if st.button("🗑  Clear Session", use_container_width=True):
            for k in defaults:
                st.session_state[k] = defaults[k]
            st.rerun()


# Main
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

if not st.session_state.processed:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">📂</div>
        <div class="empty-title">No documents loaded yet</div>
        <div class="empty-sub">Upload PDFs in the sidebar and click Analyse to start chatting</div>
    </div>""", unsafe_allow_html=True)

else:
    # Auto-summaries
    for fname, summary in st.session_state.summaries.items():
        short = fname[:40] + "…" if len(fname) > 42 else fname
        st.markdown(f"""
        <div class="summary-card">
            <div class="summary-title">📋 Summary · {short}</div>
            {summary}
        </div>""", unsafe_allow_html=True)

    # Chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="msg-row user">
                <div class="avatar usr">👤</div>
                <div class="bubble usr">{msg["content"]}</div>
            </div>""", unsafe_allow_html=True)
        else:
            chips = "".join(f'<span class="cite-chip">📄 {s}</span>' for s in msg.get("sources", []))
            citations = f'<div class="citations">{chips}</div>' if chips else ""
            st.markdown(f"""
            <div class="msg-row">
                <div class="avatar ai">✦</div>
                <div>
                    <div class="bubble ai">{msg["content"]}</div>
                    {citations}
                </div>
            </div>""", unsafe_allow_html=True)

    # Input
    user_input = st.chat_input("Ask anything about your documents…")

    if user_input and st.session_state.chain:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.markdown(f"""
        <div class="msg-row user">
            <div class="avatar usr">👤</div>
            <div class="bubble usr">{user_input}</div>
        </div>""", unsafe_allow_html=True)

        # Streaming response
        st.markdown('<div class="msg-row"><div class="avatar ai">✦</div><div class="bubble ai">', unsafe_allow_html=True)
        stream_box = st.empty()
        st.markdown("</div></div>", unsafe_allow_html=True)

        handler = StreamHandler(stream_box)
        st.session_state.chain.combine_docs_chain.llm_chain.llm = get_llm(streaming=True, handler=handler)

        result = st.session_state.chain({"question": user_input})
        source_docs = result.get("source_documents", [])

        seen, sources = set(), []
        for doc in source_docs:
            m = doc.metadata
            label = f"{m.get('source','?')} · p{m.get('page','?')}"
            if label not in seen:
                seen.add(label)
                sources.append(label)

        if sources:
            chips = "".join(f'<span class="cite-chip">📄 {s}</span>' for s in sources)
            st.markdown(f'<div class="citations">{chips}</div>', unsafe_allow_html=True)

        st.session_state.messages.append({
            "role": "assistant",
            "content": handler.text,
            "sources": sources
        })

st.markdown("</div>", unsafe_allow_html=True)