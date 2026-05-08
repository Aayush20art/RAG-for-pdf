import streamlit as st
import os
import tempfile
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from groq import Groq

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Assistant",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load Groq API Key from Streamlit Secrets ───────────────────────────────────
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("❌ GROQ_API_KEY not found in Streamlit Secrets. Please add it in App Settings → Secrets.")
    st.stop()

# ── Custom CSS — Red & Black Theme ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #0a0a0a;
        color: #f0f0f0;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #111111;
        border-right: 1px solid #2a0000;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #ff2222;
    }

    /* ── Headings ── */
    h1, h2, h3 { color: #ff2222 !important; }
    h1 { font-size: 2rem !important; font-weight: 700 !important; }
    h2 { font-size: 1.4rem !important; }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #cc0000, #880000);
        color: #fff !important;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.2rem;
        transition: all 0.2s;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #ff1111, #aa0000);
        box-shadow: 0 4px 20px rgba(200,0,0,0.5);
        transform: translateY(-1px);
    }

    /* ── Text input & text area ── */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: #1a1a1a !important;
        border: 1px solid #3a0000 !important;
        border-radius: 8px !important;
        color: #f0f0f0 !important;
        font-size: 0.95rem;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #cc0000 !important;
        box-shadow: 0 0 0 2px rgba(200,0,0,0.25) !important;
    }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        background: #1a1a1a;
        border: 2px dashed #3a0000;
        border-radius: 10px;
        padding: 1rem;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #cc0000;
    }

    /* ── Cards ── */
    .rag-card {
        background: #161616;
        border: 1px solid #2a0000;
        border-left: 4px solid #cc0000;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin: 0.8rem 0;
    }
    .rag-card-success {
        border-left-color: #00cc44;
        background: #0d1a10;
    }
    .rag-card-source {
        background: #1a1212;
        border: 1px solid #2a0000;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
        font-size: 0.85rem;
        color: #bbbbbb;
    }

    /* ── Spinner ── */
    .stSpinner > div { border-top-color: #cc0000 !important; }

    /* ── Divider ── */
    hr { border-color: #2a0000 !important; }

    /* ── Metric ── */
    [data-testid="stMetricValue"] { color: #ff2222 !important; font-weight: 700; }

    /* ── Select box ── */
    .stSelectbox > div > div {
        background: #1a1a1a !important;
        border-color: #3a0000 !important;
        color: #f0f0f0 !important;
    }

    /* ── Logo ── */
    .logo-text {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ff2222;
        letter-spacing: -0.5px;
    }
    .logo-sub {
        font-size: 0.8rem;
        color: #888;
        margin-top: -4px;
    }

    /* ── Chat bubbles ── */
    .user-bubble {
        background: #1e0000;
        border: 1px solid #3a0000;
        border-radius: 12px 12px 0 12px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        color: #f0f0f0;
    }
    .ai-bubble {
        background: #130a0a;
        border: 1px solid #cc0000;
        border-radius: 12px 12px 12px 0;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        color: #f0f0f0;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State ───────────────────────────────────────────────────────────────
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False
if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = 0


# ── Helper Functions ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


def process_pdf(uploaded_file) -> int:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load_and_split()

    embeddings = st.session_state.embeddings
    vs = InMemoryVectorStore(embedding=embeddings)
    vs.add_documents(docs)

    st.session_state.vector_store = vs
    st.session_state.pdf_loaded = True
    st.session_state.doc_chunks = len(docs)

    os.unlink(tmp_path)
    return len(docs)


def retrieve_context(query: str, k: int = 4) -> list:
    return st.session_state.vector_store.similarity_search(query, k=k)


def ask_groq(query: str, context: str, model: str, history: list) -> str:
    client = Groq(api_key=GROQ_API_KEY)

    system_prompt = (
        "You are a helpful AI assistant that answers questions strictly based on the "
        "provided PDF context. If the answer is not found in the context, say so clearly. "
        "Be concise, accurate, and structured in your responses."
    )

    messages = [{"role": "system", "content": system_prompt}]
    for turn in history[-6:]:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})

    messages.append({
        "role": "user",
        "content": f"Context from PDF:\n{context}\n\n---\nQuestion: {query}"
    })

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].message.content


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR  (no API key field — loaded from secrets)
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="logo-text">🔴 RAG PDF</div>', unsafe_allow_html=True)
    st.markdown('<div class="logo-sub">Powered by Groq + HuggingFace</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### ⚙️ Settings")
    groq_model = st.selectbox(
        "Model",
        options=[
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ],
        index=0,
    )
    top_k = st.slider("Retrieval Top-K chunks", min_value=2, max_value=8, value=4)

    st.markdown("---")

    st.markdown("### 📄 Upload PDF")
    uploaded_pdf = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_pdf:
        if st.button("🚀 Process PDF"):
            with st.spinner("Loading embeddings model..."):
                if st.session_state.embeddings is None:
                    st.session_state.embeddings = load_embeddings()
            with st.spinner("Embedding PDF chunks..."):
                n_chunks = process_pdf(uploaded_pdf)
            st.markdown(
                f'<div class="rag-card rag-card-success">✅ <b>{uploaded_pdf.name}</b> processed!<br>'
                f'<small>{n_chunks} chunks indexed.</small></div>',
                unsafe_allow_html=True,
            )

    if st.session_state.pdf_loaded:
        st.markdown("---")
        st.markdown("### 📊 Status")
        col1, col2 = st.columns(2)
        col1.metric("Chunks", st.session_state.doc_chunks)
        col2.metric("Q&A Turns", len(st.session_state.chat_history))

    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

    if st.button("♻️ Reset All"):
        st.session_state.vector_store = None
        st.session_state.embeddings = None
        st.session_state.chat_history = []
        st.session_state.pdf_loaded = False
        st.session_state.doc_chunks = 0
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🔴 RAG PDF Assistant")
st.markdown("Upload a PDF → Ask anything about it. Powered by **Groq LLM** + **HuggingFace Embeddings**.")
st.markdown("---")

if not st.session_state.pdf_loaded:
    st.markdown(
        '<div class="rag-card">📌 <b>Getting Started:</b><br>'
        '1. Upload a <b>PDF file</b> from the sidebar.<br>'
        '2. Click <b>Process PDF</b>.<br>'
        '3. Ask your questions below!</div>',
        unsafe_allow_html=True,
    )

if st.session_state.chat_history:
    st.markdown("### 💬 Conversation")
    for turn in st.session_state.chat_history:
        st.markdown(f'<div class="user-bubble">🙋 <b>You:</b> {turn["user"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="ai-bubble">🤖 <b>Assistant:</b><br>{turn["assistant"]}</div>', unsafe_allow_html=True)

        if turn.get("sources"):
            with st.expander("📎 Source Chunks Used"):
                for i, src in enumerate(turn["sources"], 1):
                    pg = src.metadata.get("page", "?")
                    snippet = src.page_content[:300].replace("\n", " ")
                    st.markdown(
                        f'<div class="rag-card-source">📄 <b>Chunk {i}</b> — Page {pg}<br>{snippet}…</div>',
                        unsafe_allow_html=True,
                    )
    st.markdown("---")

if st.session_state.pdf_loaded:
    st.markdown("### ❓ Ask a Question")
    with st.form(key="qa_form", clear_on_submit=True):
        user_query = st.text_area(
            "Your question",
            placeholder="e.g. Summarize the document. / What is mentioned about XYZ?",
            height=100,
        )
        submitted = st.form_submit_button("🔍 Ask")

    if submitted and user_query.strip():
        with st.spinner("Retrieving context & generating answer..."):
            sources = retrieve_context(user_query, k=top_k)
            context = "\n\n".join([doc.page_content for doc in sources])
            answer = ask_groq(
                query=user_query,
                context=context,
                model=groq_model,
                history=st.session_state.chat_history,
            )

        st.session_state.chat_history.append({
            "user": user_query,
            "assistant": answer,
            "sources": sources,
        })
        st.rerun()

else:
    st.info("⬅️ Upload and process a PDF from the sidebar to start chatting.")