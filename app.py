import os
import tempfile
import traceback

import streamlit as st
from groq import Groq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NeonDoc · PDF Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Groq API key
# ---------------------------------------------------------------------------
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error(
        "GROQ_API_KEY was not found. Add it under Streamlit Cloud → Manage app → Settings → Secrets."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Neon / Cyber theme (animated)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

        :root {
            --bg: #05030d;
            --panel: rgba(14, 10, 30, 0.72);
            --neon-pink: #ff2bd6;
            --neon-cyan: #00f0ff;
            --neon-purple: #a855f7;
            --neon-lime: #b6ff3b;
            --text: #e9e6ff;
            --muted: #8f8aa8;
        }

        @property --angle {
            syntax: '<angle>';
            initial-value: 0deg;
            inherits: false;
        }

        /* ---------- Keyframes ---------- */
        @keyframes spin-border { to { --angle: 360deg; } }
        @keyframes gradient-flow {
            0%   { background-position: 0% 50%; }
            50%  { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        @keyframes float-a {
            0%, 100% { transform: translate(0, 0) scale(1); }
            33%      { transform: translate(8vw, 10vh) scale(1.15); }
            66%      { transform: translate(-6vw, 5vh) scale(0.9); }
        }
        @keyframes float-b {
            0%, 100% { transform: translate(0, 0) scale(1); }
            50%      { transform: translate(-10vw, -8vh) scale(1.2); }
        }
        @keyframes float-c {
            0%, 100% { transform: translate(0, 0) scale(1); }
            50%      { transform: translate(6vw, -12vh) scale(1.1); }
        }
        @keyframes grid-move {
            from { background-position: 0 0, 0 0; }
            to   { background-position: 0 60px, 60px 0; }
        }
        @keyframes neon-flicker {
            0%, 18%, 22%, 25%, 53%, 57%, 100% {
                text-shadow: 0 0 6px #fff, 0 0 14px var(--neon-pink), 0 0 30px var(--neon-pink), 0 0 60px var(--neon-purple);
            }
            20%, 24%, 55% { text-shadow: none; }
        }
        @keyframes pulse-glow {
            0%, 100% { box-shadow: 0 0 8px var(--neon-cyan), 0 0 0 0 rgba(0,240,255,0.5); }
            50%      { box-shadow: 0 0 18px var(--neon-cyan), 0 0 0 8px rgba(0,240,255,0); }
        }
        @keyframes slide-up {
            from { opacity: 0; transform: translateY(18px) scale(0.98); filter: blur(4px); }
            to   { opacity: 1; transform: translateY(0) scale(1); filter: blur(0); }
        }
        @keyframes shimmer {
            0%   { transform: translateX(-120%) skewX(-20deg); }
            100% { transform: translateX(260%) skewX(-20deg); }
        }
        @keyframes scan {
            0%   { top: -10%; }
            100% { top: 110%; }
        }
        @keyframes blink { 50% { opacity: 0; } }
        @keyframes bob {
            0%, 100% { transform: translateY(0); }
            50%      { transform: translateY(-6px); }
        }

        /* ---------- App background ---------- */
        .stApp {
            background:
                radial-gradient(1200px 600px at 10% -10%, rgba(168,85,247,0.18), transparent 60%),
                radial-gradient(900px 500px at 100% 0%, rgba(0,240,255,0.12), transparent 60%),
                var(--bg);
            color: var(--text);
            font-family: 'Inter', sans-serif;
        }
        header[data-testid="stHeader"] { background: transparent; }
        .block-container { padding-top: 2rem; position: relative; z-index: 2; }

        /* Moving neon grid */
        .stApp::before {
            content: "";
            position: fixed; inset: 0;
            background-image:
                linear-gradient(rgba(168,85,247,0.10) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0,240,255,0.08) 1px, transparent 1px);
            background-size: 60px 60px;
            animation: grid-move 6s linear infinite;
            mask-image: radial-gradient(ellipse at center, #000 30%, transparent 80%);
            -webkit-mask-image: radial-gradient(ellipse at center, #000 30%, transparent 80%);
            pointer-events: none; z-index: 0;
        }
        /* Scanline */
        .stApp::after {
            content: "";
            position: fixed; left: 0; right: 0; height: 120px;
            background: linear-gradient(180deg, transparent, rgba(0,240,255,0.06), transparent);
            animation: scan 7s linear infinite;
            pointer-events: none; z-index: 1;
        }

        /* Floating aurora orbs */
        .orb {
            position: fixed; border-radius: 50%;
            filter: blur(90px); opacity: 0.45;
            pointer-events: none; z-index: 0;
        }
        .orb.a { width: 38vw; height: 38vw; left: -8vw; top: 5vh;
                 background: var(--neon-purple); animation: float-a 18s ease-in-out infinite; }
        .orb.b { width: 32vw; height: 32vw; right: -6vw; top: 30vh;
                 background: var(--neon-cyan); opacity: 0.30; animation: float-b 22s ease-in-out infinite; }
        .orb.c { width: 28vw; height: 28vw; left: 35vw; bottom: -10vh;
                 background: var(--neon-pink); opacity: 0.28; animation: float-c 26s ease-in-out infinite; }

        /* ---------- Hero ---------- */
        .hero { text-align: center; padding: 1.2rem 0 0.6rem; animation: slide-up 0.9s ease both; }
        .hero .badge {
            display: inline-flex; align-items: center; gap: 8px;
            font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;
            letter-spacing: 0.18em; text-transform: uppercase;
            color: var(--neon-cyan);
            padding: 6px 14px; border-radius: 999px;
            border: 1px solid rgba(0,240,255,0.45);
            background: rgba(0,240,255,0.06);
        }
        .hero .badge .dot {
            width: 8px; height: 8px; border-radius: 50%;
            background: var(--neon-cyan); animation: pulse-glow 1.8s ease-in-out infinite;
        }
        .hero h1.title {
            font-family: 'Orbitron', sans-serif !important;
            font-weight: 900 !important;
            font-size: clamp(2.4rem, 6vw, 4.6rem) !important;
            letter-spacing: 0.06em;
            margin: 0.9rem 0 0.2rem !important;
            color: #fff !important;
            animation: neon-flicker 5s infinite, bob 5s ease-in-out infinite;
        }
        .hero h1.title span {
            background: linear-gradient(90deg, var(--neon-pink), var(--neon-purple), var(--neon-cyan), var(--neon-pink));
            background-size: 300% 100%;
            -webkit-background-clip: text; background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradient-flow 6s linear infinite;
            filter: drop-shadow(0 0 14px rgba(255,43,214,0.55));
        }
        .hero p.sub { color: var(--muted); font-size: 1.02rem; margin-top: 0.3rem; }
        .hero p.sub b { color: var(--neon-cyan); font-weight: 500; }

        h1, h2, h3 {
            font-family: 'Orbitron', sans-serif !important;
            color: var(--neon-purple) !important;
            letter-spacing: 0.04em;
        }

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(10,6,24,0.92), rgba(6,4,16,0.96));
            border-right: 1px solid transparent;
            border-image: linear-gradient(180deg, var(--neon-pink), var(--neon-purple), var(--neon-cyan)) 1;
            backdrop-filter: blur(10px);
        }
        section[data-testid="stSidebar"] h3 {
            text-shadow: 0 0 12px rgba(168,85,247,0.8);
        }
        .chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 6px 0 14px; }
        .chip {
            font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
            padding: 5px 10px; border-radius: 8px;
            color: var(--neon-lime);
            border: 1px solid rgba(182,255,59,0.45);
            background: rgba(182,255,59,0.07);
            box-shadow: 0 0 10px rgba(182,255,59,0.18);
            animation: slide-up 0.6s ease both;
        }
        .chip.cyan { color: var(--neon-cyan); border-color: rgba(0,240,255,0.45);
                     background: rgba(0,240,255,0.07); box-shadow: 0 0 10px rgba(0,240,255,0.2); }
        .chip.pink { color: var(--neon-pink); border-color: rgba(255,43,214,0.45);
                     background: rgba(255,43,214,0.07); box-shadow: 0 0 10px rgba(255,43,214,0.2); }

        /* ---------- Rotating neon border: uploader ---------- */
        [data-testid="stFileUploader"] > section,
        [data-testid="stFileUploaderDropzone"] {
            border: 2px solid transparent !important;
            border-radius: 16px !important;
            background:
                linear-gradient(var(--panel), var(--panel)) padding-box,
                conic-gradient(from var(--angle), var(--neon-pink), var(--neon-purple), var(--neon-cyan), var(--neon-lime), var(--neon-pink)) border-box !important;
            animation: spin-border 4s linear infinite;
            box-shadow: 0 0 25px rgba(168,85,247,0.25);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        [data-testid="stFileUploaderDropzone"]:hover {
            transform: translateY(-3px) scale(1.01);
            box-shadow: 0 0 45px rgba(0,240,255,0.4);
        }
        [data-testid="stFileUploader"] label, [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] span { color: var(--text) !important; }

        /* ---------- Buttons ---------- */
        .stButton > button, [data-testid="stFileUploader"] button {
            position: relative; overflow: hidden;
            background: linear-gradient(90deg, var(--neon-pink), var(--neon-purple), var(--neon-cyan)) !important;
            background-size: 250% 100% !important;
            animation: gradient-flow 5s linear infinite;
            color: #fff !important; border: none !important;
            border-radius: 10px; font-weight: 600; letter-spacing: 0.04em;
            padding: 0.55rem 1.2rem; width: 100%;
            box-shadow: 0 0 16px rgba(255,43,214,0.45);
            transition: transform 0.25s ease, box-shadow 0.25s ease;
        }
        .stButton > button::after {
            content: ""; position: absolute; top: 0; bottom: 0; width: 40px;
            background: rgba(255,255,255,0.35);
            animation: shimmer 2.8s ease-in-out infinite;
        }
        .stButton > button:hover {
            transform: translateY(-2px) scale(1.02);
            box-shadow: 0 0 30px rgba(0,240,255,0.7), 0 0 60px rgba(255,43,214,0.4);
            color: #fff !important;
        }
        .stButton > button:active { transform: scale(0.97); }

        /* ---------- Select box ---------- */
        div[data-baseweb="select"] > div {
            background: rgba(14,10,30,0.85) !important;
            border: 1px solid rgba(168,85,247,0.55) !important;
            border-radius: 10px !important;
            transition: box-shadow 0.3s ease, border-color 0.3s ease;
        }
        div[data-baseweb="select"] > div:hover {
            border-color: var(--neon-cyan) !important;
            box-shadow: 0 0 16px rgba(0,240,255,0.45);
        }
        div[data-baseweb="select"] * { color: var(--text) !important; }

        /* ---------- Alerts ---------- */
        [data-testid="stAlert"] {
            background: var(--panel) !important;
            border: 1px solid rgba(0,240,255,0.4) !important;
            border-radius: 12px !important;
            box-shadow: 0 0 18px rgba(0,240,255,0.18);
            animation: slide-up 0.6s ease both;
            backdrop-filter: blur(8px);
        }

        /* ---------- Chat bubbles ---------- */
        [data-testid="stChatMessage"] {
            background: var(--panel);
            border: 1px solid rgba(168,85,247,0.35);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
            backdrop-filter: blur(10px);
            animation: slide-up 0.55s cubic-bezier(.2,.8,.2,1) both;
            transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
        }
        [data-testid="stChatMessage"]:hover {
            transform: translateX(4px);
            border-color: var(--neon-cyan);
            box-shadow: 0 0 24px rgba(0,240,255,0.25);
        }
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            border-color: rgba(255,43,214,0.45);
            box-shadow: 0 0 18px rgba(255,43,214,0.15);
        }
        [data-testid="chatAvatarIcon-user"] {
            background: linear-gradient(135deg, var(--neon-pink), var(--neon-purple)) !important;
            color: #fff !important;
            box-shadow: 0 0 14px var(--neon-pink);
            animation: pulse-glow 2.4s ease-in-out infinite;
        }
        [data-testid="chatAvatarIcon-assistant"] {
            background: linear-gradient(135deg, var(--neon-cyan), var(--neon-purple)) !important;
            color: #fff !important;
            box-shadow: 0 0 14px var(--neon-cyan);
            animation: pulse-glow 2.4s ease-in-out infinite;
        }
        [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li { color: var(--text); line-height: 1.65; }
        [data-testid="stChatMessage"] code {
            font-family: 'JetBrains Mono', monospace; color: var(--neon-lime);
            background: rgba(182,255,59,0.08); border-radius: 6px;
        }

        /* ---------- Expander ---------- */
        [data-testid="stExpander"] {
            background: rgba(10,6,24,0.7);
            border: 1px solid rgba(0,240,255,0.3) !important;
            border-radius: 12px;
            transition: box-shadow 0.3s ease;
        }
        [data-testid="stExpander"]:hover { box-shadow: 0 0 18px rgba(0,240,255,0.25); }
        [data-testid="stExpander"] summary { color: var(--neon-cyan) !important; }

        /* ---------- Chat input (rotating neon ring) ---------- */
        [data-testid="stChatInput"], .stChatInputContainer > div {
            border: 2px solid transparent !important;
            border-radius: 16px !important;
            background:
                linear-gradient(#0a0618, #0a0618) padding-box,
                conic-gradient(from var(--angle), var(--neon-pink), var(--neon-purple), var(--neon-cyan), var(--neon-pink)) border-box !important;
            animation: spin-border 3.5s linear infinite;
            box-shadow: 0 0 28px rgba(168,85,247,0.35);
        }
        [data-testid="stChatInput"] textarea, .stChatInputContainer textarea {
            color: #fff !important; caret-color: var(--neon-cyan);
        }
        [data-testid="stChatInput"] button { color: var(--neon-cyan) !important; }
        [data-testid="stBottom"] > div { background: transparent !important; }

        /* Spinner + scrollbar */
        .stSpinner > div { border-top-color: var(--neon-cyan) !important; }
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(var(--neon-pink), var(--neon-cyan)); border-radius: 8px;
        }
        ::-webkit-scrollbar-track { background: #0a0618; }
        hr { border-color: rgba(168,85,247,0.35) !important; }

        /* Empty state card */
        .empty-card {
            text-align: center; margin: 1.5rem auto; max-width: 640px;
            padding: 1.6rem; border-radius: 18px;
            border: 1px solid rgba(168,85,247,0.4);
            background: var(--panel); backdrop-filter: blur(10px);
            color: var(--muted);
            animation: slide-up 0.8s ease both, bob 6s ease-in-out infinite;
        }
        .empty-card b { color: var(--neon-pink); }
        .cursor { display: inline-block; width: 9px; height: 1.1em; background: var(--neon-cyan);
                  vertical-align: text-bottom; margin-left: 4px; animation: blink 1s steps(2) infinite; }
    </style>

    <div class="orb a"></div>
    <div class="orb b"></div>
    <div class="orb c"></div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="badge"><span class="dot"></span>RAG · Groq · Live</div>
        <h1 class="title">NEON<span>DOC</span></h1>
        <p class="sub">Drop a PDF. Ask anything. Get <b>instant, grounded answers</b> from your document.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state + models
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


@st.cache_resource
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


embeddings = get_embeddings_model()
client = Groq(api_key=GROQ_API_KEY)

# ---------------------------------------------------------------------------
# File processing
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def process_pdf_to_vectorstore(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_file_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load_and_split()
        vector_store = InMemoryVectorStore(embedding=embeddings)
        vector_store.add_documents(docs)
    finally:
        os.remove(tmp_file_path)

    return vector_store, len(docs)


# ---------------------------------------------------------------------------
# Upload area
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
vector_store = None
chunk_count = 0

if uploaded_file is not None:
    with st.spinner("Indexing document into the neon grid..."):
        file_bytes = uploaded_file.getvalue()
        vector_store, chunk_count = process_pdf_to_vectorstore(file_bytes)
    st.success("Document indexed. You can now ask questions.")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ Control Panel")
    try:
        available_models = client.models.list().data
        model_options = [m.id for m in available_models if "whisper" not in m.id and "guard" not in m.id]
    except Exception:
        model_options = ["llama-3.3-70b-versatile"]

    selected_model = st.selectbox("Select Model", model_options, index=0)

    st.markdown("---")
    st.markdown("### Status")
    doc_chip = (
        f'<span class="chip">● DOC LOADED</span><span class="chip cyan">{chunk_count} CHUNKS</span>'
        if vector_store is not None
        else '<span class="chip pink">○ NO DOCUMENT</span>'
    )
    st.markdown(
        f'<div class="chip-row">{doc_chip}'
        f'<span class="chip cyan">{len(st.session_state.messages)} MSGS</span></div>',
        unsafe_allow_html=True,
    )
    if uploaded_file is not None:
        st.caption(f"📄 {uploaded_file.name}")

    st.markdown("---")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Chat UI & conversational logic
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if vector_store is not None:
    query = st.chat_input("Ask a question about the document...")

    if query:
        # 1. Show user query
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # 2. Retrieve relevant chunks
        results = vector_store.similarity_search(query, k=3)
        context_text = "\n\n".join([res.page_content for res in results])

        # 3. Build API messages
        api_messages = [
            {
                "role": "system",
                "content": "You are a highly intelligent AI assistant. Use the provided document context to answer the user's questions. If the answer is not in the context, explicitly state that you cannot find it in the document. Do not invent information.",
            }
        ]
        for m in st.session_state.messages[:-1]:
            api_messages.append(m)

        augmented_prompt = f"Context from document:\n{context_text}\n\nUser Question: {query}"
        api_messages.append({"role": "user", "content": augmented_prompt})

        # 4. Stream response
        with st.chat_message("assistant"):
            try:
                chat_completion = client.chat.completions.create(
                    messages=api_messages,
                    model=selected_model,
                    temperature=0.2,
                    max_tokens=1024,
                    stream=True,
                )

                response_placeholder = st.empty()
                full_response = ""

                for chunk in chat_completion:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        response_placeholder.markdown(
                            full_response + '<span class="cursor"></span>',
                            unsafe_allow_html=True,
                        )

                response_placeholder.markdown(full_response)

                with st.expander("⚡ View Retrieved Context"):
                    st.write(context_text)

                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception:
                st.error(f"Error communicating with model '{selected_model}'.")
                with st.expander("Show detailed error trace"):
                    st.code(traceback.format_exc())
else:
    st.markdown(
        """
        <div class="empty-card">
            <b>// AWAITING INPUT</b><br><br>
            Upload a PDF above to power up the assistant.
            <span class="cursor"></span>
        </div>
        """,
        unsafe_allow_html=True,
    )
