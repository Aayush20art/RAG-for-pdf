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
    page_title="PDF Assistant",
    page_icon="📄",
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
# Professional theme with continuous, subtle motion
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        :root {
            --bg: #0f1117;
            --surface: #171a23;
            --surface-2: #1e2230;
            --border: #2a2f3f;
            --border-strong: #3a4054;
            --text: #e6e8ee;
            --muted: #8b93a7;
            --accent: #6366f1;
            --accent-hover: #818cf8;
            --accent-2: #8b5cf6;
            --accent-3: #38bdf8;
        }

        @property --angle {
            syntax: '<angle>';
            initial-value: 0deg;
            inherits: false;
        }

        /* ---------- Keyframes ---------- */
        @keyframes fade-in {
            from { opacity: 0; transform: translateY(8px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes spin-border { to { --angle: 360deg; } }
        @keyframes gradient-flow {
            0%   { background-position: 0% 50%; }
            50%  { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        @keyframes drift-a {
            0%, 100% { transform: translate(0, 0) scale(1); }
            50%      { transform: translate(7vw, 8vh) scale(1.12); }
        }
        @keyframes drift-b {
            0%, 100% { transform: translate(0, 0) scale(1); }
            50%      { transform: translate(-8vw, -6vh) scale(1.15); }
        }
        @keyframes drift-c {
            0%, 100% { transform: translate(0, 0) scale(1); }
            50%      { transform: translate(5vw, -9vh) scale(1.08); }
        }
        @keyframes grid-pan {
            from { background-position: 0 0; }
            to   { background-position: 48px 48px; }
        }
        @keyframes pulse-dot {
            0%, 100% { box-shadow: 0 0 0 0 rgba(99,102,241,0.55); }
            50%      { box-shadow: 0 0 0 7px rgba(99,102,241,0); }
        }
        @keyframes pulse-ok {
            0%, 100% { box-shadow: 0 0 0 0 rgba(74,222,128,0.5); }
            50%      { box-shadow: 0 0 0 6px rgba(74,222,128,0); }
        }
        @keyframes shimmer {
            0%   { transform: translateX(-130%) skewX(-20deg); }
            60%, 100% { transform: translateX(300%) skewX(-20deg); }
        }
        @keyframes float {
            0%, 100% { transform: translateY(0); }
            50%      { transform: translateY(-5px); }
        }
        @keyframes sweep {
            0%   { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        @keyframes blink { 50% { opacity: 0; } }
        @keyframes breathe {
            0%, 100% { border-color: var(--border); }
            50%      { border-color: rgba(99,102,241,0.55); }
        }

        /* ---------- Base ---------- */
        .stApp {
            background:
                radial-gradient(900px 500px at 8% -10%, rgba(99,102,241,0.14), transparent 60%),
                radial-gradient(800px 450px at 100% 0%, rgba(56,189,248,0.08), transparent 60%),
                var(--bg);
            color: var(--text);
            font-family: 'Inter', sans-serif;
        }
        header[data-testid="stHeader"] { background: transparent; }
        .block-container { padding-top: 2.5rem; max-width: 980px; position: relative; z-index: 2; }

        /* Slowly panning grid */
        .stApp::before {
            content: "";
            position: fixed; inset: 0;
            background-image:
                linear-gradient(rgba(139,147,167,0.06) 1px, transparent 1px),
                linear-gradient(90deg, rgba(139,147,167,0.06) 1px, transparent 1px);
            background-size: 48px 48px;
            animation: grid-pan 10s linear infinite;
            mask-image: radial-gradient(ellipse at center, #000 20%, transparent 75%);
            -webkit-mask-image: radial-gradient(ellipse at center, #000 20%, transparent 75%);
            pointer-events: none; z-index: 0;
        }

        /* Ambient drifting light */
        .orb {
            position: fixed; border-radius: 50%;
            filter: blur(110px); pointer-events: none; z-index: 0;
        }
        .orb.a { width: 34vw; height: 34vw; left: -8vw; top: 4vh;
                 background: var(--accent); opacity: 0.22; animation: drift-a 22s ease-in-out infinite; }
        .orb.b { width: 28vw; height: 28vw; right: -6vw; top: 28vh;
                 background: var(--accent-3); opacity: 0.12; animation: drift-b 26s ease-in-out infinite; }
        .orb.c { width: 26vw; height: 26vw; left: 38vw; bottom: -12vh;
                 background: var(--accent-2); opacity: 0.16; animation: drift-c 30s ease-in-out infinite; }

        h1, h2, h3, h4 {
            font-family: 'Inter', sans-serif !important;
            color: var(--text) !important;
            font-weight: 600 !important;
            letter-spacing: -0.01em;
        }

        /* ---------- Header ---------- */
        .app-header {
            position: relative; overflow: hidden;
            padding-bottom: 1.1rem; margin-bottom: 1.4rem;
            border-bottom: 1px solid var(--border);
            animation: fade-in 0.5s ease both;
        }
        .app-header::after {
            content: ""; position: absolute; left: 0; bottom: -1px;
            height: 2px; width: 40%;
            background: linear-gradient(90deg, transparent, var(--accent-hover), transparent);
            animation: sweep 3.6s ease-in-out infinite;
        }
        .eyebrow {
            display: inline-flex; align-items: center; gap: 8px;
            font-size: 0.75rem; font-weight: 600;
            letter-spacing: 0.1em; text-transform: uppercase;
            color: var(--accent-hover);
        }
        .eyebrow .dot {
            width: 7px; height: 7px; border-radius: 50%;
            background: var(--accent);
            animation: pulse-dot 2s ease-in-out infinite;
        }
        .app-header h1 {
            font-size: 2rem !important;
            margin: 0.35rem 0 0.3rem !important;
            background: linear-gradient(90deg, #e6e8ee, #a5b4fc, #7dd3fc, #e6e8ee);
            background-size: 300% 100%;
            -webkit-background-clip: text; background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradient-flow 9s ease-in-out infinite;
        }
        .app-header p { color: var(--muted); font-size: 0.97rem; margin: 0; }

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {
            background: var(--surface);
            border-right: 1px solid var(--border);
        }
        section[data-testid="stSidebar"]::after {
            content: ""; position: absolute; top: 0; right: -1px; bottom: 0; width: 1px;
            background: linear-gradient(180deg, transparent, var(--accent), transparent);
            background-size: 100% 300%;
            animation: gradient-flow 7s ease-in-out infinite;
            pointer-events: none;
        }
        section[data-testid="stSidebar"] h3 {
            font-size: 0.8rem !important;
            text-transform: uppercase; letter-spacing: 0.08em;
            color: var(--muted) !important; font-weight: 600 !important;
        }
        .status-row { display: flex; flex-wrap: wrap; gap: 6px; margin: 4px 0 10px; }
        .tag {
            display: inline-flex; align-items: center; gap: 6px;
            font-size: 0.75rem; font-weight: 500;
            padding: 4px 10px; border-radius: 6px;
            color: var(--text); background: var(--surface-2);
            border: 1px solid var(--border);
            transition: transform 0.2s ease, border-color 0.2s ease;
            animation: fade-in 0.4s ease both;
        }
        .tag:hover { transform: translateY(-1px); border-color: var(--accent); }
        .tag.ok   { color: #86efac; border-color: rgba(134,239,172,0.3); background: rgba(34,197,94,0.08); }
        .tag.ok::before {
            content: ""; width: 6px; height: 6px; border-radius: 50%;
            background: #4ade80; animation: pulse-ok 2s ease-in-out infinite;
        }
        .tag.idle { color: var(--muted); }

        /* ---------- File uploader (rotating accent border) ---------- */
        [data-testid="stFileUploaderDropzone"] {
            border: 1.5px solid transparent !important;
            border-radius: 12px !important;
            background:
                linear-gradient(var(--surface), var(--surface)) padding-box,
                conic-gradient(from var(--angle), var(--border-strong) 0%, var(--accent) 20%, var(--accent-3) 35%, var(--border-strong) 55%, var(--accent-2) 80%, var(--border-strong) 100%) border-box !important;
            animation: spin-border 6s linear infinite;
            transition: transform 0.25s ease, box-shadow 0.25s ease;
        }
        [data-testid="stFileUploaderDropzone"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(99,102,241,0.18);
        }
        [data-testid="stFileUploader"] label,
        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] span { color: var(--text) !important; }

        /* ---------- Buttons ---------- */
        .stButton > button, [data-testid="stFileUploader"] button {
            position: relative; overflow: hidden;
            background: linear-gradient(90deg, var(--accent), var(--accent-2), var(--accent)) !important;
            background-size: 220% 100% !important;
            animation: gradient-flow 8s ease-in-out infinite;
            color: #fff !important; border: none !important;
            border-radius: 8px; font-weight: 500;
            padding: 0.5rem 1.1rem;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .stButton > button { width: 100%; }
        .stButton > button::after {
            content: ""; position: absolute; top: 0; bottom: 0; width: 36px;
            background: rgba(255,255,255,0.22);
            animation: shimmer 4s ease-in-out infinite;
        }
        .stButton > button:hover, [data-testid="stFileUploader"] button:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 22px rgba(99,102,241,0.35);
            color: #fff !important;
        }
        .stButton > button:active { transform: scale(0.98); }

        /* ---------- Select box ---------- */
        div[data-baseweb="select"] > div {
            background: var(--surface-2) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }
        div[data-baseweb="select"] > div:hover {
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 3px rgba(99,102,241,0.15);
        }
        div[data-baseweb="select"] * { color: var(--text) !important; }

        /* ---------- Alerts ---------- */
        [data-testid="stAlert"] {
            background: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: 10px !important;
            animation: fade-in 0.4s ease both, breathe 4s ease-in-out infinite;
        }

        /* ---------- Chat ---------- */
        [data-testid="stChatMessage"] {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.9rem;
            animation: fade-in 0.4s cubic-bezier(.2,.8,.2,1) both;
            transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
        }
        [data-testid="stChatMessage"]:hover {
            transform: translateX(3px);
            border-color: var(--accent);
            box-shadow: 0 6px 24px rgba(99,102,241,0.12);
        }
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            background: var(--surface-2);
        }
        [data-testid="chatAvatarIcon-user"] {
            background-color: #475569 !important; color: #fff !important;
        }
        [data-testid="chatAvatarIcon-assistant"] {
            background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
            color: #fff !important;
            animation: pulse-dot 2.8s ease-in-out infinite;
        }
        [data-testid="stChatMessage"] p,
        [data-testid="stChatMessage"] li { color: var(--text); line-height: 1.65; }
        [data-testid="stChatMessage"] code {
            background: rgba(99,102,241,0.12); color: #c7d2fe; border-radius: 4px;
        }

        /* ---------- Expander ---------- */
        [data-testid="stExpander"] {
            background: var(--surface);
            border: 1px solid var(--border) !important;
            border-radius: 10px;
            transition: border-color 0.25s ease;
        }
        [data-testid="stExpander"]:hover { border-color: var(--accent) !important; }
        [data-testid="stExpander"] summary { color: var(--muted) !important; }

        /* ---------- Chat input (rotating accent ring) ---------- */
        [data-testid="stChatInput"], .stChatInputContainer > div {
            border: 1.5px solid transparent !important;
            border-radius: 12px !important;
            background:
                linear-gradient(var(--surface), var(--surface)) padding-box,
                conic-gradient(from var(--angle), var(--border-strong) 0%, var(--accent) 25%, var(--accent-3) 40%, var(--border-strong) 60%, var(--accent-2) 85%, var(--border-strong) 100%) border-box !important;
            animation: spin-border 7s linear infinite;
            transition: box-shadow 0.25s ease;
        }
        [data-testid="stChatInput"]:focus-within,
        .stChatInputContainer > div:focus-within {
            box-shadow: 0 0 0 4px rgba(99,102,241,0.15);
        }
        [data-testid="stChatInput"] textarea,
        .stChatInputContainer textarea { color: var(--text) !important; caret-color: var(--accent-hover); }
        [data-testid="stBottom"] > div { background: transparent !important; }

        /* ---------- Misc ---------- */
        hr { border-color: var(--border) !important; }
        .stSpinner > div { border-top-color: var(--accent) !important; }
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 8px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--accent); }
        ::-webkit-scrollbar-track { background: var(--bg); }

        .empty-state {
            margin: 1.5rem 0; padding: 1.6rem; text-align: center;
            color: var(--muted);
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            animation: fade-in 0.6s ease both, float 6s ease-in-out infinite, breathe 4s ease-in-out infinite;
        }
        .empty-state strong { color: var(--text); font-weight: 600; }
        .cursor {
            display: inline-block; width: 8px; height: 1.05em;
            background: var(--accent-hover); vertical-align: text-bottom;
            margin-left: 3px; animation: blink 1s steps(2) infinite;
        }

        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after { animation: none !important; transition: none !important; }
        }
    </style>

    <div class="orb a"></div>
    <div class="orb b"></div>
    <div class="orb c"></div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <div class="eyebrow"><span class="dot"></span>Document Intelligence</div>
        <h1>PDF Assistant</h1>
        <p>Upload a PDF and ask questions. Answers are grounded in the document's content.</p>
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
    with st.spinner("Processing document..."):
        file_bytes = uploaded_file.getvalue()
        vector_store, chunk_count = process_pdf_to_vectorstore(file_bytes)
    st.success("Document processed successfully. You can now ask questions.")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Settings")
    try:
        available_models = client.models.list().data
        model_options = [m.id for m in available_models if "whisper" not in m.id and "guard" not in m.id]
    except Exception:
        model_options = ["llama-3.3-70b-versatile"]

    selected_model = st.selectbox("Model", model_options, index=0)

    st.markdown("---")
    st.markdown("### Session")
    if vector_store is not None:
        doc_tags = f'<span class="tag ok">Document loaded</span><span class="tag">{chunk_count} chunks</span>'
    else:
        doc_tags = '<span class="tag idle">No document</span>'
    st.markdown(
        f'<div class="status-row">{doc_tags}'
        f'<span class="tag">{len(st.session_state.messages)} messages</span></div>',
        unsafe_allow_html=True,
    )
    if uploaded_file is not None:
        st.caption(uploaded_file.name)

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

                with st.expander("View retrieved context"):
                    st.write(context_text)

                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception:
                st.error(f"Error communicating with model '{selected_model}'.")
                with st.expander("Show detailed error trace"):
                    st.code(traceback.format_exc())
else:
    st.markdown(
        """
        <div class="empty-state">
            <strong>No document loaded</strong><br>
            Upload a PDF above to start asking questions.
        </div>
        """,
        unsafe_allow_html=True,
    )
