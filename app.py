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
# Professional theme
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
        }

        @keyframes fade-in {
            from { opacity: 0; transform: translateY(6px); }
            to   { opacity: 1; transform: translateY(0); }
        }

        /* ---------- Base ---------- */
        .stApp {
            background-color: var(--bg);
            color: var(--text);
            font-family: 'Inter', sans-serif;
        }
        header[data-testid="stHeader"] { background: transparent; }
        .block-container { padding-top: 2.5rem; max-width: 980px; }

        h1, h2, h3, h4 {
            font-family: 'Inter', sans-serif !important;
            color: var(--text) !important;
            font-weight: 600 !important;
            letter-spacing: -0.01em;
        }

        /* ---------- Header ---------- */
        .app-header {
            padding-bottom: 1.1rem;
            margin-bottom: 1.4rem;
            border-bottom: 1px solid var(--border);
            animation: fade-in 0.4s ease both;
        }
        .app-header .eyebrow {
            font-size: 0.75rem; font-weight: 600;
            letter-spacing: 0.1em; text-transform: uppercase;
            color: var(--accent-hover);
        }
        .app-header h1 {
            font-size: 1.9rem !important;
            margin: 0.25rem 0 0.3rem !important;
        }
        .app-header p { color: var(--muted); font-size: 0.97rem; margin: 0; }

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {
            background: var(--surface);
            border-right: 1px solid var(--border);
        }
        section[data-testid="stSidebar"] h3 {
            font-size: 0.8rem !important;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--muted) !important;
            font-weight: 600 !important;
        }
        .status-row { display: flex; flex-wrap: wrap; gap: 6px; margin: 4px 0 10px; }
        .tag {
            font-size: 0.75rem; font-weight: 500;
            padding: 4px 10px; border-radius: 6px;
            color: var(--text);
            background: var(--surface-2);
            border: 1px solid var(--border);
        }
        .tag.ok   { color: #86efac; border-color: rgba(134,239,172,0.3); background: rgba(34,197,94,0.08); }
        .tag.idle { color: var(--muted); }

        /* ---------- File uploader ---------- */
        [data-testid="stFileUploaderDropzone"] {
            background: var(--surface);
            border: 1px dashed var(--border-strong);
            border-radius: 10px;
            transition: border-color 0.2s ease, background 0.2s ease;
        }
        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: var(--accent);
            background: var(--surface-2);
        }
        [data-testid="stFileUploader"] label,
        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] span { color: var(--text) !important; }

        /* ---------- Buttons ---------- */
        .stButton > button, [data-testid="stFileUploader"] button {
            background: var(--accent) !important;
            color: #fff !important;
            border: none !important;
            border-radius: 8px;
            font-weight: 500;
            padding: 0.5rem 1.1rem;
            transition: background 0.2s ease, transform 0.1s ease;
        }
        .stButton > button { width: 100%; }
        .stButton > button:hover, [data-testid="stFileUploader"] button:hover {
            background: var(--accent-hover) !important;
            color: #fff !important;
        }
        .stButton > button:active { transform: scale(0.98); }

        /* ---------- Select box ---------- */
        div[data-baseweb="select"] > div {
            background: var(--surface-2) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
            transition: border-color 0.2s ease;
        }
        div[data-baseweb="select"] > div:hover { border-color: var(--accent) !important; }
        div[data-baseweb="select"] * { color: var(--text) !important; }

        /* ---------- Alerts ---------- */
        [data-testid="stAlert"] {
            background: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: 10px !important;
        }

        /* ---------- Chat ---------- */
        [data-testid="stChatMessage"] {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.9rem;
            animation: fade-in 0.3s ease both;
        }
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            background: var(--surface-2);
        }
        [data-testid="chatAvatarIcon-user"] {
            background-color: #475569 !important; color: #fff !important;
        }
        [data-testid="chatAvatarIcon-assistant"] {
            background-color: var(--accent) !important; color: #fff !important;
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
        }
        [data-testid="stExpander"] summary { color: var(--muted) !important; }

        /* ---------- Chat input ---------- */
        [data-testid="stChatInput"], .stChatInputContainer > div {
            background: var(--surface) !important;
            border: 1px solid var(--border-strong) !important;
            border-radius: 12px !important;
            transition: border-color 0.2s ease;
        }
        [data-testid="stChatInput"]:focus-within,
        .stChatInputContainer > div:focus-within { border-color: var(--accent) !important; }
        [data-testid="stChatInput"] textarea,
        .stChatInputContainer textarea { color: var(--text) !important; }
        [data-testid="stBottom"] > div { background: transparent !important; }

        /* ---------- Misc ---------- */
        hr { border-color: var(--border) !important; }
        .stSpinner > div { border-top-color: var(--accent) !important; }
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 8px; }
        ::-webkit-scrollbar-track { background: var(--bg); }

        .empty-state {
            margin: 1.5rem 0;
            padding: 1.5rem;
            text-align: center;
            color: var(--muted);
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
        }
        .empty-state strong { color: var(--text); font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <div class="eyebrow">Document Intelligence</div>
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
                        response_placeholder.markdown(full_response + " ▌")

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
