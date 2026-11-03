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
    page_title="Neon PDF Assistant",
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
# Cyberpunk / Neon Theme CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

        /* Main App Background */
        .stApp {
            background-color: #06070d;
            background-image: radial-gradient(circle at 50% 0%, #150a21, #06070d 80%);
            color: #e0e0e0;
            font-family: 'Inter', sans-serif;
        }

        /* Neon Headers */
        h1, h2, h3 { 
            font-family: 'Orbitron', sans-serif !important; 
            color: #00ffff !important; 
            text-shadow: 0 0 5px #00ffff, 0 0 15px rgba(0, 255, 255, 0.6) !important;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background: rgba(10, 5, 20, 0.7);
            border-right: 1px solid #b900ff;
            box-shadow: 2px 0 15px rgba(185, 0, 255, 0.2);
            backdrop-filter: blur(10px);
        }

        /* File Uploader Pulse Animation */
        [data-testid="stFileUploader"] {
            background: rgba(20, 10, 30, 0.6);
            border: 2px dashed #b900ff;
            border-radius: 12px;
            padding: 1.5rem;
            animation: neon-pulse 2s infinite alternate;
        }
        @keyframes neon-pulse {
            0% { box-shadow: 0 0 5px rgba(185, 0, 255, 0.2), inset 0 0 5px rgba(185, 0, 255, 0.2); }
            100% { box-shadow: 0 0 15px rgba(185, 0, 255, 0.8), inset 0 0 10px rgba(185, 0, 255, 0.5); }
        }

        /* Chat Bubbles (Glassmorphism) */
        [data-testid="stChatMessage"] {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(0, 255, 255, 0.2);
            border-radius: 15px;
            padding: 1rem;
            backdrop-filter: blur(8px);
            margin-bottom: 1rem;
            box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        }
        
        /* User/Assistant Avatars */
        [data-testid="chatAvatarIcon-user"] { background-color: #b900ff !important; color: white !important; }
        [data-testid="chatAvatarIcon-assistant"] { background-color: #00ffff !important; color: black !important; }

        /* Neon Buttons */
        .stButton > button {
            background: transparent !important;
            color: #00ffff !important;
            border: 1px solid #00ffff !important;
            border-radius: 8px;
            font-weight: 600;
            padding: 0.5rem 1.2rem;
            transition: all 0.3s ease;
            box-shadow: 0 0 8px rgba(0, 255, 255, 0.3) inset;
            width: 100%;
        }
        .stButton > button:hover {
            background: #00ffff !important;
            color: #000 !important;
            box-shadow: 0 0 15px #00ffff, inset 0 0 10px #00ffff !important;
            transform: translateY(-2px);
        }

        /* Chat Input Field */
        .stChatInputContainer > div {
            background: #0d0e15 !important;
            border: 1px solid #b900ff !important;
            box-shadow: 0 0 10px rgba(185, 0, 255, 0.3) !important;
            border-radius: 10px !important;
        }
        .stChatInputContainer textarea {
            color: #fff !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------------
# App Logic & Initialization
# ---------------------------------------------------------------------------
st.title("NEON_PDF_ASSISTANT ⚡")
st.markdown("*Upload data protocols. Initialize semantic query interface.*")

# Initialize Chat History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

embeddings = get_embeddings_model()
client = Groq(api_key=GROQ_API_KEY)

# ---------------------------------------------------------------------------
# Sidebar Settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### SYSTEM SETTINGS ⚙️")
    try:
        available_models = client.models.list().data
        model_options = [m.id for m in available_models if "whisper" not in m.id and "guard" not in m.id]
    except Exception:
        model_options = ["llama-3.3-70b-versatile"]
    
    selected_model = st.selectbox("Select Neural Core (Model)", model_options, index=0)
    
    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# File Processing
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def process_pdf_to_vectorstore(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    docs = loader.load_and_split()
    
    vector_store = InMemoryVectorStore(embedding=embeddings)
    vector_store.add_documents(docs)
    
    os.remove(tmp_file_path)
    return vector_store

uploaded_file = st.file_uploader("UPLOAD DATABANK (PDF)", type=["pdf"])
vector_store = None

if uploaded_file is not None:
    with st.spinner("Extracting databank vectors..."):
        file_bytes = uploaded_file.getvalue()
        vector_store = process_pdf_to_vectorstore(file_bytes)
    st.success("DATABANK LINK ESTABLISHED.")

# ---------------------------------------------------------------------------
# Chat UI & Conversational Logic
# ---------------------------------------------------------------------------
# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Only show the chat input if a PDF is loaded
if vector_store is not None:
    query = st.chat_input("Enter query command...")
    
    if query:
        # 1. Display user query immediately
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
            
        # 2. Retrieve relevant PDF chunks based on the current query
        results = vector_store.similarity_search(query, k=3)
        context_text = "\n\n".join([res.page_content for res in results])
        
        # 3. Build API Message List (System Prompt + History + Augmented Latest Prompt)
        api_messages = [
            {
                "role": "system",
                "content": "You are a highly intelligent AI assistant. Use the provided document context to answer the user's questions. If the answer is not in the context, explicitly state that you cannot find it in the document. Do not invent information."
            }
        ]
        
        # Add historical conversation context (excluding the very last prompt we just added)
        for m in st.session_state.messages[:-1]:
            api_messages.append(m)
            
        # Add the latest prompt injected with the fresh PDF context
        augmented_prompt = f"Context from document:\n{context_text}\n\nUser Question: {query}"
        api_messages.append({"role": "user", "content": augmented_prompt})
        
        # 4. Stream response from Groq
        with st.chat_message("assistant"):
            try:
                chat_completion = client.chat.completions.create(
                    messages=api_messages,
                    model=selected_model, 
                    temperature=0.2,
                    max_tokens=1024,
                    stream=True  # Enables typewriter streaming
                )
                
                # Create an empty placeholder to stream text into
                response_placeholder = st.empty()
                full_response = ""
                
                for chunk in chat_completion:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        # Add a blinking cursor effect while streaming
                        response_placeholder.markdown(full_response + " █")
                
                # Final output without the cursor
                response_placeholder.markdown(full_response)
                
                with st.expander("🔍 View Retrieved Context"):
                    st.write(context_text)
                    
                # Save the final response to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                st.error(f"SYSTEM ERROR: Connection to neural core '{selected_model}' failed.")
                with st.expander("Show detailed error trace"):
                    st.code(traceback.format_exc())
else:
    st.info("Awaiting PDF upload to initialize chat interface...")
