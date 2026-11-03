import os
import tempfile

import streamlit as st
import groq
from groq import Groq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PDF Assistant",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Groq API key: add GROQ_API_KEY to Streamlit Cloud > App > Settings > Secrets
# ---------------------------------------------------------------------------
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error(
        "GROQ_API_KEY was not found. Add it under Streamlit Cloud → "
        "Manage app → Settings → Secrets."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        .stApp {
            background: #0a0a0a;
            color: #f0f0f0;
        }
        section[data-testid="stSidebar"] {
            background: #111111;
            border-right: 1px solid #2a0000;
        }
        section[data-testid="stSidebar"] .stMarkdown h1,
        section[data-testid="stSidebar"] .stMarkdown h2,
        section[data-testid="stSidebar"] .stMarkdown h3 {
            color: #ff2222;
        }
        h1, h2, h3 { color: #ff2222 !important; }
        h1 { font-size: 2rem !important; font-weight: 700 !important; }
        h2 { font-size: 1.4rem !important; }
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
        [data-testid="stFileUploader"] {
            background: #1a1a1a;
            border: 2px dashed #3a0000;
            border-radius: 10px;
            padding: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------------
# App Logic
# ---------------------------------------------------------------------------
st.title("PDF Semantic Search 🔴")
st.write("Upload a PDF and ask questions about its contents.")

# Cache the embeddings model so it doesn't reload on every interaction
@st.cache_resource
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

embeddings = get_embeddings_model()

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# File Uploader
uploaded_file = st.file_uploader("Upload your PDF document", type=["pdf"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location so PyPDFLoader can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    with st.spinner("Processing PDF and generating embeddings..."):
        # Load and split the document
        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load_and_split()
        
        # Create vector store and add documents
        vector_store = InMemoryVectorStore(embedding=embeddings)
        vector_store.add_documents(docs)
        
        st.success("PDF processed successfully!")

    # Search Interface
    st.subheader("Ask a Question")
    query = st.text_input("What would you like to know about this document?")
    
    if query:
        # 1. Retrieve the most relevant text chunks from the PDF
        results = vector_store.similarity_search(query, k=2)
        
        # 2. Combine the retrieved chunks into a single context string
        context_text = "\n\n".join([res.page_content for res in results])
        
        # 3. Create a prompt instructing the LLM to answer using only the context
        prompt = f"""
        Answer the following question based ONLY on the provided context from the user's document.
        If the answer is not contained in the context, say "I cannot answer this based on the uploaded document."
        
        Context:
        {context_text}
        
        Question: {query}
        """
        
        with st.spinner("Thinking..."):
            # 4. Send the prompt to Groq to generate the final answer
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="llama3-8b-8192", # Using Llama 3 8B model via Groq
                temperature=0.2,
            )
            
            # Extract and display the answer
            answer = chat_completion.choices[0].message.content
            
            st.markdown("### Answer")
            st.write(answer)
            
            # Optional: Keep the retrieved text hidden in an expander for debugging
            with st.expander("View Retrieved Context"):
                st.write(context_text)

    # Clean up the temporary file
    os.remove(tmp_file_path)
