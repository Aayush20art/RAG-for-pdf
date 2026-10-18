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
