🔴 RAG PDF Assistant
<div align="center">
Show Image
Show Image
Show Image
Show Image
Show Image
A Retrieval-Augmented Generation (RAG) app that lets you chat with any PDF using Groq LLMs and HuggingFace Embeddings.
Show Image
 
Show Image
</div>

📌 Table of Contents

Overview
Features
Tech Stack
How It Works
Getting Started

Prerequisites
Installation
Running Locally


Deploying on Streamlit Cloud
Project Structure
Screenshots
Author


🧠 Overview
RAG PDF Assistant is a Streamlit-based web application that implements a full Retrieval-Augmented Generation (RAG) pipeline. Upload any PDF document, and the app will embed its content, retrieve the most relevant chunks for your question, and generate accurate answers using a Groq-powered LLM — all in a sleek red and black UI.

✨ Features

📄 Upload any PDF — auto-chunked and embedded on the fly
🔍 Semantic Search — retrieves the most relevant chunks using cosine similarity
🤖 Groq LLM Integration — fast inference with models like LLaMA 3.3 70B, Mixtral, Gemma2
💬 Multi-turn Conversation — chat history passed to LLM for contextual follow-ups
📎 Source Transparency — expandable source chunks with page numbers shown per answer
⚙️ Configurable Settings — choose model and Top-K retrieval chunks from sidebar
🔐 Secure API Handling — Groq API key loaded from Streamlit Secrets (never exposed in UI)
🎨 Red & Black Theme — premium dark UI with glassmorphism card styling


🛠 Tech Stack
LayerTechnologyFrontendStreamlitEmbeddingsHuggingFace sentence-transformers/all-mpnet-base-v2Vector StoreLangChain InMemoryVectorStorePDF LoaderLangChain PyPDFLoaderLLMGroq API (LLaMA 3.3 70B / Mixtral / Gemma2)OrchestrationLangChain CoreDeploymentStreamlit Cloud

⚙️ How It Works
  User uploads PDF
        │
        ▼
  PyPDFLoader splits PDF into chunks
        │
        ▼
  HuggingFace Embeddings encode each chunk
        │
        ▼
  Chunks stored in InMemoryVectorStore
        │
        ▼
  User asks a question
        │
        ▼
  Similarity Search retrieves Top-K relevant chunks
        │
        ▼
  Groq LLM generates answer using retrieved context
        │
        ▼
  Answer + Source Chunks displayed in chat UI

🚀 Getting Started
Prerequisites

Python 3.10+
A free Groq API Key → console.groq.com


Installation
1. Clone the repository
bashgit clone https://github.com/Aayush20art/RAG-for-pdf.git
cd RAG-for-pdf
2. Create a virtual environment (recommended)
bashpython -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
3. Install dependencies
bashpip install -r requirements.txt

Running Locally
1. Create the secrets file
bashmkdir -p .streamlit
Create .streamlit/secrets.toml and add your Groq API key:
tomlGROQ_API_KEY = "gsk_your_key_here"

⚠️ Important: Add .streamlit/secrets.toml to your .gitignore — never commit your API key.

2. Run the app
bashstreamlit run app.py
3. Open in browser
http://localhost:8501

☁️ Deploying on Streamlit Cloud

Push your code to GitHub (make sure app.py and requirements.txt are in the root)
Go to share.streamlit.io → New App
Connect your GitHub repository
Go to App Settings → Secrets and paste:

tomlGROQ_API_KEY = "gsk_your_key_here"

Click Deploy — done! 🎉


Note: First deployment downloads the HuggingFace embedding model (~420MB). Streamlit Cloud caches it via @st.cache_resource so subsequent runs are fast.


📁 Project Structure
RAG-for-pdf/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── secrets.toml        # Local secrets (DO NOT commit)
├── embeddings.ipynb        # Original RAG prototype notebook
└── README.md               # Project documentation

📦 Requirements
streamlit>=1.35.0
langchain>=0.2.0
langchain-huggingface>=0.0.3
langchain-community>=0.2.0
sentence-transformers>=2.7.0
pypdf>=4.2.0
groq>=0.9.0
