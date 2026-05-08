<div align="center">

# 🔴 RAG PDF Assistant

![Python](https://img.shields.io/badge/Python-3.10+-red?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-LLM-black?style=for-the-badge&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-red?style=for-the-badge)

**A Retrieval-Augmented Generation (RAG) app that lets you chat with any PDF**  
**using Groq LLMs + HuggingFace Embeddings — deployed on Streamlit Cloud.**

<br/>

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Click%20Here-cc0000?style=for-the-badge)](https://rag-for-pdf-tgf6tjgm4k8bdiqdcbdgnz.streamlit.app/)
&nbsp;&nbsp;
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](https://github.com/Aayush20art/RAG-for-pdf)

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [How It Works](#️-how-it-works)
- [Getting Started](#-getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Running Locally](#running-locally)
- [Deploying on Streamlit Cloud](#️-deploying-on-streamlit-cloud)
- [Project Structure](#-project-structure)
- [Requirements](#-requirements)
- [Author](#-author)

---

## 🧠 Overview

**RAG PDF Assistant** is a Streamlit web app implementing a complete **Retrieval-Augmented Generation (RAG)** pipeline.

> Upload any PDF → Chunks are embedded → Your question retrieves relevant context → Groq LLM answers accurately.

All in a fast, clean **red & black** dark UI with no API key exposed to end users.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 PDF Upload | Auto-chunked and embedded on the fly |
| 🔍 Semantic Search | Retrieves most relevant chunks via cosine similarity |
| 🤖 Groq LLM | Fast inference — LLaMA 3.3 70B, Mixtral, Gemma2 |
| 💬 Multi-turn Chat | Full conversation history passed to the LLM |
| 📎 Source Transparency | Source chunks + page numbers shown per answer |
| ⚙️ Configurable | Choose model & Top-K chunks from sidebar |
| 🔐 Secure API | Groq key loaded from Streamlit Secrets — never in UI |
| 🎨 Custom Theme | Premium red & black dark UI |

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit |
| **Embeddings** | HuggingFace — `sentence-transformers/all-mpnet-base-v2` |
| **Vector Store** | LangChain `InMemoryVectorStore` |
| **PDF Loader** | LangChain `PyPDFLoader` |
| **LLM Provider** | Groq API |
| **LLM Models** | LLaMA 3.3 70B · Mixtral 8x7B · Gemma2 9B |
| **Orchestration** | LangChain Core |
| **Deployment** | Streamlit Cloud |

---

## ⚙️ How It Works

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   1.  User uploads PDF                                  │
│            │                                            │
│            ▼                                            │
│   2.  PyPDFLoader  →  splits into chunks                │
│            │                                            │
│            ▼                                            │
│   3.  HuggingFace Embeddings  →  encode each chunk      │
│            │                                            │
│            ▼                                            │
│   4.  InMemoryVectorStore  →  store vectors             │
│            │                                            │
│            ▼                                            │
│   5.  User asks a question                              │
│            │                                            │
│            ▼                                            │
│   6.  Similarity Search  →  retrieve Top-K chunks       │
│            │                                            │
│            ▼                                            │
│   7.  Groq LLM  →  generate answer from context         │
│            │                                            │
│            ▼                                            │
│   8.  Answer + Source Chunks displayed in chat UI       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

- Python **3.10+**
- A free **Groq API Key** — get one at [console.groq.com](https://console.groq.com)

---

### Installation

**Step 1 — Clone the repository**

```bash
git clone https://github.com/Aayush20art/RAG-for-pdf.git
cd RAG-for-pdf
```

**Step 2 — Create a virtual environment** *(recommended)*

```bash
# Create venv
python -m venv venv

# Activate — Linux / macOS
source venv/bin/activate

# Activate — Windows
venv\Scripts\activate
```

**Step 3 — Install dependencies**

```bash
pip install -r requirements.txt
```

---

### Running Locally

**Step 1 — Create the secrets file**

```bash
mkdir -p .streamlit
```

Create `.streamlit/secrets.toml` and paste:

```toml
GROQ_API_KEY = "gsk_your_key_here"
```

> ⚠️ **Never commit this file.** Add `.streamlit/secrets.toml` to your `.gitignore`.

**Step 2 — Run the app**

```bash
streamlit run app.py
```

**Step 3 — Open in browser**

```
http://localhost:8501
```

---

## ☁️ Deploying on Streamlit Cloud

**Step 1** — Push `app.py` + `requirements.txt` to your GitHub repo

**Step 2** — Go to [share.streamlit.io](https://share.streamlit.io) → click **New App**

**Step 3** — Connect your GitHub repository and select `app.py` as the entry point

**Step 4** — Go to **App Settings → Secrets** and paste:

```toml
GROQ_API_KEY = "gsk_your_key_here"
```

**Step 5** — Click **Deploy** 🎉

> 💡 **Note:** First boot downloads the HuggingFace embedding model (~420 MB).  
> Streamlit Cloud caches it via `@st.cache_resource` — all subsequent runs are fast.

---

## 📁 Project Structure

```
RAG-for-pdf/
│
├── app.py                  ← Main Streamlit application
├── requirements.txt        ← Python dependencies
├── embeddings.ipynb        ← Original RAG prototype notebook
├── README.md               ← Project documentation
│
└── .streamlit/
    └── secrets.toml        ← Local secrets (DO NOT commit)
```

---

## 📦 Requirements

```text
streamlit>=1.35.0
langchain>=0.2.0
langchain-huggingface>=0.0.3
langchain-community>=0.2.0
sentence-transformers>=2.7.0
pypdf>=4.2.0
groq>=0.9.0
```

⭐ **If this project helped you, please consider giving it a star!** ⭐

</div>
