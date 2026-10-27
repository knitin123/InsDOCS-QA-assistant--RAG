# 📅 AI Medical Assistant Chatbot — RAG-based Application

[![CI](https://github.com/knitin123/InsDocs-QA-assistant--RAG/actions/workflows/ci.yml/badge.svg)](https://github.com/knitin123/InsDocs-QA-assistant--RAG/actions/workflows/ci.yml)

## 🧠 Project Overview

This application is a **Medical Domain Chatbot** built using **Retrieval-Augmented Generation (RAG)**. It allows users to upload their own medical documents (e.g., textbooks, reports), and the system intelligently answers queries by retrieving the most relevant content before generating a final response.

---

## 🎓 What is RAG?

**RAG (Retrieval-Augmented Generation)** enhances language models by supplying relevant external context from a knowledge base, preventing hallucinations and improving accuracy, especially for factual or specialized domains like **medicine**.

---

## 🔄 Architecture

```
User Input
   ↓
Query Embedding → Pinecone Vector DB ← Embedded Chunks ← Chunking ← PDF Loader
   ↓
Retrieved Docs
   ↓
     RAG Chain (Groq + LangChain)
   ↓
LLM-generated Answer
```

For a detailed view, refer to the **[MedicalAssistant.pdf](./assets/MedicalAssistant.pdf)**

---

## 📚 Features

- Upload medical PDFs (notes, books, etc.)
- Auto-extracts text and splits into semantic chunks
- Embeds using Google `gemini-embedding-001` (768 dims)
- Stores vectors in **Pinecone DB**
- Uses **Groq's Llama 3.3 70B** via LangChain
- FastAPI backend with endpoints for file upload and Q\&A

---

## 🌐 Tech Stack

| Component  | Tech Used                  |
| ---------- | -------------------------- |
| LLM        | Groq API (Llama 3.3 70B)   |
| Embeddings | Google Generative AI       |
| Vector DB  | Pinecone                   |
| Framework  | LangChain                  |
| Backend    | FastAPI                    |
| Deployment | Render                     |

---

## 📚 API Endpoints

```http
POST /upload_pdfs/ --- Upload one or more PDF files

POST /ask/ --- Ask a question --- Form field: `question`

GET /health --- Health check

```

---

## 📁 Folder Structure

```
└── 📁assets
    ├── DIABETES.pdf
    ├── MedicalAssistant.pdf
    └── medicalAssistant.png
```

```
└── 📁client
    └── 📁__pycache__
        ├── config.cpython-311.pyc
    └── 📁components
        └── 📁__pycache__
            ├── chatUI.cpython-311.pyc
            ├── history_download.cpython-311.pyc
            ├── upload.cpython-311.pyc
        ├── chatUI.py
        ├── history_download.py
        ├── upload.py
    └── 📁utils
        └── 📁__pycache__
            ├── api.cpython-311.pyc
        ├── api.py
    ├── app.py
    ├── config.py
    └── requirements.txt
```

```
└── 📁server
    └── 📁__pycache__
        ├── logger.cpython-311.pyc
        ├── main.cpython-311.pyc
        ├── test.cpython-311.pyc
    └── 📁middlewares
        └── 📁__pycache__
            ├── exception_handlers.cpython-311.pyc
        ├── exception_handlers.py
    └── 📁modules
        └── 📁__pycache__
            ├── llm.cpython-311.pyc
            ├── load_vectorstore.cpython-311.pyc
            ├── query_handlers.cpython-311.pyc
        ├── llm.py
        ├── load_vectorstore.py
        ├── pdf_handlers.py
        ├── query_handlers.py
    └── 📁routes
        └── 📁__pycache__
            ├── ask_question.cpython-311.pyc
            ├── upload_pdfs.cpython-311.pyc
        ├── ask_question.py
        ├── upload_pdfs.py
    └── 📁uploaded_docs
        ├── DIABETES.pdf
        ├── Supratim Nag - LOR.pdf
    ├── .env
    ├── logger.py
    ├── main.py
    ├── requirements.txt
    └── test.py
```

---

## ⚡ Quick Setup

```bash
# Clone the repo
$ git clone https://github.com/knitin123/InsDocs-QA-assistant--RAG/Medical-Assistant---RAG-Application.git
$ cd Medical-Assistant---RAG-Application/server

# Create virtual env
$ uv venv
$ .venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
$ uv pip install -r requirements.txt

# Set environment variables (.env)
GOOGLE_API_KEY=...
GROQ_API_KEY=...
PINECONE_API_KEY=...

# Run the server
$ uvicorn main:app --reload --port 8000


$ cd Medical-Assistant---RAG-Application/client

# Create virtual env
$ uv venv
$ .venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
$ uv pip install -r requirements.txt

# Run the server
$ streamlit run app.py
```

Optional environment variables: `PINECONE_INDEX_NAME` (default `insightdocs`), `GROQ_MODEL` (default `llama-3.3-70b-versatile`), `EMBED_MODEL` (default `models/gemini-embedding-001`).

---

## 🧪 Testing & CI

The pytest suite runs the full pipeline (PDF chunking → embedding → Pinecone upsert → retrieval → LLM answer → FastAPI endpoints) against in-memory fakes, so it needs no API keys:

```bash
$ cd server
$ pip install -r requirements-dev.txt
$ pytest -v
```

A live smoke test uploads `assets/DIABETES.pdf` to real Pinecone and asks Groq about it:

```bash
$ pytest -m integration   # needs GOOGLE_API_KEY, PINECONE_API_KEY, GROQ_API_KEY
```

GitHub Actions ([.github/workflows/ci.yml](../.github/workflows/ci.yml)) runs the offline suite on every push and pull request, then the live smoke test on pushes to `main` using the repository secrets `GOOGLE_API_KEY`, `PINECONE_API_KEY` and `GROQ_API_KEY` (it skips if they aren't set).

---

## 🌐 Deployment

- Hosted on [Render](https://render.com)
- Configure `start command` as:

  ```bash
  uvicorn main:app --host 0.0.0.0 --port 10000
  ```

---

## 🎉 License

This project is licensed under the MIT License.
