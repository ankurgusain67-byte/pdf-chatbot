# 📄 PDF Chatbot — RAG-Powered Document Q&A

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green?style=flat-square)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-red?style=flat-square)
![Groq](https://img.shields.io/badge/Groq-LLaMA3-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b?style=flat-square)
![Render](https://img.shields.io/badge/Deployed-Render-purple?style=flat-square)

> Upload any PDF and ask questions about it in plain English.  
> Powered by Retrieval-Augmented Generation (RAG) — LangChain + FAISS + Groq (LLaMA 3).

🔴 **Live Demo:** [pdf-chatbot on Render](https://pdf-chatbot.onrender.com)

---

## 📌 What This Does

Traditional search finds keywords. **RAG finds meaning.**

Upload a PDF → the app:
1. Splits the document into chunks
2. Converts chunks into vector embeddings
3. Stores them in a FAISS vector database
4. On your question — retrieves the most relevant chunks
5. Passes them to LLaMA 3 via Groq to generate a grounded answer

The LLM only answers from **your document** — no hallucination from general knowledge.

---

## ✨ Features

- 📤 **Upload any PDF** — reports, manuals, research papers, contracts
- 🔍 **Semantic search** — finds meaning, not just keywords
- 🧠 **Grounded answers** — responses sourced from your document only
- ⚡ **Fast inference** — Groq's LPU delivers near-instant responses
- 🖥️ **Clean UI** — built with Streamlit, no setup needed for end users
- 🚀 **Deployed** — live on Render

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq API (LLaMA 3) |
| RAG Framework | LangChain |
| Vector Database | FAISS |
| Embeddings | HuggingFace / OpenAI Embeddings |
| UI | Streamlit |
| Language | Python 3.10 |
| Deployment | Render |

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/ankurgusain67-byte/pdf-chatbot.git
cd pdf-chatbot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables
cp .env.example .env
# Add your GROQ_API_KEY inside .env

# 4. Run
streamlit run app.py
```

Then open: `http://localhost:8501`

---

## 🔑 Environment Variables
