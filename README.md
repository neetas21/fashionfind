# 👗 FashionFind AI — Fashion Recommendation Assistant

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![LLaMA](https://img.shields.io/badge/LLaMA_3.3-70B-8A2BE2?style=for-the-badge)](https://groq.com)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-00BFFF?style=for-the-badge)](https://faiss.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

> A production-style **RAG (Retrieval-Augmented Generation)** pipeline that acts as an AI-powered fashion assistant. Users describe what they're looking for and get intelligent, personalized product recommendations in natural language.

---

## 🎯 What It Does

Ask it anything:
- *"I need something for a wedding under ₹3000"*
- *"Show me casual outfits in pink"*
- *"What should I wear in winter?"*

And it responds like a real stylist — not a boring list of links.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
Sentence Transformers (all-MiniLM-L6-v2)
    │  converts text → 384-dimensional embedding
    ▼
FAISS Vector Index
    │  cosine similarity search across product catalog
    ▼
Top-K Matching Products (with budget filter)
    │
    ▼
Groq LLaMA 3.3 (70B)
    │  generates natural language recommendation
    ▼
Friendly Response to User 🎉
```

---

## 🧰 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) | Convert text → 384-dim vectors |
| Vector Search | `FAISS` (IndexFlatL2) | Fast similarity search |
| LLM | Groq `llama-3.3-70b-versatile` | Natural language response generation |
| Data Pipeline | `Pandas` | Product catalog management |
| UI | `Streamlit` | Interactive chat interface |

---

## 📁 Project Structure

```
fashionfind/
├── app.py                  # Streamlit chat UI + RAG pipeline
├── FashionFind.ipynb       # Original research notebook
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── .gitignore              # Ignores .env and cache files
└── README.md               # You are here!
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/fashionfind-ai.git
cd fashionfind-ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up your API key
```bash
cp .env.example .env
# Open .env and add your Groq API key
# Get a free key at: https://console.groq.com
```

### 4. Run the app
```bash
https://fashionfind-ai.streamlit.app
```

Open your browser at `http://localhost:8501` and start chatting! 🎉

---

## 💡 How RAG Works Here

**RAG = Retrieval Augmented Generation**

Instead of asking an LLM to "know" all fashion products (which it can't), we:

1. **Store** all product descriptions as vector embeddings in FAISS
2. **Retrieve** the most similar products when a user asks a question
3. **Augment** the LLM prompt with those real products
4. **Generate** a natural, accurate response grounded in actual catalog data

This means the LLM never hallucinates products — it only recommends what actually exists.

---

## 🗺️ Roadmap

- [x] Core RAG pipeline (embeddings + FAISS + LLM)
- [x] Budget filtering
- [x] Streamlit chat UI
- [ ] Expand product catalog (100+ items)
- [ ] Add Pinecone for production vector storage
- [ ] Add image-based search (upload a photo, find similar)
- [ ] Apache Airflow pipeline for catalog auto-updates
- [ ] Deploy on AWS / GCP with CI/CD

---

## 🤝 Data Engineering Relevance

This project demonstrates key **Data Engineering** skills:

| DE Skill | Where Used |
|---|---|
| ETL Pipeline | Loading, cleaning, transforming product catalog with Pandas |
| Feature Engineering | Converting raw text descriptions → numeric embeddings |
| Vector Database | Indexing and querying FAISS (similar to Pinecone/Weaviate) |
| API Integration | Groq LLM API with proper error handling |
| Pipeline Architecture | End-to-end data flow design |

---

## 👩‍💻 Built By

**Neeta Singh**
- 🔗 [LinkedIn](https://linkedin.com/in/neetasingh239/)
- 🐙 [GitHub](https://github.com/neetas21)

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
