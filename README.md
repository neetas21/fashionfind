# FashionFind — AI Fashion Assistant
A RAG pipeline for fashion product recommendations using Sentence Transformers, FAISS, and LLaMA 3.3.

## Tech Stack
- Sentence Transformers (embeddings)
- FAISS (vector search)
- Groq LLaMA 3.3 (LLM)
- Pandas (data pipeline)

## How it works
1. Product descriptions → 384-dimensional embeddings
2. User query → FAISS similarity search
3. Top matches → LLM → natural response

Built by Neeta Singh
