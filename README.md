---
title: Mini RAG Sprint Backend
emoji: 🏢
colorFrom: red
colorTo: indigo
sdk: docker
app_port: 7860
---

# Mini RAG Sprint

This project is a backend API for a Retrieval-Augmented Generation (RAG) system, built as part of the Mini RAG Sprint.

The API provides endpoints to perform vector search and hybrid search on a pre-indexed document set, and to return a generated answer based on the most relevant context. The API is designed to be consumed by other applications for evaluation and testing.

### Technical Details

* **Framework**: FastAPI
* **Vector Database**: FAISS
* **Text Ingestion**: Whoosh
* **Embedding Model**: `all-MiniLM-L6-v2`

### Evaluation Results

Please fill in the table below with the results from your evaluation script (`eval.py`).

| Metric | Result |
| :--- | :--- |
| **Pass Rate** | [Your Pass Rate Here] |
| **Faithfulness Score** | [Your Faithfulness Score Here] |