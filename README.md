---
title: Mini RAG + Reranker Technical Assessment
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
---

# Mini RAG + Reranker Sprint

This repository contains a simple, extractive Question-Answering (Q&A) service built over a small set of industrial and machine safety documents. The service is built on a two-pass architecture: a fast vector retrieval baseline followed by a smart reranker to improve the quality of the evidence used to generate the answer.

## 📦 Project Structure

| File/Folder | Purpose |
| :--- | :--- |
| `Dockerfile` | Defines the environment and installs dependencies, including data download. |
| `app.py` | The FastAPI service with the `/ask` endpoint, index loading, and Q&A logic. |
| `requirements.txt` | Lists all necessary Python dependencies (e.g., `fastapi`, `uvicorn`, `faiss-cpu`, `sentence-transformers`, `requests`). |
| `ingest.py` | Script used to chunk documents, generate embeddings, and create `faiss_index.bin` and `chunks.pkl`. (Not needed at runtime). |
| `sources.json` | JSON file listing the title and URL for every document processed. |
| `8_questions.txt` | The file containing the eight test questions and their expected answers. |
| `data/` | **[DELETED LOCALLY]** Used locally for initial processing; files are hosted externally for deployment. |

## 🚀 Setup and Running

### Prerequisites

* Python 3.8+
* `pip`
* `git`
* Docker (if running locally, not required for Hugging Face deployment)

### 1. Local Setup

First, install the necessary libraries:
```bash
pip install -r requirements.txt
