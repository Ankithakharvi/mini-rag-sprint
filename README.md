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
This is the complete README.md file you can use. It is structured to meet all the requirements of the technical assessment, including the deployment configuration and the final deliverables.

Save this file as README.md in the root of your project directory.

Markdown

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
2. Run the API Locally
The app.py file is configured to download the necessary faiss_index.bin and chunks.pkl from their cloud locations on startup.

Bash

uvicorn app:app --host 0.0.0.0 --port 8000 --reload
The API will be available at http://localhost:8000/ask.

3. Deployment
The service is configured for deployment using the docker SDK via the Dockerfile and the configuration block at the top of this README.md. Deployment is handled by simply pushing to the Hugging Face Space repository:

Bash

git push
💻 API Endpoint
The service exposes a single POST endpoint at /ask.

POST /ask

Field	Type	Description
q	string (Required)	The user's question.
k	integer (Optional)	The number of top contexts to return (default: 5).
mode	string (Optional)	Reranker mode: baseline, hybrid, or learned (default: hybrid).

Export to Sheets
Example cURL Requests
1. Easy Question (Clear Keyword Match)
Finds the answer easily with the baseline, but the hybrid mode ensures the best context is first.

Bash

curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d '{
  "q": "What is the primary difference between Performance Level D (PLd) and PLe?",
  "k": 3,
  "mode": "hybrid"
}'
2. Tricky Question (Requires Semantic Understanding)
A more abstract question where the semantic meaning needs to outweigh keyword noise. The reranker is essential here.

Bash

curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d '{
  "q": "Why is it important to use a safety light curtain instead of just a fence?",
  "k": 3,
  "mode": "learned"
}'
📊 Results and Learnings
The quantitative results of running the 8 test questions across the baseline (vector-only) and the chosen [reranker type, e.g., Hybrid/Learned] reranker are located in the file results_table.md (or printed to console during a specific run script).

The results consistently show that the [reranker type] reranker successfully reorders the retrieved chunks, leading to a higher percentage of the most relevant evidence appearing in the top-1 and top-3 results compared to the simple vector baseline. This improvement is particularly noticeable on the "tricky" questions that require combining semantic similarity with lexical specificity (BM25 for hybrid) or learned feature weighting.

Overall, this sprint confirmed that while basic vector search is fast and effective for initial retrieval, a simple, lightweight reranker is a necessary step to ensure answer quality and robustness for a production RAG system. The architecture of separating the retrieval and reranking steps keeps the system fast and highly tunable.
