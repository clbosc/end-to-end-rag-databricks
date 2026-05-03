# 🤖 RAG Research Assistant - ML Thesis

This project implements an end-to-end **Retrieval-Augmented Generation (RAG)** architecture on the **Databricks** platform. It allows for intelligent querying of technical literature (specifically the *Big Book of Machine Learning*) using state-of-the-art foundation models.

## 🚀 Technical Architecture

*   **Large Language Model (LLM):** Llama 3.1 405B (via Databricks Model Serving).
*   **Embeddings:** BGE Base English v1.5.
*   **Vector Database:** Databricks Vector Search.
*   **Data Storage:** Delta Lake with **Change Data Feed (CDF)** enabled for incremental updates.
*   **Orchestration:** LangChain (LCEL).

## 🛠️ Key Features

- **Automated Ingestion:** Seamless text extraction from PDFs stored in Unity Catalog Volumes.
- **Incremental Indexing:** Using Delta CDF, the system only re-indexes new or modified documents, significantly reducing compute costs.
- **Advanced Retrieval:** Semantic search pipeline that retrieves the top-K most relevant document chunks.
- **Source-Grounded Responses:** The AI generates answers strictly based on the provided technical context to minimize hallucinations.

## 📋 Prerequisites

- A Databricks workspace with **Unity Catalog** enabled.
- A configured **Vector Search Endpoint**.
- Python libraries:
  ```bash
  pip install databricks-langchain langchain pymupdf