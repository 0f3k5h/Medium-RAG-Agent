# Medium Article RAG Assistant

An end-to-end Retrieval-Augmented Generation (RAG) agent built using FastAPI, Pinecone, and LLMod API. This system is designed to provide highly accurate answers to user queries, grounded strictly within a dataset of roughly 7,600 English Medium articles, ensuring complete factual alignment without external hallucinations.

## Live Deployment & Repository
* **Production URL:** https://medium-rag-agent.vercel.app/
* **GitHub Repository:** https://github.com/0f3k5h/Medium-RAG-Agent

---

## Architecture Overview

The system implements a production-grade RAG pipeline divided into two core phases:

1. **Ingestion & Embedding Pipeline:**
   * **Data Extraction:** Raw article data is parsed from the source CSV, combining title, author, URL, tags, and text content.
   * **Semantic Chunking:** Text is processed into distinct, overlapping chunks to preserve contextual continuity across boundaries.
   * **Vectorization:** Text blocks are vectorized using a dense embedding model (1536 dimensions) via a batched API client wrapper.
   * **Vector Store Upserting:** Vectors are indexed into a Pinecone vector database using a Cosine similarity metric, populated with rich metadata attributes (`title`, `authors`, `url`, `tags`, and `text`).

2. **Retrieval & Inference Pipeline (FastAPI Backend):**
   * **Query Vectorization:** Incoming user prompts are transformed into the same vector space.
   * **Vector Search:** Pinecone performs a top-$K$ nearest-neighbor search to isolate the most relevant text fragments.
   * **Prompt Augmentation:** A strictly bounded context block is constructed out of the retrieved metadata and text passages.
   * **Grounded Generation:** The augmented prompt is evaluated by the language model under rigid system instructions prohibiting external knowledge utilization.

---

## Hyperparameter Selection & Rationale

Per the project notes regarding workflow design and minimizing API costs, I opted not to brute-force re-embed the entire 7,600-article corpus multiple times. Instead, I researched with Gemini to establish an optimal baseline for processing blog-style content before embedding the data. 

I applied the following parameters, which yielded highly accurate, context-rich results during testing without requiring budget-consuming iterations.

### Selected Configuration
* **Chunk Size:** 500 tokens
* **Overlap Ratio:** 0.20 (20%)
* **Top-K Retrieval:** 5 documents

### Theoretical Rationale (Why this outperformed alternatives)
* **Why 500 / 20%?** 500 tokens perfectly captures 1 to 2 full paragraphs of a Medium article, which generally represents one complete semantic thought. The 20% (100-token) overlap ensures that if a crucial sentence or list spans across a chunk boundary, the context is not severed. A Top-K of 5 provides enough diversity for multi-result queries without overwhelming the LLM.
* **Why not smaller? (e.g., 250 tokens):** Smaller chunks cause severe semantic fragmentation. Multi-step arguments or listicles get sliced in half. The retrieval model might pull a chunk containing a conclusion, but lack the chunk containing the premise, leading to poor LLM summarization.
* **Why not larger? (e.g., 1000 tokens):** Larger chunks create a high noise-to-signal ratio. Passing five 1000-token chunks fills the LLM's context window with vast amounts of irrelevant text, which dilutes the specific facts needed for precise fact-retrieval prompts and increases the risk of hallucination.

---

## API Documentation

The backend service exposes two key public endpoints:

### 1. Query Assistant
* **Endpoint:** `POST /api/prompt`
* **Headers:** `Content-Type: application/json`
* **Payload:**
  ```json
  {
    "question": "Your question here"
  }
  ```
* **Response Format:** Returns a structured JSON payload containing the grounded answer, retrieved raw context chunks, and the full augmented prompt.

### 2. Pipeline Configuration Stats
* **Endpoint:** `GET /api/stats`
* **Response Format:**
  ```json
  {
    "chunk_size": 500,
    "overlap_ratio": 0.2,
    "top_k": 5
  }
  ```

---

## Local Setup & Installation

To run this backend application locally on your machine:

1. Clone the repository and navigate to the project directory.
2. Install the necessary production dependencies from `requirements.txt`.
3. Configure your local environment variables with your specific keys (`LLMOD_API_KEY` and `PINECONE_API_KEY`) using your operating system's standard method.
4. Boot up the local development server:
   ```bash
   uvicorn api.index:app --reload