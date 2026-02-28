# MultiModal RAG: Development Thoughts & Planning

This document tracks the architectural decisions, features, and implementation steps for the MultiModal RAG system.

## Project Vision
A production-grade RAG system capable of retrieving and reasoning across text and images from technical documentation (e.g., AWS/Azure architecture guides).

---

## Phase 1: MVP (Minimal Viable Product) - COMPLETED
**Goal:** Build a functional text-based RAG pipeline using cloud LLMs and local embeddings to minimize cost and avoid rate limits.

### Features Implemented
- [x] **Project Scaffolding:** Python virtual environment, `requirements.txt`.
- [x] **Local Vector Store:** Qdrant (Local/In-memory mode) for fast, zero-cost storage.
- [x] **Local Embeddings:** Switched from Google API to `BAAI/bge-small-en-v1.5` to bypass rate limits (429 errors) during heavy ingestion.
- [x] **Cloud LLM Reasoning:** Gemini 1.5 Flash for high-quality, long-context answers.
- [x] **Ingestion Pipeline:** Automated PDF parsing and chunking (153 chunks for the sample AWS PDF).
- [x] **Chat UI:** Streamlit interface with chat history and "Source Context" expanders.

### Steps Taken
1.  **Step 1:** Initialized environment and installed LlamaIndex + Qdrant.
2.  **Step 2:** Attempted Google Cloud Embeddings; hit rate limits.
3.  **Step 3:** Implemented `ingest.py` with local HuggingFace embeddings.
4.  **Step 4:** Built `app.py` Streamlit interface.

---

## Phase 2: Adding the "MultiModal" (Vision) - IN PROGRESS
**Goal:** Extract architecture diagrams from PDFs and allow Gemini to "see" them when answering.

### Planned Features
- [ ] **Image Extraction:** Use `pdf2image` or `pdfplumber` to pull images/diagrams from documentation.
- [ ] **Vision Ingestion:** 
    - For every extracted image, use Gemini 1.5 Flash to generate a **textual description/caption**.
    - Store these captions in the vector store with a reference (path/link) to the original image.
- [ ] **MultiModal Retrieval:**
    - Search both text chunks and image captions.
    - If an image caption is highly relevant, retrieve the raw image.
- [ ] **Multimodal Synthesis:**
    - Pass the retrieved text AND the raw images to Gemini 1.5 Flash.
    - *Example:* "Look at this diagram (Image A) and the text below. Explain how the Load Balancer connects to the EC2 instances."
- [ ] **UI Update:** Display retrieved images directly in the Streamlit chat when they are used as context.

---

## Phase 2.5: Recursive Link Ingestion (New Idea)
**Goal:** Expand the knowledge base by crawling external links found within the PDF documents.

### Planned Features
- [ ] **Link Extraction:** Automatically identify and extract all external URLs from the PDF source.
- [ ] **Web Scraping:** Fetch and clean the text content from the extracted URLs.
- [ ] **Unified Indexing:** Treat web content as first-class citizens in the vector store, linked back to the parent document.
- [ ] **MultiModal Web Capture:** (Optional) Take screenshots of linked pages to capture diagrams not present in the PDF.

---

## Hallucination Guardrails & Truthfulness
**Goal:** Ensure the system never "makes up" technical facts.

### Best Practices to Implement
- [ ] **Negative Constraints:** Update system prompts to force "I don't know" responses when context is missing.
- [ ] **Self-Correction Loop:** Implement a "Reflector" step where the LLM critiques its own answer against the source text.
- [ ] **RAGAS Evaluation:** Implement automated scoring for Faithfulness, Relevancy, and Precision.
- [ ] **Citation UI:** Make the UI "clickable"—clicking a sentence in the AI's answer should highlight the source chunk it came from.

---

## Phase 3: Production Readiness
**Goal:** Move from local scripts to a scalable architecture.

### Planned Features
- [ ] **Managed Vector DB:** Move from local Qdrant to Qdrant Cloud or Milvus.
- [ ] **Asynchronous Ingestion:** Use a task queue (Celery/Redis) for PDF processing.
- [ ] **Semantic Caching:** Store common queries in Redis to save LLM costs.
- [ ] **Evaluation:** Implement RAGAS or a "Vibe Check" framework to measure accuracy.

---

## Technical Trade-offs Made
1.  **Local vs. Cloud Embeddings:** Chose local `BGE` to avoid the 429 "Resource Exhausted" errors on the Gemini Free Tier. This makes the system more robust for large document ingestion.
2.  **Gemini 2.5 Flash:** Using Gemini 2.5 Flash for both LLM and Vision tasks as it's the stable high-quota model for the Feb 2026 environment.
3.  **Robust Ingestion:** Implemented a 5-second delay and retry logic (3 attempts with 15s wait) during image captioning to gracefully handle the 20 RPM rate limit on the Gemini Free Tier.
