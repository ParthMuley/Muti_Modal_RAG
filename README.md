# Multi-Modal RAG (Retrieval-Augmented Generation)

This project implements a Multi-Modal RAG system that can process and retrieve information from various formats, including text and images, using advanced AI models like Gemini 1.5.

## Project Overview

Traditional RAG systems are often limited to text-in and text-out. This system shifts that paradigm by incorporating multi-modal context (text, images, etc.) into the retrieval and synthesis process.

### Key Features

- **Unified Embedding Space:** Captures nuances across different modalities for more robust retrieval.
- **Multimodal Retrieval:** Performs similarity searches against a vector database (Qdrant) containing text chunks and image metadata.
- **Multimodal LLM Synthesis:** Uses Gemini 1.5 Pro/Flash to synthesize answers from both retrieved text and visual data.
- **Efficient Ingestion:** Extracts text and images from PDF documents for processing and indexing.

## Tech Stack

- **LLM:** Google Gemini 1.5 (Pro/Flash)
- **Vector Database:** Qdrant
- **Framework:** Streamlit (for the UI)
- **Orchestration:** Python-based ingestion and retrieval logic
- **Libraries:** `google-generativeai`, `qdrant-client`, `pypdfium2`, etc.

## Getting Started

### Prerequisites

- Python 3.10+
- A Google Gemini API Key

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ParthMuley/Muti_Modal_RAG.git
   cd Muti_Modal_RAG
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the root directory and add your Gemini API key:
   ```env
   GOOGLE_API_KEY=your_api_key_here
   ```

### Usage

1. **Ingest Data:**
   Run the ingestion script to process your PDF documents:
   ```bash
   python ingest.py
   ```

2. **Run the App:**
   Start the Streamlit interface:
   ```bash
   streamlit run app.py
   ```

## Implementation Strategy

The project follows a "Stable & Precise" native multimodal approach:
- **Flow:** Uses a Multi-Vector Retriever, storing high-dimensional embeddings in a vector database.
- **Pros:** High fidelity; allows the LLM to perform its own reasoning on the visual data.

## Architectural Trade-offs

| Component | Choice | Key Advantage |
| :--- | :--- | :--- |
| **LLM** | Gemini 1.5 | Massive 2M token context window; native multimodal reasoning. |
| **Vector DB** | Qdrant | Scalable and efficient similarity search for high-dimensional vectors. |

---
*For more detailed theoretical background, refer to the [readme.pdf](./readme.pdf).*
