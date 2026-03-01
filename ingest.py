import os
import qdrant_client
import time
import shutil
import json
import hashlib
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.gemini import Gemini
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader
from llama_index.core.schema import TextNode

from config import (
    MODEL_PROVIDER, OLLAMA_TEXT_MODEL, OLLAMA_BASE_URL,
    GEMINI_MODEL, GEMINI_API_KEY, QDRANT_PATH, COLLECTION_NAME,
    EMBED_MODEL_NAME, DATA_DIR, EXTRACTED_IMAGES_DIR, SCRAPED_CONTENT_DIR
)
from multimodal_utils import extract_images_from_pdf, generate_caption

MANIFEST_FILE = "ingestion_manifest.json"
CAPTION_CACHE_FILE = "caption_cache.json"
CHUNK_CONTEXT_CACHE_FILE = "chunk_context_cache.json"

def get_file_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def load_json(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return {}

def save_json(file_path, data):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def setup_settings():
    if MODEL_PROVIDER == "OLLAMA":
        Settings.llm = Ollama(model=OLLAMA_TEXT_MODEL, base_url=OLLAMA_BASE_URL, request_timeout=120.0)
    else:
        Settings.llm = Gemini(model=GEMINI_MODEL, api_key=GEMINI_API_KEY)
    
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50

def generate_chunk_context(doc_title, chunk_text):
    """Generates a 1-sentence context for a chunk to improve retrieval (Anthropic method)."""
    prompt = f"You are an expert document analyzer. Given the document titled '{doc_title}', provide a single brief sentence that explains where this chunk fits in the overall document context. \n\nChunk: {chunk_text[:500]}..."
    try:
        response = Settings.llm.complete(prompt)
        return str(response).strip()
    except Exception as e:
        print(f"Error generating chunk context: {e}")
        return ""

def ingest_data():
    setup_settings()
    manifest = load_json(MANIFEST_FILE)
    caption_cache = load_json(CAPTION_CACHE_FILE)
    chunk_context_cache = load_json(CHUNK_CONTEXT_CACHE_FILE)
    
    client = qdrant_client.QdrantClient(path=QDRANT_PATH)
    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    all_nodes = []

    # --- 1. Process PDFs with Contextual Retrieval ---
    if os.path.exists(DATA_DIR):
        pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
        for pdf_file in pdf_files:
            file_path = os.path.join(DATA_DIR, pdf_file)
            current_hash = get_file_hash(file_path)
            
            if manifest.get(pdf_file) == current_hash:
                print(f"Skipping PDF (No changes): {pdf_file}")
                # We still need to load nodes for the indexer if we were doing a full rebuild,
                # but for simplicity in this MVP, we only append new ones.
                continue
            
            print(f"Processing PDF with Contextual Retrieval: {pdf_file}")
            reader = SimpleDirectoryReader(input_files=[file_path])
            documents = reader.load_data()
            
            for i, doc in enumerate(documents):
                chunk_id = f"{pdf_file}_{i}"
                if chunk_id in chunk_context_cache:
                    context_hint = chunk_context_cache[chunk_id]
                else:
                    print(f"Generating context for chunk {i+1}/{len(documents)}...")
                    context_hint = generate_chunk_context(pdf_file, doc.text)
                    chunk_context_cache[chunk_id] = context_hint
                    save_json(CHUNK_CONTEXT_CACHE_FILE, chunk_context_cache)
                
                # PREPEND CONTEXT (SOTA Anthropic Method)
                contextualized_text = f"DOCUMENT CONTEXT: {context_hint}\n\nCONTENT: {doc.text}"
                node = TextNode(text=contextualized_text, metadata=doc.metadata)
                node.metadata["type"] = "text"
                all_nodes.append(node)
            
            # Images
            image_data = extract_images_from_pdf(file_path, EXTRACTED_IMAGES_DIR)
            for i, img_info in enumerate(image_data):
                img_path = img_info["path"]
                caption = caption_cache.get(img_path) or generate_caption(img_path)
                caption_cache[img_path] = caption
                save_json(CAPTION_CACHE_FILE, caption_cache)

                image_node = TextNode(
                    text=f"IMAGE CONTEXT: This diagram is from {pdf_file}.\nDESCRIPTION: {caption}",
                    metadata={
                        "image_path": img_path,
                        "type": "image",
                        "source_pdf": pdf_file,
                        "page": img_info["page"]
                    }
                )
                all_nodes.append(image_node)
            
            manifest[pdf_file] = current_hash

    # --- 2. Process Web Content ---
    if os.path.exists(SCRAPED_CONTENT_DIR):
        web_files = [f for f in os.listdir(SCRAPED_CONTENT_DIR) if f.endswith(".txt")]
        for web_file in web_files:
            file_path = os.path.join(SCRAPED_CONTENT_DIR, web_file)
            current_hash = get_file_hash(file_path)
            if manifest.get(web_file) == current_hash: continue
            
            print(f"Processing web content: {web_file}")
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            node = TextNode(text=content, metadata={"source": web_file, "type": "web_content"})
            all_nodes.append(node)
            manifest[web_file] = current_hash

    if all_nodes:
        print(f"Indexing {len(all_nodes)} nodes with Contextual Retrieval...")
        index = VectorStoreIndex(all_nodes, storage_context=storage_context, show_progress=True)
        save_json(MANIFEST_FILE, manifest)
        print("Ingestion complete!")
    else:
        print("No new content to ingest.")

if __name__ == "__main__":
    ingest_data()
