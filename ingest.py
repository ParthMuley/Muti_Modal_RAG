import os
import qdrant_client
import time
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core.schema import TextNode
import google.generativeai as genai
from multimodal_utils import extract_images_from_pdf
from PIL import Image

# Load environment variables
load_dotenv()

def generate_image_caption(image_path, api_key):
    """Uses Gemini 2.5 Flash to generate a description for an image with retries."""
    genai.configure(api_key=api_key)
    # Using 2.5-flash which is the stable 2026 model
    model = genai.GenerativeModel("models/gemini-2.5-flash")
    
    img = Image.open(image_path)
    prompt = "Describe this architecture diagram or image in detail. Focus on the components, services, and how they interact. This will be used for a RAG system, so be as descriptive as possible for accurate retrieval."
    
    # Simple retry logic for rate limits
    for attempt in range(3):
        try:
            response = model.generate_content([prompt, img])
            return response.text
        except Exception as e:
            if "429" in str(e):
                print(f"Rate limit hit for {image_path}. Waiting 15 seconds...")
                time.sleep(15)
            else:
                raise e
    return "Failed to generate caption after 3 attempts."

def setup_settings():
    """Configure LlamaIndex to use Google Gemini for LLM and local HuggingFace for embeddings."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Please set GOOGLE_API_KEY in your .env file")

    # Use Gemini 2.0 Flash Lite for the LLM
    # Lite models often have separate/higher quotas.
    Settings.llm = Gemini(model="models/gemini-2.0-flash-lite", api_key=api_key)
    
    # Use a local embedding model (free, no rate limits)
    print("Loading local embedding model (BAAI/bge-small-en-v1.5)...")
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50

def ingest_data(data_dir="./data", qdrant_path="./qdrant_data"):
    """Reads documents, images, and scraped links, then embeds them into Qdrant."""
    setup_settings()
    api_key = os.getenv("GOOGLE_API_KEY")

    # Clear old data if it exists to prevent duplicates
    if os.path.exists(qdrant_path):
        print("Cleaning up old vector data...")
        import shutil
        shutil.rmtree(qdrant_path, ignore_errors=True)

    print(f"Reading documents from {data_dir}...")
    documents = SimpleDirectoryReader(data_dir).load_data()
    print(f"Loaded {len(documents)} document chunks from PDF.")

    # New: Add scraped link content
    scraped_dir = "./scraped_content"
    if os.path.exists(scraped_dir):
        print(f"Reading scraped web content from {scraped_dir}...")
        web_docs = SimpleDirectoryReader(scraped_dir).load_data()
        print(f"Loaded {len(web_docs)} chunks from web scraping.")
        documents.extend(web_docs)

    all_nodes = []
    
    # Add text document nodes
    for doc in documents:
        all_nodes.append(TextNode(text=doc.text, metadata=doc.metadata))

    # Extract and caption images
    extracted_img_dir = "./extracted_images"
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(data_dir, pdf_file)
        print(f"Extracting images from {pdf_path}...")
        image_paths = extract_images_from_pdf(pdf_path, extracted_img_dir)
        
        for i, img_path in enumerate(image_paths):
            print(f"[{i+1}/{len(image_paths)}] Generating caption for {img_path}...")
            try:
                caption = generate_image_caption(img_path, api_key)
                image_node = TextNode(
                    text=f"Image Caption: {caption}",
                    metadata={
                        "image_path": img_path,
                        "type": "image",
                        "source_pdf": pdf_file
                    }
                )
                all_nodes.append(image_node)
                # Wait 5 seconds between images to avoid rate limits (20 RPM limit)
                time.sleep(5)
            except Exception as e:
                print(f"Error generating caption for {img_path}: {e}")

    # Initialize a local Qdrant client
    client = qdrant_client.QdrantClient(path=qdrant_path)
    
    # Set up the vector store
    vector_store = QdrantVectorStore(client=client, collection_name="tech_docs")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print(f"Embedding and indexing {len(all_nodes)} nodes locally...")
    index = VectorStoreIndex(
        all_nodes,
        storage_context=storage_context,
        show_progress=True
    )
    print("Indexing complete!")
    return index

if __name__ == "__main__":
    os.makedirs("./data", exist_ok=True)
    if not os.listdir("./data"):
        print("Please place files in './data' before running.")
    else:
        ingest_data()
