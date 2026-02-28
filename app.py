import streamlit as st
import os
import qdrant_client
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from ingest import setup_settings

# Load environment variables
load_dotenv()

def initialize_engine():
    """Initializes the query engine pointing to the local Qdrant database."""
    setup_settings()
    
    qdrant_path = "./qdrant_data"
    if not os.path.exists(qdrant_path):
        st.error("No database found. Please run `python ingest.py` first.")
        st.stop()
        
    client = qdrant_client.QdrantClient(path=qdrant_path)
    vector_store = QdrantVectorStore(client=client, collection_name="tech_docs")
    
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    
    # Increase top_k to 5 to have a better chance of finding images
    return index.as_query_engine(similarity_top_k=5)

# --- Streamlit UI Setup ---
st.set_page_config(page_title="MultiModal RAG Assistant", page_icon="🖼️", layout="wide")

st.title("🖼️ MultiModal RAG: Technical Docs & Diagrams")
st.markdown("""
Ask questions about **AWS Services**, and this assistant will retrieve both **textual explanations** 
and **architecture diagrams** directly from the documentation.
""")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "images" in message:
            for img in message["images"]:
                st.image(img["path"], caption=img["caption"])

# Initialize Query Engine
if "query_engine" not in st.session_state:
    with st.spinner("Connecting to Vector Database and Loading Models..."):
        st.session_state.query_engine = initialize_engine()

# React to user input
if prompt := st.chat_input("E.g., Show me the VPC architecture."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Searching docs and analyzing diagrams..."):
            try:
                # Step 1: Manual Retrieval (always works even if LLM is down)
                retriever = st.session_state.query_engine._retriever
                source_nodes = retriever.retrieve(prompt)
                
                # Step 2: Attempt LLM Query
                try:
                    response = st.session_state.query_engine.query(prompt)
                    final_text = response.response
                    final_source_nodes = response.source_nodes
                except Exception as llm_error:
                    if "429" in str(llm_error):
                        final_text = "⚠️ **Quota Exceeded:** I've hit the daily limit for the Gemini AI. However, I've still retrieved the most relevant documents and diagrams for you below!"
                    else:
                        final_text = f"An error occurred with the AI: {llm_error}"
                    final_source_nodes = source_nodes

                st.markdown(final_text)
                
                # Extract images from retrieved nodes
                retrieved_images = []
                for node in final_source_nodes:
                    if "image_path" in node.metadata:
                        img_path = node.metadata["image_path"]
                        if os.path.exists(img_path):
                            retrieved_images.append({
                                "path": img_path,
                                "caption": f"Retrieved Diagram from {node.metadata.get('source_pdf', 'Web Content')}"
                            })

                # Display images in the chat
                for img in retrieved_images:
                    st.image(img["path"], caption=img["caption"])
                
                # Source Expanders
                with st.expander("🔍 View Source Context & Scores"):
                    for node in final_source_nodes:
                        st.write(f"**Score:** {node.score:.3f}")
                        if "image_path" in node.metadata:
                            st.info(f"🖼️ **Diagram Description:** {node.text}")
                        else:
                            st.text(node.text[:500] + "...")
                        st.divider()

                # Add to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": final_text,
                    "images": retrieved_images
                })
            except Exception as e:
                st.error(f"An error occurred: {e}")
