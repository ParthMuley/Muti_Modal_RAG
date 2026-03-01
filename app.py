import streamlit as st
import os
import qdrant_client
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core import Settings
from ingest import setup_settings
from config import QDRANT_PATH, COLLECTION_NAME, MODEL_PROVIDER, RERANK_MODEL_NAME

# --- Streamlit UI Setup ---
st.set_page_config(
    page_title="SOTA MultiModal RAG (Agentic)", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar Configuration
with st.sidebar:
    st.title("⚙️ RAG Configuration")
    st.info(f"**Main Model:** {MODEL_PROVIDER}")
    st.info(f"**Reranker:** {RERANK_MODEL_NAME.split('/')[-1]}")
    st.write("---")
    st.markdown("""
    ### 🛡️ SOTA Features:
    - **Contextual Retrieval:** Anthropic method (Global Hints).
    - **Agentic Self-Correction:** Evaluates context before answering.
    - **Cross-Encoder Reranking:** Filters top 10 results.
    """)
    
    if st.button("🔄 Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

st.title("🤖 Agentic MultiModal RAG Assistant")
st.markdown(f"**Mode:** 🚀 *Self-Correction & Contextual Retrieval Enabled*")

def initialize_engine():
    """Initializes the query engine with Qdrant + SOTA Reranking."""
    setup_settings()
    
    if not os.path.exists(QDRANT_PATH):
        st.error(f"No database found at {QDRANT_PATH}. Run `python ingest.py` first.")
        st.stop()
        
    client = qdrant_client.QdrantClient(path=QDRANT_PATH)
    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
    
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    
    rerank_postprocessor = SentenceTransformerRerank(
        model=RERANK_MODEL_NAME, 
        top_n=5
    )
    
    return index.as_query_engine(
        similarity_top_k=10, 
        node_postprocessors=[rerank_postprocessor]
    )

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "images" in message:
            cols = st.columns(2)
            for idx, img in enumerate(message["images"]):
                with cols[idx % 2]:
                    st.image(img["path"], caption=img["caption"], use_container_width=True)

# Lazy Load Query Engine
if "query_engine" not in st.session_state:
    with st.spinner(f"Loading {MODEL_PROVIDER} & BGE Reranker..."):
        st.session_state.query_engine = initialize_engine()

def agentic_query(prompt):
    """
    Agentic Loop:
    1. Retrieve & Rerank.
    2. Evaluate if context is enough.
    3. Re-query if not.
    4. Synthesize.
    """
    with st.status("🔍 Processing Query (Agentic Loop)...") as status:
        # Step 1: Initial Retrieval
        status.update(label="📡 Retrieving initial context...")
        response = st.session_state.query_engine.query(prompt)
        
        # Step 2: Evaluation
        status.update(label="🧠 Evaluating context sufficiency...")
        eval_prompt = f"Given the following retrieved context, can the question '{prompt}' be answered accurately? \n\nContext: {response.source_nodes[0].text[:1000]}... \n\nAnswer only 'YES' or 'NO'."
        eval_result = str(Settings.llm.complete(eval_prompt)).strip().upper()
        
        if "NO" in eval_result:
            status.update(label="🔄 Context insufficient. Attempting re-query with expansion...")
            # Simple rephrase for better retrieval
            rephrase_prompt = f"Rephrase this technical question to be more specific for a vector search: {prompt}"
            better_query = str(Settings.llm.complete(rephrase_prompt)).strip()
            response = st.session_state.query_engine.query(better_query)
        
        status.update(label="✍️ Synthesizing final answer...", state="complete")
        return response

# Chat Input
if prompt := st.chat_input("Ask a technical question..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        try:
            response = agentic_query(prompt)
            
            final_text = response.response
            source_nodes = response.source_nodes
            
            # Extract unique images
            retrieved_images = []
            seen_paths = set()
            for node in source_nodes:
                if "image_path" in node.metadata:
                    img_path = node.metadata["image_path"]
                    if os.path.exists(img_path) and img_path not in seen_paths:
                        retrieved_images.append({
                            "path": img_path,
                            "caption": f"Retrieved from Page {node.metadata.get('page', '?')}"
                        })
                        seen_paths.add(img_path)

            st.markdown(final_text)
            
            if retrieved_images:
                st.write("---")
                st.markdown("### 🖼️ Relevant Diagrams")
                cols = st.columns(2)
                for idx, img in enumerate(retrieved_images):
                    with cols[idx % 2]:
                        st.image(img["path"], caption=img["caption"], use_container_width=True)

            with st.expander("🔍 View Contextualized Source Nodes"):
                for node in source_nodes:
                    st.write(f"**Score:** `{node.score:.4f}`")
                    st.info(node.text) # Text is now contextualized with global hints
                    st.divider()

            st.session_state.messages.append({
                "role": "assistant", 
                "content": final_text,
                "images": retrieved_images
            })

        except Exception as e:
            st.error(f"Error: {e}")
