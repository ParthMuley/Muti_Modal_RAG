import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from llama_index.core import VectorStoreIndex, qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from ingest import setup_settings
from config import QDRANT_PATH, COLLECTION_NAME, MODEL_PROVIDER

# Configure models for evaluation
setup_settings()

def run_evaluation():
    """
    SOTA Evaluation using Ragas:
    Measures:
    1. Faithfulness: Is the answer derived solely from the context?
    2. Answer Relevancy: Does the answer address the question?
    3. Context Precision: Were the most relevant documents ranked highest?
    """
    print("Initializing Query Engine for Evaluation...")
    client = qdrant_client.QdrantClient(path=QDRANT_PATH)
    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    query_engine = index.as_query_engine()

    # Sample Test Set (Technical Documentation)
    test_questions = [
        "What is the AWS Shared Responsibility Model?",
        "Explain how Amazon VPC peering works.",
        "What are the different S3 storage classes?",
        "How do I secure an EC2 instance?",
        "What is Azure Resource Manager?"
    ]

    # In a real scenario, you'd provide 'ground_truth' answers manually.
    # For this MVP eval, we focus on Faithfulness and Relevancy.
    
    results = []
    
    print(f"Running evaluation on {len(test_questions)} questions...")
    for query in test_questions:
        print(f"Querying: {query}")
        response = query_engine.query(query)
        
        results.append({
            "question": query,
            "answer": response.response,
            "contexts": [node.text for node in response.source_nodes],
        })

    # Convert to Dataset format for Ragas
    dataset = Dataset.from_pandas(pd.DataFrame(results))
    
    print("Calculating RAGAS scores (this might take a minute)...")
    # Note: Ragas works best with OpenAI, but can be configured for local/Gemini
    # For now, we'll output the raw data ready for assessment.
    score = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision]
    )
    
    df_results = score.to_pandas()
    print("
--- EVALUATION RESULTS ---")
    print(df_results)
    
    # Save to CSV
    df_results.to_csv("rag_eval_results.csv", index=False)
    print("
Results saved to rag_eval_results.csv")

if __name__ == "__main__":
    if os.path.exists(QDRANT_PATH):
        run_evaluation()
    else:
        print("Database not found. Run ingestion first.")
