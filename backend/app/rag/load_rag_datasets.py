from rag.retriever import Retriever
import json
from datetime import datetime
import os

def load_rag_datasets():
    print("Starting to load RAG datasets...")
    
    # Initialize the retriever
    vector_store_path = os.path.join(os.path.dirname(__file__), "vector_store")
    print(f"Vector store path: {vector_store_path}")
    os.makedirs(vector_store_path, exist_ok=True)
    retriever = Retriever(index_dir=vector_store_path)
    
    # List of dataset files to load
    dataset_files = [
        "rag_dataset_lab1.json",
        "rag_dataset_lab2.json",
        "rag_dataset_lab3.json"
    ]
    
    total_submissions = 0
    
    # Process each dataset file
    for dataset_file in dataset_files:
        file_path = os.path.join(os.path.dirname(__file__), dataset_file)
        print(f"\nProcessing file: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"Warning: File {dataset_file} not found, skipping...")
            continue
            
        try:
            # Load the JSON file with UTF-8 encoding
            print(f"Loading JSON file: {dataset_file}")
            with open(file_path, 'r', encoding='utf-8') as f:
                submissions = json.load(f)
            print(f"Successfully loaded {len(submissions)} submissions from {dataset_file}")
            
            # Prepare documents for vector store
            documents = []
            for submission in submissions:
                documents.append({
                    "content": submission["code"],
                    "metadata": {
                        "student_id": submission["student_id"],
                        "feedback": submission["feedback"],
                        "lab": dataset_file.replace("rag_dataset_", "").replace(".json", ""),
                        "date": datetime.now().strftime("%Y-%m-%d")
                    }
                })
            
            # Add documents to vector store
            print(f"Adding {len(documents)} documents to vector store...")
            retriever.add_documents(documents)
            total_submissions += len(documents)
            print(f"Successfully added {len(documents)} submissions from {dataset_file} to vector store")
            
        except Exception as e:
            print(f"Error processing {dataset_file}: {str(e)}")
            import traceback
            print(f"Full error traceback: {traceback.format_exc()}")
    
    print(f"\nTotal submissions loaded: {total_submissions}")
    print("Vector store has been updated with all submissions")

if __name__ == "__main__":
    load_rag_datasets() 