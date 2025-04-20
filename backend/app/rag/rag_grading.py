from app.rag.rag_llm import GraderLLM
from dataclasses import dataclass
import os
import json
from app.rag.document_loader import DocumentLoader
from app.rag.retriever import Retriever
from typing import List, Dict, Any
import traceback

"""
Pipeline:
1. Preprocessing
	- Compute lines of code
	- Compute number of files
    - Code style analysing
        - Pass extracted metadata to the code style analyser
        - Ask LLM to assess comment quality, code coverage, and modularity
2. RAG Integration
    - Load and process code submission
    - Load and process marking scheme
    - Retrieve similar past submissions and feedback
3. Grading
	- Get all the metadata, code style analysis, and retrieved examples
	- Fit into the grading prompt template with the original code, rubrics, metadata, and examples
	- Get the grade and feedback from LLM
	- Return to frontend
"""


@dataclass
class Metadata:
    lines_of_code: int
    num_files: int


class Grader:
    def __init__(self, index_dir: str = "vector_store"):
        # Initialize LLM with required parameters
        self.llm = GraderLLM()
        self.document_loader = DocumentLoader()
        self.retriever = Retriever(index_dir=index_dir)

    def _preprocess(self, extract_path: str, code_files: list) -> tuple[Metadata, str]:
        num_files = len(code_files)
        code = ""
        num_code_files = 0
        for filename in code_files:
            if not filename.lower().endswith((".py", ".c", ".java")):
                continue
            file_path = os.path.join(extract_path, filename)
            num_code_files += 1
            with open(file_path, "r", encoding='utf-8') as file:
                code += f"///{filename}///\n{file.read()}"
        lines_of_code = code.count("\n") + 1 - num_code_files

        try:
            return Metadata(
                lines_of_code=lines_of_code,
                num_files=num_files,
            ), code
        except Exception as e:
            print(e)
            return Metadata(
                lines_of_code=lines_of_code,
                num_files=num_files,
            ), code

    def _prepare_rag_context(self, code: str, rubrics: str) -> List[Dict[str, Any]]:
        """Prepare context for RAG by retrieving similar submissions."""
        try:
            # Create a query combining code and rubrics
            query = f"Code: {code}\nRubrics: {rubrics}"
            
            # Retrieve similar submissions
            print("Retrieving similar submissions...")
            similar_docs = self.retriever.retrieve_relevant_documents(query, k=3)
            
            # Format the retrieved documents
            context = []
            for doc in similar_docs:
                context.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
            
            print(f"Retrieved {len(context)} similar submissions")
            return context
        except Exception as e:
            print(f"Error preparing RAG context: {str(e)}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            return []

    def grade(
        self,
        extract_path: str,
        code_files: list,
        language: str,
        rubrics: str,
        code_run_output: str = "",
        rag_context: List[Dict[str, Any]] = None,
    ) -> tuple[int, str]:
        try:
            # Preprocess the submission
            print("Preprocessing submission...")
            metadata, code = self._preprocess(extract_path, code_files)
            
            # Get RAG context if not provided
            if rag_context is None:
                print("Preparing RAG context...")
                rag_context = self._prepare_rag_context(code, rubrics)
            else:
                print("Using provided RAG context...")
            
            # Prepare input for grading with RAG context
            input_data = {
                "code": code,
                "language": language,
                "rubrics": rubrics,
                "lines_of_code": metadata.lines_of_code,
                "num_files": metadata.num_files,
                "code_run_output": code_run_output,
                "similar_examples": rag_context
            }
            
            print("Sending request to LLM...")
            result = self.llm.invoke(input_data)
            print(f"LLM Response: {result}")

            if not isinstance(result, dict):
                raise ValueError(f"Invalid response type: {type(result)}")

            marks = result.get("marks", 0)
            feedback = result.get("feedback", "No feedback available")

            # Ensure feedback is a string
            if not isinstance(feedback, str):
                feedback = str(feedback) if feedback is not None else "No feedback available"

            # Convert marks to int if it's a string
            if isinstance(marks, str):
                try:
                    marks = int(marks)
                except ValueError:
                    marks = 0

            return marks, feedback.strip()
        except json.JSONDecodeError as e:
            error_msg = f"JSON parsing error: {str(e)}\nResponse: {result if 'result' in locals() else 'No response'}"
            print(error_msg)
            return 0, error_msg
        except Exception as e:
            error_msg = f"Error in grading: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return 0, error_msg