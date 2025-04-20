from typing import Dict, Any
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentLoader:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

    def load_submission(self, code_path: str) -> Dict[str, Any]:
        """Load and preprocess a code submission."""
        try:
            with open(code_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Split the code into chunks for better processing
            chunks = self.text_splitter.split_text(content)
            
            return {
                'content': content,
                'chunks': chunks,
                'metadata': {
                    'file_path': code_path,
                    'file_name': os.path.basename(code_path),
                    'file_size': os.path.getsize(code_path)
                }
            }
        except Exception as e:
            raise Exception(f"Error loading submission: {str(e)}")

    def load_marking_scheme(self, marking_scheme_path: str) -> Dict[str, Any]:
        """Load and preprocess a marking scheme."""
        try:
            with open(marking_scheme_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Split the marking scheme into chunks
            chunks = self.text_splitter.split_text(content)
            
            return {
                'content': content,
                'chunks': chunks,
                'metadata': {
                    'file_path': marking_scheme_path,
                    'file_name': os.path.basename(marking_scheme_path)
                }
            }
        except Exception as e:
            raise Exception(f"Error loading marking scheme: {str(e)}") 