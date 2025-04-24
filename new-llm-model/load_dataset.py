import os
import pandas as pd
from typing import List, Dict
import logging
import zipfile
from helper import read_code
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class DatasetLoader:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.excel_path = os.path.join(dataset_path, 'feedback.xlsx')
        
        # Check if source code directory exists
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset directory does not exist: {dataset_path}")
            
        # Excel file is optional
        self.has_excel = os.path.exists(self.excel_path)
        if self.has_excel:
            self._validate_excel()

    def _load_examples(self) -> List[Document]:
        """Load examples from Excel file and source code directory"""
        examples = []
        
        # Load from Excel if available
        if self.has_excel:
            try:
                df = pd.read_excel(self.excel_path)
                for _, row in df.iterrows():
                    examples.append(Document(
                        page_content=row['code'],
                        metadata={"feedback": row['feedback']}
                    ))
                logger.info(f"Loaded {len(examples)} examples from Excel")
            except Exception as e:
                logger.warning(f"Error loading Excel file: {str(e)}")
        
        # Load from source code directory
        try:
            code_files = [f for f in os.listdir(self.dataset_path) 
                         if f.endswith('.py') or f.endswith('.zip')]
            for file in code_files:
                file_path = os.path.join(self.dataset_path, file)
                if file.endswith('.zip'):
                    # For zip files, use the read_code helper function
                    code, num_files = read_code(file_path, "Python")
                    examples.append(Document(
                        page_content=code,
                        metadata={"source": file, "type": "zip"}
                    ))
                else:
                    # For Python files, read directly
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code = f.read()
                        examples.append(Document(
                            page_content=code,
                            metadata={"source": file, "type": "python"}
                        ))
            logger.info(f"Loaded {len(code_files)} code files")
        except Exception as e:
            logger.error(f"Error loading code files: {str(e)}")
            raise
        
        if not examples:
            raise ValueError("No examples found in dataset directory")
            
        return examples

    def _validate_excel(self):
        """Validate Excel file structure if it exists"""
        try:
            df = pd.read_excel(self.excel_path)
            required_columns = ['code', 'feedback']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Excel file missing required columns: {missing_columns}")
        except Exception as e:
            logger.warning(f"Error validating Excel file: {str(e)}")
            self.has_excel = False

    def load(self) -> List[Document]:
        """Load and return all examples"""
        return self._load_examples() 