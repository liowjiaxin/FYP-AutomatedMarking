import os
import PyPDF2
import zipfile
import tempfile
import shutil
from typing import Tuple, List


def _get_code_files(directory: str, language: str) -> List[str]:
    """Get list of code files from directory."""
    extensions = {"C": [".c", ".h"], "Python": [".py"], "Java": [".java"]}[language]
    code_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                code_files.append(os.path.join(root, file))
    return code_files


def read_code(directory: str, language: str) -> Tuple[str, int]:
    """Read all code files in a directory and concatenate into one string."""
    # Handle zip files
    if directory.endswith('.zip'):
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Extract zip file
                with zipfile.ZipFile(directory, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Get code files
                code_files = _get_code_files(temp_dir, language)
                if not code_files:
                    raise ValueError(f"No {language} code files found in {directory}")
                
                # Read all code files
                code = []
                for file_path in code_files:
                    try:
                        with open(file_path, "r", encoding='utf-8') as f:
                            relative_path = os.path.relpath(file_path, temp_dir)
                            code.append(f"# File: {relative_path}\n{f.read()}\n")
                    except UnicodeDecodeError:
                        print(f"Warning: Could not read {file_path} - binary file?")
                
                return "\n".join(code), len(code_files)
            except Exception as e:
                raise ValueError(f"Error processing zip file {directory}: {str(e)}")

    # Handle regular directory
    code_files = _get_code_files(directory, language)
    if not code_files:
        raise ValueError(f"No {language} code files found in {directory}")
    
    # Read all code files
    code = []
    for file_path in code_files:
        try:
            with open(file_path, "r", encoding='utf-8') as f:
                relative_path = os.path.relpath(file_path, directory)
                code.append(f"# File: {relative_path}\n{f.read()}\n")
        except UnicodeDecodeError:
            print(f"Warning: Could not read {file_path} - binary file?")
    
    return "\n".join(code), len(code_files)


def read_pdf(pdf_path: str) -> str:
    """Extract text content from a PDF marking scheme."""
    text = []
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text.append(page.extract_text())
    except Exception as e:
        raise RuntimeError(f"Failed to read PDF: {str(e)}")

    return "\n".join(text) 