import os
import sys
import zipfile
import tempfile
import shutil
import csv
from typing import List, Dict, Any
import json
from PyPDF2 import PdfReader
from tqdm import tqdm

# Add backend path to Python path
backend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                           "backend")
sys.path.append(backend_path)

# Import RAG components from backend
from app.rag.rag_grading import Grader
from app.rag.document_loader import DocumentLoader
from app.rag.retriever import Retriever

# Add the app directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(app_dir)

from app.rag.rag_grading import Grader
from app.rag.document_loader import DocumentLoader
from app.rag.retriever import Retriever

class RAGGrader:
    def __init__(self, vector_store_dir: str = "vector_store"):
        """Initialize the RAG grader with vector store directory."""
        self.grader = Grader(index_dir=vector_store_dir)
        self.document_loader = DocumentLoader()
        self.retriever = Retriever(index_dir=vector_store_dir)
        self.temp_dir = None

    def _extract_zip(self, zip_path: str) -> str:
        """Extract zip file to a temporary directory."""
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Submission zip file not found: {zip_path}")
            
        self.temp_dir = tempfile.mkdtemp()
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.temp_dir)
            return self.temp_dir
        except zipfile.BadZipFile:
            raise ValueError(f"Invalid zip file: {zip_path}")

    def _get_code_files(self, directory: str) -> List[str]:
        """Get list of code files from directory."""
        code_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.py', '.java', '.c', '.cpp', '.js')):
                    code_files.append(os.path.relpath(os.path.join(root, file), directory))
        return code_files

    def _load_marking_scheme(self, scheme_path: str) -> str:
        """Load marking scheme from file."""
        if not os.path.exists(scheme_path):
            raise FileNotFoundError(f"Marking scheme file not found: {scheme_path}")
            
        try:
            if scheme_path.lower().endswith('.pdf'):
                # Handle PDF files
                reader = PdfReader(scheme_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
            else:
                # Handle text files
                with open(scheme_path, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            raise ValueError(f"Error reading marking scheme file: {str(e)}")

    def _prepare_rag_context(self, code: str, rubrics: str) -> List[Dict[str, Any]]:
        """Prepare RAG context by retrieving similar submissions."""
        try:
            # Create a query combining code and rubrics
            MAX_QUERY_CHARS = 15000
            query = f"Code: {code[:MAX_QUERY_CHARS // 2]}\nRubrics: {rubrics[:MAX_QUERY_CHARS // 2]}"

            # Retrieve similar submissions
            print("Retrieving similar submissions for RAG context...")
            similar_docs = self.retriever.retrieve_relevant_documents(query, k=3)
            
            # Format the retrieved documents
            context = []
            for doc in similar_docs:
                context.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
            
            print(f"Retrieved {len(context)} similar submissions for RAG context")
            return context
        except Exception as e:
            print(f"Error preparing RAG context: {str(e)}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            return []

    def grade_submission(
        self,
        submission_zip: str,
        marking_scheme_path: str,
        language: str = "python",
        code_run_output: str = "",
        total_marks: float = 10.0  # Default to 10 if not specified
    ) -> Dict[str, Any]:
        """
        Grade a submission using RAG.
        
        Args:
            submission_zip: Path to the student's submission zip file
            marking_scheme_path: Path to the marking scheme file
            language: Programming language of the submission
            code_run_output: Optional test case output
            total_marks: Total marks for the assignment (default: 10.0)
        
        Returns:
            Dict containing marks and feedback
        """
        try:
            # Extract submission
            print("Extracting submission...")
            extract_path = self._extract_zip(submission_zip)
            
            # Get code files
            print("Finding code files...")
            code_files = self._get_code_files(extract_path)
            if not code_files:
                raise ValueError("No code files found in submission")
            
            print(f"Found {len(code_files)} code files:")
            for file in code_files:
                print(f"- {file}")
            
            # Load marking scheme
            print("Loading marking scheme...")
            rubrics = self._load_marking_scheme(marking_scheme_path)
            
            # Read all code files
            code = "\n".join([
                open(os.path.join(extract_path, f), 'r', encoding='utf-8').read()
                for f in code_files
            ])
            
            # Prepare RAG context
            print("Preparing RAG context...")
            rag_context = self._prepare_rag_context(code, rubrics)
            
            # Grade submission using RAG
            print("Grading submission with RAG...")
            marks, feedback = self.grader.grade(
                extract_path=extract_path,
                code_files=code_files,
                language=language,
                rubrics=rubrics,
                code_run_output=code_run_output,
                rag_context=rag_context
            )
            
            # Scale marks to the specified total_marks
            scaled_marks = (marks / 100) * total_marks
            
            # Cleanup
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
            
            return {
                "marks": round(scaled_marks, 2),  # Round to 2 decimal places
                "feedback": feedback,
                "code_files": code_files,
                "rag_context_used": len(rag_context) > 0,
                "rag_context_count": len(rag_context),
                "total_marks": total_marks
            }
            
        except Exception as e:
            print(f"Error during grading: {str(e)}")
            import traceback
            print(f"Full error traceback: {traceback.format_exc()}")
            return {
                "error": str(e),
                "marks": 0,
                "feedback": f"Error during grading: {str(e)}"
            }

def list_lab_folders(base_dir: str) -> List[str]:
    """List all lab submission folders."""
    lab_folders = [
        f for f in os.listdir(base_dir) 
        if os.path.isdir(os.path.join(base_dir, f)) 
        and f.startswith("Lab") 
        and "zipped_submissions" in f
    ]
    return sorted(lab_folders)

def list_submissions(lab_dir: str) -> List[str]:
    """List all zip files in the lab directory."""
    if not os.path.exists(lab_dir):
        print(f"Lab directory not found: {lab_dir}")
        return []
    
    zip_files = [f for f in os.listdir(lab_dir) if f.lower().endswith('.zip')]
    return sorted(zip_files)

def list_marking_schemes(schemes_dir: str) -> List[str]:
    """List all marking scheme files in the schemes directory."""
    if not os.path.exists(schemes_dir):
        print(f"Creating marking schemes directory: {schemes_dir}")
        os.makedirs(schemes_dir)
    
    scheme_files = [f for f in os.listdir(schemes_dir) if f.lower().endswith(('.txt', '.md', '.pdf'))]
    return scheme_files

def save_results_to_csv(results: Dict[str, Any], output_file: str):
    """Save grading results to CSV file."""
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['student_id', 'marks', 'feedback'])
        
        for result in results['successful']:
            student_id = os.path.splitext(result['submission'])[0]
            writer.writerow([
                student_id,
                result['marks'],
                result['feedback']
            ])

def process_all_submissions(lab_dir: str, marking_scheme_path: str, total_marks: float, test_mode: bool = True, test_limit: int = 5) -> List[Dict]:
    """Process all submissions in a lab directory.
    
    Args:
        lab_dir (str): Path to the lab directory containing student submissions
        marking_scheme_path (str): Path to the marking scheme file
        total_marks (float): Total marks for the assignment
        test_mode (bool, optional): Whether to run in test mode. Defaults to True.
        test_limit (int, optional): Number of submissions to process in test mode. Defaults to 5.
        
    Returns:
        List[Dict]: List of grading results for each submission
    """
    submissions = list_submissions(lab_dir)
    if not submissions:
        return {"error": f"No submissions found in {lab_dir}"}
    
    if test_mode:
        print(f"\nTEST MODE: Processing only {test_limit} submissions")
        submissions = submissions[:test_limit]
    
    results = {
        "successful": [],
        "failed": [],
        "total_submissions": len(submissions),
        "test_mode": test_mode,
        "test_limit": test_limit if test_mode else None
    }
    
    grader = RAGGrader()
    
    # Add progress bar
    print(f"\nProcessing {len(submissions)} submissions...")
    for submission in tqdm(submissions, desc="Grading submissions"):
        try:
            submission_path = os.path.join(lab_dir, submission)
            
            result = grader.grade_submission(
                submission_zip=submission_path,
                marking_scheme_path=marking_scheme_path,
                total_marks=total_marks
            )
            
            if "error" in result:
                results["failed"].append({
                    "submission": submission,
                    "error": result["error"]
                })
            else:
                results["successful"].append({
                    "submission": submission,
                    "marks": result["marks"],
                    "feedback": result["feedback"],
                    "code_files": result["code_files"]
                })
                
        except Exception as e:
            results["failed"].append({
                "submission": submission,
                "error": str(e)
            })
    
    return results

def main():
    # Setup directories
    base_dir = os.path.join(os.path.dirname(__file__), "grading_data")
    schemes_dir = os.path.join(os.path.dirname(__file__), "marking_schemes")
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    
    # Create directories if they don't exist
    os.makedirs(schemes_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print("\n" + "="*50)
    print("RAG Grading System")
    print("="*50)
    print(f"\nBase directory: {base_dir}")
    print(f"Marking schemes directory: {schemes_dir}")
    print(f"Results directory: {results_dir}")
    
    try:
        # Get total marks from user
        while True:
            try:
                total_marks = float(input("\nEnter total marks for this assignment (e.g., 10.0): "))
                if total_marks > 0:
                    break
                print("Please enter a positive number.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Test mode parameters
        test_mode = False
        test_limit = 5  # Only used if test_mode is True
        
        # List available lab folders
        lab_folders = list_lab_folders(base_dir)
        if not lab_folders:
            print("\nNo lab folders found. Please ensure lab folders follow the pattern 'LabX_zipped_submissions'")
            return
        
        print("\nAvailable labs:")
        for i, lab in enumerate(lab_folders, 1):
            print(f"{i}. {lab}")
        
        # Select lab
        lab_choice = int(input("\nEnter the number of the lab to grade: ")) - 1
        if lab_choice < 0 or lab_choice >= len(lab_folders):
            raise ValueError("Invalid lab choice")
        
        lab_dir = os.path.join(base_dir, lab_folders[lab_choice])
        
        # List available marking schemes
        schemes = list_marking_schemes(schemes_dir)
        if not schemes:
            print("\nNo marking scheme files found. Please add .txt, .md, or .pdf files to the marking schemes directory.")
            print(f"Marking schemes directory: {schemes_dir}")
            return
        
        print("\nAvailable marking schemes:")
        for i, scheme in enumerate(schemes, 1):
            print(f"{i}. {scheme}")
        
        # Select marking scheme
        scheme_choice = int(input("\nEnter the number of the marking scheme to use: ")) - 1
        if scheme_choice < 0 or scheme_choice >= len(schemes):
            raise ValueError("Invalid marking scheme choice")
        
        marking_scheme = os.path.join(schemes_dir, schemes[scheme_choice])
        
        # Get language
        language = input("\nEnter programming language (python/java/c/cpp/js): ").lower()
        
        # Optional: Get test case output
        code_run_output = input("Enter test case output (press Enter to skip): ")
        
        print(f"\nProcessing submissions in {lab_folders[lab_choice]}...")
        results = process_all_submissions(
            lab_dir=lab_dir,
            marking_scheme_path=marking_scheme,
            total_marks=total_marks,
            test_mode=test_mode,
            test_limit=test_limit
        )
        
        print("\nBatch Processing Results:")
        print("=======================")
        print(f"Total submissions processed: {results['total_submissions']}")
        print(f"Successful: {len(results['successful'])}")
        print(f"Failed: {len(results['failed'])}")
        
        if results['failed']:
            print("\nFailed submissions:")
            for fail in results['failed']:
                print(f"- {fail['submission']}: {fail['error']}")
        
        # Save results to files in results directory
        output_prefix = f"grading_results_{lab_folders[lab_choice]}"
        if test_mode:
            output_prefix += "_test"
        
        # Save JSON results
        json_file = os.path.join(results_dir, f"{output_prefix}.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save CSV results
        csv_file = os.path.join(results_dir, f"{output_prefix}.csv")
        save_results_to_csv(results, csv_file)
        
        print(f"\nDetailed results saved to:")
        print(f"- JSON: {json_file}")
        print(f"- CSV: {csv_file}")
            
    except KeyboardInterrupt:
        print("\nGrading cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 