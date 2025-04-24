import os
import sys
from PyPDF2 import PdfReader
import zipfile

# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from batch_rag_mark import RAGGrader

def test_single_submission():
    # Initialize the grader
    grader = RAGGrader()
    
    # Paths to files - now in the test directory
    base_dir = os.path.dirname(__file__)
    student_file = os.path.join(base_dir, "student_submission", "Task2_student.py")
    marking_scheme = os.path.join(base_dir, "marking_schemes", "Task2_marking_scheme.pdf")
    
    # Create zip file for submission
    submission_zip = os.path.join(base_dir, "student_submission", "Task2_student.zip")
    with zipfile.ZipFile(submission_zip, 'w') as zipf:
        zipf.write(student_file, os.path.basename(student_file))
    
    # Verify files exist
    if not os.path.exists(submission_zip):
        print(f"Error: Submission zip file not found at {submission_zip}")
        return
    if not os.path.exists(marking_scheme):
        print(f"Error: Marking scheme file not found at {marking_scheme}")
        return
    
    try:
        # Try to read the marking scheme first to catch any PDF reading errors
        print("Reading marking scheme...")
        reader = PdfReader(marking_scheme)
        text = ""
        for page in reader.pages:
            try:
                text += page.extract_text() + "\n"
            except Exception as e:
                print(f"Warning: Could not extract text from page: {str(e)}")
                continue
        
        if not text.strip():
            print("Warning: No text could be extracted from the marking scheme PDF")
        
        # Grade the submission
        print("\nGrading submission...")
        result = grader.grade_submission(
            submission_zip=submission_zip,
            marking_scheme_path=marking_scheme,
            language="python",
            total_marks=10.0  # Adjust this based on your marking scheme
        )
        
        # Print results
        print("\nGrading Results:")
        print("===============")
        print(f"Marks: {result['marks']}")
        print(f"Feedback: {result['feedback']}")
        print(f"Code files processed: {result['code_files']}")
        print(f"RAG context used: {result['rag_context_used']}")
        print(f"RAG context count: {result['rag_context_count']}")
        
    except Exception as e:
        print(f"Error during grading: {str(e)}")
        import traceback
        print(f"Full error traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    test_single_submission() 