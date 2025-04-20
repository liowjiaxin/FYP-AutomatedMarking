from rag_grading import Grader
import os

def test_grading():
    # Initialize the grader
    grader = Grader()
    
    # Example submission details
    extract_path = "path/to/submission"  # Replace with actual path
    code_files = ["main.py"]  # Replace with actual code files
    language = "python"
    rubrics = """
    Grading Rubric:
    1. Code Quality (40%)
       - Proper indentation and formatting
       - Meaningful variable names
       - Appropriate comments
    2. Functionality (60%)
       - Correct implementation of requirements
       - Error handling
       - Input validation
    """
    
    # Example code run output (if available)
    code_run_output = """
    Test Case 1: Passed
    Test Case 2: Passed
    Test Case 3: Failed - Incorrect output
    """
    
    try:
        # Grade the submission
        marks, feedback = grader.grade(
            extract_path=extract_path,
            code_files=code_files,
            language=language,
            rubrics=rubrics,
            code_run_output=code_run_output
        )
        
        # Print results
        print("\nGrading Results:")
        print(f"Marks: {marks}/100")
        print("\nFeedback:")
        print(feedback)
        
    except Exception as e:
        print(f"Error during grading: {str(e)}")
        import traceback
        print(f"Full error traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    test_grading() 