from rag.rag_grading import Grader
import os
from datetime import datetime

def test_rag():
    # Initialize the grader with RAG
    vector_store_path = os.path.join(os.path.dirname(__file__), "vector_store")
    os.makedirs(vector_store_path, exist_ok=True)
    grader = Grader(index_dir=vector_store_path)
    
    # Example code submission
    code_path = "test_submission.py"
    code_files = [code_path]
    
    # Create a test submission file
    with open(code_path, "w") as f:
        f.write("""
def calculate_factorial(n):
    if n < 0:
        return "Error: Factorial is not defined for negative numbers"
    elif n == 0 or n == 1:
        return 1
    else:
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

# Test cases
print(calculate_factorial(5))  # Should print 120
print(calculate_factorial(0))  # Should print 1
print(calculate_factorial(-1)) # Should print error message
""")
    
    # Example rubrics
    rubrics = """
    Grading Rubric:
    1. Functionality (40 points)
       - Correct calculation of factorial
       - Proper handling of edge cases (0, 1, negative numbers)
    2. Code Quality (30 points)
       - Clear variable names
       - Proper comments
       - Efficient implementation
    3. Error Handling (20 points)
       - Proper input validation
       - Clear error messages
    4. Testing (10 points)
       - Test cases for different scenarios
    """
    
    try:
        # Grade the submission
        marks, feedback = grader.grade(
            extract_path=".",
            code_files=code_files,
            language="python",
            rubrics=rubrics,
            code_run_output="120\n1\nError: Factorial is not defined for negative numbers"
        )
        
        print(f"\nMarks: {marks}")
        print(f"\nFeedback:\n{feedback}")
        
        # After successful grading, store the submission
        grader.retriever.add_documents([{
            "content": code_files[0],
            "metadata": {
                "student_id": "test_student",  # You'll need to pass this as a parameter
                "grade": marks,
                "feedback": feedback,
                "language": "python",
                "date": datetime.now().strftime("%Y-%m-%d")
            }
        }])
        
        return marks, feedback
    except Exception as e:
        print(f"Error in grading: {str(e)}")
        return 0, f"An error occurred during grading: {str(e)}"
    finally:
        # Clean up
        if os.path.exists(code_path):
            os.remove(code_path)

if __name__ == "__main__":
    test_rag() 