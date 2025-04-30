import sys
import os

# Add the backend directory to Python path
backend_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(backend_path)

from app.rag.rag_grading import Grader

def test_grading():
    # Initialize the grader
    grader = Grader()
    
    # Example code and rubrics
    code = """
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
"""
    
    rubrics = """
Grading Rubric:
1. Functionality (40 points)
   - Correct calculation of factorial
   - Proper handling of edge cases
2. Code Quality (30 points)
   - Clear variable names
   - Proper comments
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
            code_files=["test.py"],
            language="python",
            rubrics=rubrics,
            code_run_output="120\n1\nError: Factorial is not defined for negative numbers"
        )
        
        # Print results
        print("\nGrading Results:")
        print("===============")
        print(f"Marks: {marks}")
        print(f"Feedback: {feedback}")
        
    except Exception as e:
        print(f"Error during grading: {str(e)}")
        import traceback
        print(f"Full error traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    test_grading() 