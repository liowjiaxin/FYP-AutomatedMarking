import argparse
import os
import logging
import time
import random
import csv
from datetime import datetime
from helper import read_code, read_pdf
from grader import GraderRAG

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('grading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Batch grade student submissions')
    parser.add_argument('--modelname', default='google/gemini-2.5-pro-exp-03-25:free', help='Model name to use')
    parser.add_argument('--batch-size', type=int, default=1, help='Number of submissions to process in each batch')
    parser.add_argument('--delay-between-files', type=float, default=5.0, help='Delay between processing files (seconds)')
    parser.add_argument('--delay-between-batches', type=float, default=30.0, help='Delay between batches (seconds)')
    return parser.parse_args()

def validate_inputs():
    """Validate input directories and files"""
    test_dir = os.path.join(os.path.dirname(__file__), 'test')
    student_dir = os.path.join(test_dir, 'student_submission')
    marking_dir = os.path.join(test_dir, 'marking_schemes')

    # Specific files to process
    student_file = "Task1_student.zip"
    marking_file = "Task1_marking_scheme.pdf"

    student_path = os.path.join(student_dir, student_file)
    marking_path = os.path.join(marking_dir, marking_file)

    if not os.path.exists(student_path):
        raise ValueError(f"Student submission file does not exist: {student_path}")
    if not os.path.exists(marking_path):
        raise ValueError(f"Marking scheme file does not exist: {marking_path}")

    # Use student submission directory as source code directory
    return student_path, marking_path, student_dir

def process_submission(grader, code_str, language, rubrics, num_files):
    """Process a single submission"""
    try:
        lines_of_code = len(code_str.split('\n'))
        result = grader.grade(
            code=code_str,
            language=language,
            lines_of_code=lines_of_code,
            num_files=num_files,
            rubrics=rubrics
        )
        return result
    except Exception as e:
        logger.error(f"Error processing submission: {str(e)}")
        return {"marks": 0, "feedback": f"Error: {str(e)}"}

def main():
    args = parse_args()
    logger.info("Arguments parsed successfully")
    
    try:
        student_path, marking_path, source_dir = validate_inputs()
        logger.info("Input validation passed")
        logger.info(f"Using {source_dir} as source code directory")

        # Initialize grader
        try:
            logger.info("Initializing grader...")
            grader = GraderRAG(source_dir, args.modelname)
            logger.info("Grader initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing grader: {str(e)}")
            return

        # Create results file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f'grading_results_{timestamp}.csv'
        with open(results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Submission', 'Marking Scheme', 'Marks', 'Feedback'])

        try:
            # Read marking scheme
            logger.info(f"Reading marking scheme: {marking_path}")
            rubrics = read_pdf(marking_path)
            logger.info("Marking scheme read successfully")

            # Process submission
            logger.info(f"Processing submission: {student_path}")
            
            # Read code files
            code_str, num_files = read_code(student_path, "Python")
            
            # Process submission
            result = process_submission(grader, code_str, "Python", rubrics, num_files)
            
            # Save results
            with open(results_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([os.path.basename(student_path), 
                               os.path.basename(marking_path), 
                               result['marks'], 
                               result['feedback']])
            
            logger.info(f"Completed grading for {student_path}")
            
        except Exception as e:
            logger.error(f"Error processing submission: {str(e)}")
            with open(results_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([os.path.basename(student_path), 
                               os.path.basename(marking_path), 
                               0, 
                               f"Error: {str(e)}"])

        logger.info(f"Grading completed. Results saved to {results_file}")

    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        return

if __name__ == "__main__":
    main() 