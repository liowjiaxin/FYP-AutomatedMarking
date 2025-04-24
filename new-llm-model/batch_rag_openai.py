import argparse
import os
import logging
import time
import random
import csv
import zipfile
from datetime import datetime
from tqdm import tqdm
import backoff
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

def get_base_dir():
    """Get the base directory of the script"""
    return os.path.dirname(os.path.abspath(__file__))

def parse_args():
    parser = argparse.ArgumentParser(description='Batch grade student submissions')
    parser.add_argument('--modelname', default='openai/gpt-3.5-turbo', help='Model name to use (openai/gpt-3.5-turbo or openai/gpt-4-turbo-preview)')
    parser.add_argument('--batch-size', type=int, default=1, help='Number of submissions to process in each batch')
    parser.add_argument('--delay-between-files', type=float, default=5.0, help='Delay between processing files (seconds)')
    parser.add_argument('--delay-between-batches', type=float, default=30.0, help='Delay between batches (seconds)')
    return parser.parse_args()

def validate_inputs():
    """Validate input directories and files"""
    # Get base directory
    base_dir = get_base_dir()
    
    # Define relative paths
    submissions_dir = os.path.join(base_dir, 'grading_data', 'Lab2_zipped_submissions')
    marking_scheme_path = os.path.join(base_dir, 'marking_schemes', 'marking_scheme_lab2.pdf')

    if not os.path.exists(submissions_dir):
        raise ValueError(f"Submissions directory does not exist: {submissions_dir}")
    if not os.path.exists(marking_scheme_path):
        raise ValueError(f"Marking scheme file does not exist: {marking_scheme_path}")

    # Get all zip files in the submissions directory
    submission_files = [f for f in os.listdir(submissions_dir) if f.endswith('.zip')]
    if not submission_files:
        raise ValueError(f"No zip files found in submissions directory: {submissions_dir}")

    # Extract lab number from directory name
    lab_number = os.path.basename(submissions_dir).split('_')[0]  # This will give us "Lab1"

    return submissions_dir, marking_scheme_path, submission_files, lab_number

def list_zip_contents(zip_path):
    """List all files in a zip file and identify C files"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            files = zip_ref.namelist()
            c_files = []
            logger.info(f"\nDetailed contents of {os.path.basename(zip_path)}:")
            
            # First pass: list all files with their paths
            for file in files:
                logger.info(f"  - {file}")
                if file.lower().endswith(('.c', '.h')):
                    c_files.append(file)
                    logger.info(f"    ✓ C file found: {file}")
            
            if not c_files:
                logger.warning("    No C files found in this submission")
            else:
                logger.info(f"    Found {len(c_files)} C files")
            
            return files, c_files
    except Exception as e:
        error_msg = f"Error reading zip file: {str(e)}"
        logger.error(error_msg)
        return error_msg, []

@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def process_submission_with_retry(grader, code_str, language, rubrics, num_files, zip_path):
    """Process a single submission with retry logic"""
    try:
        if not code_str:
            feedback = "No C code could be extracted from the submission"
            logger.warning(feedback)
            return {
                "marks": 0, 
                "feedback": feedback
            }
            
        lines_of_code = len(code_str.split('\n'))
        logger.info(f"Processing {lines_of_code} lines of code from {num_files} files")
        
        # Split code into smaller chunks if too large
        max_tokens = 500  # Reduced from 1000 to stay within limits
        if lines_of_code > max_tokens:
            logger.warning(f"Code exceeds {max_tokens} lines, processing in chunks")
            chunks = [code_str[i:i+max_tokens] for i in range(0, len(code_str), max_tokens)]
            results = []
            for chunk in chunks:
                result = grader.grade(
                    code=chunk,
                    language=language,
                    lines_of_code=len(chunk.split('\n')),
                    num_files=num_files,
                    rubrics=rubrics
                )
                results.append(result)
            # Combine results
            final_marks = sum(r['marks'] for r in results) / len(results)
            final_feedback = "\n".join(r['feedback'] for r in results)
            return {"marks": final_marks, "feedback": final_feedback}
        else:
            return grader.grade(
                code=code_str,
                language=language,
                lines_of_code=lines_of_code,
                num_files=num_files,
                rubrics=rubrics
            )
    except Exception as e:
        if "Insufficient credits" in str(e) or "token limit exceeded" in str(e):
            logger.error(f"API limit reached: {str(e)}")
            raise  # This will trigger the backoff retry
        logger.error(f"Error processing submission: {str(e)}")
        return {"marks": 0, "feedback": f"Error: {str(e)}"}

def extract_student_id(filename):
    """Extract student ID from filename"""
    # Assuming format is either "studentid.zip" or "studentid_name.zip"
    base_name = os.path.splitext(filename)[0]
    return base_name.split('_')[0]

def main():
    args = parse_args()
    logger.info("Arguments parsed successfully")
    
    try:
        submissions_dir, marking_scheme_path, submission_files, lab_number = validate_inputs()
        logger.info(f"Found {len(submission_files)} submissions to process")
        logger.info(f"Using {submissions_dir} as submissions directory")

        # Initialize grader
        try:
            logger.info("Initializing grader...")
            grader = GraderRAG(submissions_dir, args.modelname)
            logger.info("Grader initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing grader: {str(e)}")
            return

        # Create results file with lab number
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f'grading_results_{lab_number.lower()}_{timestamp}.csv'
        with open(results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Student ID', 'Marks', 'Feedback'])

        try:
            # Read marking scheme
            logger.info(f"Reading marking scheme: {marking_scheme_path}")
            rubrics = read_pdf(marking_scheme_path)
            logger.info("Marking scheme read successfully")

            # Process each submission with progress bar
            for submission_file in tqdm(submission_files, desc="Grading submissions"):
                submission_path = os.path.join(submissions_dir, submission_file)
                student_id = extract_student_id(submission_file)
                
                try:
                    # First check if the zip file exists and is readable
                    if not os.path.exists(submission_path):
                        error_msg = f"Submission file does not exist: {submission_path}"
                        logger.error(error_msg)
                        with open(results_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([student_id, 0, error_msg])
                        continue

                    # List zip contents before attempting to read code
                    zip_contents, c_files = list_zip_contents(submission_path)
                    
                    # Read code files
                    code_str, num_files = read_code(submission_path, "C")
                    
                    # Process submission with retry logic
                    result = process_submission_with_retry(grader, code_str, "C", rubrics, num_files, submission_path)
                    
                    # Save results
                    with open(results_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([student_id, 
                                       result['marks'], 
                                       result['feedback']])
                    
                    # Add delay between files if specified
                    if args.delay_between_files > 0:
                        time.sleep(args.delay_between_files)
                        
                except Exception as e:
                    error_msg = f"Error processing submission: {str(e)}"
                    logger.error(error_msg)
                    with open(results_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([student_id, 0, error_msg])

        except Exception as e:
            logger.error(f"Error in main process: {str(e)}")
            return

        logger.info(f"Grading completed. Results saved to {results_file}")

    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        return

if __name__ == "__main__":
    main() 