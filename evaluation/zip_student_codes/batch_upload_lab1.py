import os
import json
import time
import requests
import csv
from tqdm import tqdm
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8080"
UPLOAD_URL = f"{BASE_URL}/api/upload"
GRADE_URL = f"{BASE_URL}/api/grade"
MARKING_SCHEME_JSON = os.path.join("..", "batch_marking", "marking_scheme.json")
DELAY_BETWEEN_UPLOADS = 1
TEST_MODE = False
MAX_MARKS = 10
TEST_LIMIT = 5
OUTPUT_PREFIX = "evaluation_result_lab1"

def extract_student_id(zip_filename):
    return os.path.splitext(zip_filename)[0]

def smart_scale(marks):
    """Return marks as-is if <= 10, else scale from 100 to 10."""
    try:
        marks = float(marks)
        return round((marks / 100) * 10, 1) if marks > 10 else round(marks, 1)
    except (ValueError, TypeError):
        return 0.0

def process_submission(zip_path, marking_scheme_path):
    try:
        with open(zip_path, 'rb') as zip_file, open(marking_scheme_path, 'rb') as json_file:
            files = {
                'file': ('submission.zip', zip_file, 'application/zip'),
                'marking_scheme': ('marking_scheme.json', json_file, 'application/json')
            }
            data = {
                'student_id': extract_student_id(os.path.basename(zip_path)),
                'language': 'C'
            }

            print(f"\nUploading submission for student {data['student_id']}...")
            upload_response = requests.post(UPLOAD_URL, files=files, data=data)
            upload_response.raise_for_status()
            upload_data = upload_response.json()

            if 'filename' not in upload_data:
                return False, "Invalid upload response: missing filename"

            print(f"Upload successful: {upload_data.get('info')}")

            grade_data = {
                "student_id": data['student_id'],
                "code_zip_filename": upload_data['filename'],
                "marking_scheme_filename": os.path.basename(marking_scheme_path),
                "language": "C",
                "max_marks": MAX_MARKS
            }

            print(f"Requesting grading for student {data['student_id']}...")
            grade_response = requests.post(GRADE_URL, json=grade_data)
            grade_response.raise_for_status()

            grade_result = grade_response.json()

            if 'marks' not in grade_result or 'feedback' not in grade_result:
                return False, "Invalid grade result: missing marks or feedback"

            print(f"Grading complete for student {data['student_id']}")
            print("Grade Response:", json.dumps(grade_result, indent=2))

            return True, {
                'upload_response': upload_data,
                'grade_result': grade_result
            }

    except requests.exceptions.RequestException as e:
        error_msg = f"API Error: {str(e)}"
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_msg += f"\nResponse: {e.response.text}"
            except:
                pass
        return False, error_msg
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"

def save_results_to_csv(results, output_file=f'{OUTPUT_PREFIX}.csv'):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['student_id', 'marks', 'feedback', 'error']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for result in results['successful']:
            student_id = extract_student_id(os.path.basename(result['file']))
            grade_result = result.get('response', {}).get('grade_result', {})
            marks = smart_scale(grade_result.get('marks', 0))

            writer.writerow({
                'student_id': student_id,
                'marks': marks,
                'feedback': grade_result.get('feedback', 'N/A'),
                'error': 'N/A'
            })

        for result in results['failed']:
            student_id = extract_student_id(os.path.basename(result['file']))
            writer.writerow({
                'student_id': student_id,
                'marks': 'N/A',
                'feedback': 'N/A',
                'error': result['error']
            })

def main():
    if not os.path.exists(MARKING_SCHEME_JSON):
        print(f"Error: Marking scheme JSON not found at {MARKING_SCHEME_JSON}")
        return

    submissions_dir = "Lab2_zipped_submissions"
    zip_files = list(Path(submissions_dir).glob('*.zip'))

    if not zip_files:
        print(f"No zip files found in {submissions_dir} folder!")
        return

    if TEST_MODE:
        zip_files = zip_files[:TEST_LIMIT]
        print(f"\nTEST MODE: Processing only {TEST_LIMIT} submissions")

    print(f"Found {len(zip_files)} student submissions to process")
    print(f"Using marking scheme: {MARKING_SCHEME_JSON}")
    print(f"Upload API: {UPLOAD_URL}")
    print(f"Grade API: {GRADE_URL}")

    results = {'successful': [], 'failed': []}

    for zip_path in tqdm(zip_files, desc="Processing submissions"):
        success, response = process_submission(zip_path, MARKING_SCHEME_JSON)
        if success:
            results['successful'].append({'file': str(zip_path), 'response': response})
        else:
            print(f"\nFailed to process {os.path.basename(zip_path)}: {response}")
            results['failed'].append({'file': str(zip_path), 'error': response})

        time.sleep(DELAY_BETWEEN_UPLOADS)

    with open(f'{OUTPUT_PREFIX}.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    save_results_to_csv(results)

    print(f"\nBatch processing complete!")
    print(f"Total submissions processed: {len(zip_files)}")
    print(f"Successful: {len(results['successful'])}")
    print(f"Failed: {len(results['failed'])}")
    print(f"Results saved to {OUTPUT_PREFIX}.json and {OUTPUT_PREFIX}.csv")

    if results['successful']:
        print("\nMarks Progression (out of 10):")
        successful_results = sorted(
            results['successful'],
            key=lambda x: x['response'].get('grade_result', {}).get('marks', 0)
        )
        for result in successful_results:
            student_id = extract_student_id(os.path.basename(result['file']))
            marks = smart_scale(result['response'].get('grade_result', {}).get('marks', 0))
            print(f"Student {student_id}: {marks}/10")

if __name__ == "__main__":
    main()
