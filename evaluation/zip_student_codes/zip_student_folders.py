import os
import zipfile

# Get current directory and use absolute paths
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(current_dir, "Lab1_sourcecodes")  # Updated path
output_dir = os.path.join(current_dir, "Lab1_zipped_submissions")
os.makedirs(output_dir, exist_ok=True)

# Check if the directory exists
if not os.path.exists(base_dir):
    print(f"Error: Directory not found: {base_dir}")
    exit(1)

# Count total folders to process
total_folders = len([name for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))])
processed = 0

print(f"Found {total_folders} student folders to process...")

for student_id in os.listdir(base_dir):
    student_path = os.path.join(base_dir, student_id)
    if os.path.isdir(student_path):
        zip_filename = os.path.join(output_dir, f"{student_id}.zip")
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(student_path):
                for file in files:
                    full_path = os.path.join(root, file)
                    arcname = os.path.relpath(full_path, start=student_path)
                    zipf.write(full_path, arcname)
        processed += 1
        print(f"Zipped {processed}/{total_folders}: {student_id}.zip")

print(f"\n🎉 Successfully zipped {processed} student folders!")
