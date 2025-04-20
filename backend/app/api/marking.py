from fastapi import APIRouter, UploadFile, File, Form
from pathlib import Path
import shutil
import os
from ..services.marking_service import MarkingService
from ..config import settings

router = APIRouter()
marking_service = MarkingService(index_dir=settings.INDEX_DIR)

@router.post("/process-submission")
async def process_submission(
    student_id: str = Form(...),
    marking_scheme: UploadFile = File(...),
    student_answer: UploadFile = File(...),
    stdin_file: UploadFile = File(None)
):
    """Process a student submission and return the results."""
    # Create temporary directory for uploaded files
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Save uploaded files
        marking_scheme_path = temp_dir / marking_scheme.filename
        student_answer_path = temp_dir / student_answer.filename
        stdin_path = temp_dir / stdin_file.filename if stdin_file else None
        
        with open(marking_scheme_path, "wb") as buffer:
            shutil.copyfileobj(marking_scheme.file, buffer)
            
        with open(student_answer_path, "wb") as buffer:
            shutil.copyfileobj(student_answer.file, buffer)
            
        if stdin_file:
            with open(stdin_path, "wb") as buffer:
                shutil.copyfileobj(stdin_file.file, buffer)
        
        # Process the submission
        result = await marking_service.process_submission(
            student_id=student_id,
            code_path=str(student_answer_path),
            marking_scheme_path=str(marking_scheme_path),
            stdin_path=str(stdin_path) if stdin_path else None
        )
        
        return result
        
    finally:
        # Clean up temporary files
        shutil.rmtree(temp_dir, ignore_errors=True) 