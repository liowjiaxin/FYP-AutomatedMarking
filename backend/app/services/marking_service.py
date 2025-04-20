from typing import Dict, Any, Optional
from pathlib import Path
from ..rag.retriever import Retriever
from ..rag.document_loader import DocumentLoader
import json

class MarkingService:
    def __init__(self, index_dir: str):
        """Initialize the marking service with RAG components."""
        self.retriever = Retriever(index_dir=index_dir)
        self.document_loader = DocumentLoader()

    async def process_submission(
        self,
        student_id: str,
        code_path: str,
        marking_scheme_path: str,
        stdin_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process a student submission and generate feedback.
        
        Args:
            student_id: ID of the student
            code_path: Path to the student's code file
            marking_scheme_path: Path to the marking scheme
            stdin_path: Optional path to stdin file
            
        Returns:
            Dictionary containing marks, grade, and feedback
        """
        # Load the submission and marking scheme
        submission = self.document_loader.load_submission(code_path)
        marking_scheme = self.document_loader.load_marking_scheme(marking_scheme_path)
        
        # Create a query combining the code and marking scheme
        query = f"Code: {submission['content']}\nMarking Scheme: {marking_scheme['content']}"
        
        # Retrieve relevant past submissions and feedback
        relevant_docs = self.retriever.retrieve_relevant_documents(query, k=3)
        
        # Generate feedback based on retrieved documents
        feedback = self._generate_feedback(
            submission=submission,
            marking_scheme=marking_scheme,
            relevant_docs=relevant_docs
        )
        
        # Calculate marks and grade
        marks, grade = self._calculate_marks(
            submission=submission,
            marking_scheme=marking_scheme,
            relevant_docs=relevant_docs
        )
        
        return {
            "student_id": student_id,
            "marks": marks,
            "grade": grade,
            "feedback": feedback
        }

    def _generate_feedback(
        self,
        submission: Dict[str, Any],
        marking_scheme: Dict[str, Any],
        relevant_docs: list
    ) -> str:
        """Generate feedback based on the submission and relevant documents."""
        # TODO: Implement feedback generation using LLM
        # For now, return a simple feedback based on retrieved documents
        feedback_parts = []
        
        for doc in relevant_docs:
            if doc.get("feedback"):
                feedback_parts.append(doc["feedback"])
        
        return "\n\n".join(feedback_parts) if feedback_parts else "No feedback available."

    def _calculate_marks(
        self,
        submission: Dict[str, Any],
        marking_scheme: Dict[str, Any],
        relevant_docs: list
    ) -> tuple[float, str]:
        """Calculate marks and grade based on the submission and marking scheme."""
        # TODO: Implement mark calculation using LLM
        # For now, return placeholder values
        return 75.0, "B" 