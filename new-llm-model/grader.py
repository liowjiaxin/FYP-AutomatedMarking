from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from load_dataset import DatasetLoader
from typing import Literal, List, Dict, Any
import os
import logging
from openai import OpenAI
from dotenv import load_dotenv
import requests

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default to GPT-3.5 Turbo which is more cost-effective
GRADER_MODEL_NAME = "openai/gpt-3.5-turbo"
VECTOR_STORE_PATH = "c_dataset_faiss"  # Directory for FAISS index

# List of supported models with their token limits
SUPPORTED_MODELS = {
    "openai/gpt-3.5-turbo": 1000,  # More cost-effective
    "openai/gpt-4-turbo-preview": 2000
}

system_prompt = """
**Role**: You are CodeGrader-ULTRA, an expert AI teaching assistant for programming courses.
Your task is to rigorously evaluate student submissions by combining software engineering best practices with pedagogical expertise.

**Core Principles**:
1. Be precise and consistent in grading
2. Provide constructive, actionable feedback
3. Consider both technical correctness and educational value
4. Align strictly with the provided marking scheme
5. Maintain academic integrity standards
"""

grader_prompt = """
You are an expert programming assignment grader with 20 years of experience. Your task is to evaluate the submission according to the provided marking scheme.

**Marking Scheme Analysis**
First, carefully analyze the provided marking scheme:
{rubrics}

**Submission Context**
- Language: {language}
- Lines of Code: {lines_of_code}
- Number of Files: {num_files}

**Evaluation Framework**
1. **Code Comprehension** (10%)
- Alignment with marking scheme requirements
- Understanding of core concepts
- Conceptual clarity and accuracy

2. **Functional Correctness** (30%)
- Implementation of specified requirements
- Test case coverage and validation
- Error handling and edge cases
- Output correctness

3. **Code Quality** (25%)
- Code organization and structure
- Naming conventions and readability
- Documentation and comments
- Code reusability and modularity

4. **Technical Implementation** (25%)
- Algorithm efficiency
- Appropriate use of language features
- Resource management
- Security considerations

5. **Testing & Verification** (10%)
- Test coverage
- Input validation
- Error handling
- Debugging approach

**Grading Process**
1. **Initial Assessment**
   - Review marking scheme requirements
   - Identify key evaluation criteria
   - Note specific grading points

2. **Code Analysis**
   - Check requirement implementation
   - Evaluate code structure
   - Assess technical quality

3. **Scoring**
   - Apply marking scheme criteria
   - Assign partial marks where appropriate
   - Consider both correctness and quality

4. **Feedback Generation**
   - Highlight strengths
   - Identify areas for improvement
   - Provide specific examples
   - Suggest concrete improvements

**Scoring Guidelines**
- 90-100: Exceeds requirements, excellent implementation
- 80-89: Meets all requirements, minor improvements needed
- 70-79: Meets most requirements, some issues present
- 60-69: Partial implementation, significant gaps
- <60: Incomplete or non-functional submission

**Submission Code**
{code}

**Similar Examples for Context**
{similar_examples}

**IMPORTANT: Response Format**
You MUST respond with a valid JSON object in the following format:
{{
  "marks": <number between 0 and 100>,
  "feedback": {{
    "strengths": ["list", "of", "strong", "points"],
    "improvements": ["list", "of", "areas", "for", "improvement"],
    "specific_feedback": "detailed feedback aligned with marking scheme"
  }}
}}

**Evaluation Notes**
1. Be specific in feedback, referencing code sections
2. Consider both technical and educational aspects
3. Maintain consistency with marking scheme
4. Provide actionable improvement suggestions
5. Ensure your response is valid JSON
"""


class GradeFeedback(BaseModel):
    marks: float
    feedback: Dict[str, Any]  # Changed from str to Dict to handle structured feedback

    class Config:
        arbitrary_types_allowed = True


class GradingOutput(BaseModel):
    code_comprehension: float  # 10% of 15 = 1.5 marks
    functional_correctness: float  # 30% of 15 = 4.5 marks
    code_quality: float  # 25% of 15 = 3.75 marks
    technical_implementation: float  # 25% of 15 = 3.75 marks
    testing_verification: float  # 10% of 15 = 1.5 marks
    feedback: Dict[str, Any]  # Changed from str to Dict
    total_mark: float  # Will be scaled to 15

    def calculate_total(self):
        # Scale each component to 15% total
        self.total_mark = (
            (self.code_comprehension * 0.015) +  # 10% of 15
            (self.functional_correctness * 0.045) +  # 30% of 15
            (self.code_quality * 0.0375) +  # 25% of 15
            (self.technical_implementation * 0.0375) +  # 25% of 15
            (self.testing_verification * 0.015)  # 10% of 15
        )
        return self.total_mark


class GraderRAG:
    def __init__(self, dataset_path: str, model_type: str = GRADER_MODEL_NAME):
        if model_type not in SUPPORTED_MODELS:
            logger.warning(f"Model {model_type} not in supported models list. Using default: {GRADER_MODEL_NAME}")
            model_type = GRADER_MODEL_NAME
            
        logger.info(f"Initializing GraderRAG with model_type={model_type}")
        self.model_type = model_type
        self.max_tokens = SUPPORTED_MODELS[model_type]
        self.dataset_path = dataset_path
        
        # Initialize embeddings (using existing Gemini embeddings)
        logger.info("Initializing embeddings model...")
        self.embedder = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001"
        )
        logger.info("Embeddings model initialized")
        
        # Load or create vector store
        logger.info("Loading vector store...")
        self.vector_store = self._load_or_create_vector_store()
        
        # Initialize OpenRouter client
        logger.info("Initializing OpenRouter client...")
        if not os.getenv("OPENROUTER_API_KEY"):
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        
        # Create grader chain
        logger.info("Creating grader chain...")
        self.grader = self._create_grader_chain()
        logger.info("Grader chain created")

    def _prepare_rag_context(self, code: str, rubrics: str) -> List[Dict[str, Any]]:
        """Prepare RAG context by retrieving similar submissions."""
        try:
            # Create a query combining code and rubrics
            MAX_QUERY_CHARS = 15000
            query = f"Code: {code[:MAX_QUERY_CHARS // 2]}\nRubrics: {rubrics[:MAX_QUERY_CHARS // 2]}"

            # Retrieve similar submissions
            logger.info("Retrieving similar submissions for RAG context...")
            similar_docs = self.vector_store.similarity_search(query, k=3)
            
            # Format the retrieved documents
            context = []
            for doc in similar_docs:
                context.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "type": doc.metadata.get("type", "unknown"),
                    "feedback": doc.metadata.get("feedback", "No feedback available")
                })
            
            logger.info(f"Retrieved {len(context)} similar submissions for RAG context")
            return context
        except Exception as e:
            logger.error(f"Error preparing RAG context: {str(e)}")
            return []

    def _load_or_create_vector_store(self) -> FAISS:
        try:
            if os.path.exists(VECTOR_STORE_PATH):
                logger.info(f"Loading vector store from {VECTOR_STORE_PATH}")
                return FAISS.load_local(VECTOR_STORE_PATH, self.embedder, allow_dangerous_deserialization=True)
            else:
                logger.info(f"Creating vector store from {self.dataset_path}")
                vector_store = self._initialize_vector_store(self.dataset_path)
                self._save_vector_store(vector_store)
                return vector_store
        except Exception as e:
            logger.warning(f"Failed to load vector store, creating new one: {str(e)}")
            vector_store = self._initialize_vector_store(self.dataset_path)
            self._save_vector_store(vector_store)
            return vector_store

    def _initialize_vector_store(self, dataset_path: str) -> FAISS:
        # Load and preprocess dataset
        examples = self._load_examples(dataset_path)
        return FAISS.from_documents(examples, self.embedder)

    def _save_vector_store(self, vector_store: FAISS):
        vector_store.save_local(VECTOR_STORE_PATH)

    def _load_examples(self, path: str) -> list:
        loader = DatasetLoader(path)
        return loader.load()

    def _create_grader_chain(self):
        return self._initialize_llm

    def _initialize_llm(self, prompt: str):
        """Initialize the LLM based on selected model type"""
        try:
            logger.info(f"Making API call to OpenRouter with model: {self.model_type}")
            
            # Verify API key is set
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY environment variable not set")
            
            # Prepare the request data
            data = {
                "model": self.model_type,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": self.max_tokens
            }
            
            # Make the API call using requests
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": os.getenv("SITE_URL", "http://localhost"),
                    "X-Title": os.getenv("SITE_NAME", "CodeGrader")
                },
                json=data
            )
            
            # Check if the request was successful
            response.raise_for_status()
            
            # Parse the response
            result = response.json()
            
            if 'error' in result:
                error_msg = result['error'].get('message', 'Unknown error')
                if 'credits' in error_msg.lower() or 'tokens' in error_msg.lower():
                    logger.error(f"Credit/token limit error: {error_msg}")
                    raise ValueError(f"Insufficient credits or token limit exceeded: {error_msg}")
                raise ValueError(f"API Error: {error_msg}")
            
            if 'choices' not in result or not result['choices']:
                logger.error(f"Unexpected response structure: {result}")
                raise ValueError("No choices in API response")
                
            message = result['choices'][0]['message']
            if 'content' not in message:
                logger.error(f"Unexpected message structure: {message}")
                raise ValueError("No content in message")
                
            return message['content']
            
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP error calling OpenRouter API: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error calling OpenRouter API: {str(e)}")
            logger.error(f"API Key present: {bool(os.getenv('OPENROUTER_API_KEY'))}")
            logger.error(f"Model type: {self.model_type}")
            raise

    def grade(self, code: str, language: str = "Python", lines_of_code: int = 0, 
              num_files: int = 1, rubrics: str = "", total_marks: float = 10.0) -> Dict[str, Any]:
        """Grade the submission using RAG-enhanced LLM"""
        try:
            # Prepare RAG context
            rag_context = self._prepare_rag_context(code, rubrics)
            
            # Format similar examples for the prompt
            similar_examples = []
            for doc in rag_context:
                similar_examples.append(
                    f"Example from {doc['source']} ({doc['type']}):\n"
                    f"{doc['content']}\n"
                    f"Feedback: {doc['feedback']}\n"
                )
            
            # Prepare the prompt
            prompt = grader_prompt.format(
                language=language,
                lines_of_code=lines_of_code,
                num_files=num_files,
                rubrics=rubrics,
                code=code,
                similar_examples="\n\n".join(similar_examples)
            )
            
            # Get grading result
            result = self._initialize_llm(prompt)
            
            # Parse the result
            try:
                import json
                # Clean the response if needed
                result = result.strip()
                if not result.startswith('{'):
                    result = '{' + result
                if not result.endswith('}'):
                    result = result + '}'
                
                parsed_result = json.loads(result)
                
                # Validate the parsed result
                if not isinstance(parsed_result, dict):
                    raise ValueError("Invalid response format: not a dictionary")
                
                if 'marks' not in parsed_result or 'feedback' not in parsed_result:
                    raise ValueError("Missing required fields in response")
                
                if not isinstance(parsed_result['feedback'], dict):
                    raise ValueError("Feedback must be a dictionary")
                
                # Ensure feedback has required fields
                feedback = parsed_result['feedback']
                if 'strengths' not in feedback:
                    feedback['strengths'] = []
                if 'improvements' not in feedback:
                    feedback['improvements'] = []
                if 'specific_feedback' not in feedback:
                    feedback['specific_feedback'] = "No specific feedback provided"
                
                # Scale marks to total_marks
                scaled_marks = (float(parsed_result['marks']) / 100) * total_marks
                
                return {
                    "marks": round(scaled_marks, 2),
                    "feedback": feedback,
                    "rag_context_used": len(rag_context) > 0,
                    "rag_context_count": len(rag_context),
                    "total_marks": total_marks,
                    "similar_examples": [doc['source'] for doc in rag_context]
                }
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM output as JSON: {str(e)}")
                logger.error(f"Raw response: {result}")
                return {
                    "marks": 0,
                    "feedback": {
                        "strengths": [],
                        "improvements": ["Error: Could not parse grading result"],
                        "specific_feedback": "Technical error in processing the grading result"
                    },
                    "rag_context_used": False,
                    "rag_context_count": 0,
                    "total_marks": total_marks,
                    "similar_examples": []
                }
                
        except Exception as e:
            logger.error(f"Error during grading: {str(e)}")
            return {
                "marks": 0,
                "feedback": {
                    "strengths": [],
                    "improvements": [f"Error during grading: {str(e)}"],
                    "specific_feedback": "An error occurred during the grading process"
                },
                "rag_context_used": False,
                "rag_context_count": 0,
                "total_marks": total_marks,
                "similar_examples": []
            } 