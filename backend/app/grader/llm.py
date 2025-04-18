from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import os
import json

GRADER_MODEL_NAME = os.environ.get("GRADER_MODEL_NAME", "gemini-1.5-flash")

grader_prompt = """
You are an expert programming assignment grader with 20 years of experience. Conduct a comprehensive evaluation of the submission using the multi-dimensional rubric below.

**Submission Analysis Framework**

1. **Code Comprehension** (10%)
- Does the code demonstrate understanding of core concepts?
- Is there evidence of conceptual misunderstandings?

2. **Functional Correctness** (30%)
- Does the code solve the problem as specified?
- Test case coverage (basic, edge, stress cases)
- Error handling and robustness

3. **Code Quality** (25%)
- [Comments] Clarity, relevance, and density (aim for 20-30% comment ratio)
- [Modularity] Logical decomposition, function length (<30 lines), DRY principle
- [Readability] Naming conventions, spacing, structural organization

4. **Technical Implementation** (25%)
- Algorithm efficiency (time/space complexity)
- Language feature appropriateness
- Resource management (memory, files, connections)

5. **Testing & Verification** (10%)
- Test existence and quality (if required)
- Input validation
- Debugging evidence

**Evaluation Context**
- Submission Language: {language}
- Lines of Code: {lines_of_code}
- Number of Files: {num_files}
- Program Output: {code_run_output}
- Special Requirements: {rubrics}

**Analysis Protocol**
1. **Structural Scan**: Perform initial code structure evaluation
2. **Semantic Analysis**: Cross-reference requirements with implementation
3. **Defect Identification**: List technical and stylistic issues
4. **Strength Recognition**: Highlight exemplary practices
5. **Improvement Roadmap**: Create prioritized suggestions

**Scoring Guidelines**
- 9-10: Flawless implementation exceeding requirements
- 8-8.9: Minor issues with excellent fundamentals
- 7-7.9: Functional but needs quality improvements
- 6-6.9: Partial solution with significant gaps
- <6: Non-working or severely deficient

**Output Schema**
{
  "marks": <0-10>,
  "feedback": str
}

Please return raw JSON only. Do not wrap it in a code block or use Markdown formatting.


**Submission Code**
{code}

**Evaluation Process**
1. Analyze the code through multiple passes
2. Cross-validate program output with implementation
3. Compare against language best practices
4. Generate scores using weighted rubric components
5. Maintain strict but fair academic standards

**Special Instructions**
- Penalize security risks severely (-1-2 points)
- Reward innovative but appropriate solutions (+0.5-1 bonus)
- Flag academic integrity concerns explicitly
- Consider lines/file ratios in quality assessment
"""


class GradeFeedback(BaseModel):
    marks: str
    feedback: str


class LLM:
    def __init__(
        self,
        model_name: str,
        prompt_template: str,
        system_prompt: str,
        parser: JsonOutputParser | None = None,
    ):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        self.unified_prompt_template = prompt_template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", self.unified_prompt_template),
        ])
        if parser:
            self.parser = parser
            self.chain = self.prompt_template | self.llm | self.parser
        else:
            self.chain = self.prompt_template | self.llm

    def invoke(self, input: dict):
    try:
        result = self.chain.invoke(input)

        # Case 1: If result is a dictionary (parsed by LangChain's JsonOutputParser)
        if isinstance(result, dict):
            return result

        # Case 2: If result is a string (fallback when parser not used or fails)
        if isinstance(result, str):
            # Clean code block wrappers
            cleaned = (
                result.replace("```json", "")
                      .replace("```", "")
                      .replace("json", "")
                      .strip()
            )
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError as json_err:
                print("JSON decode failed even after cleaning:", json_err)
                return {
                    "marks": 0,
                    "feedback": f"Error during grading: Invalid JSON after code block cleanup. Raw:\n{result}"
                }

        # Fallback for unexpected types
        return {
            "marks": 0,
            "feedback": f"Error during grading: Unexpected LLM output type: {type(result)}"
        }

    except Exception as e:
        print(f"Exception while invoking LLM: {str(e)}")
        return {
            "marks": 0,
            "feedback": f"Error during grading: {str(e)}"
        }


class GraderLLM(LLM):
    def __init__(self):
        model_name = GRADER_MODEL_NAME
        system_prompt = """
        **Role**: You are CodeGrader-ULTRA, an expert AI teaching assistant for programming courses.
        Your task is to rigorously evaluate student submissions by combining software engineering best practices with pedagogical expertise.
        """
        parser = JsonOutputParser(pydantic_object=GradeFeedback)
        super().__init__(model_name, grader_prompt, system_prompt, parser)
