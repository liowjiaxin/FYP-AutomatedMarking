from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import os
import json
import re

GRADER_MODEL_NAME = os.environ.get("GRADER_MODEL_NAME", "gemini-1.5-flash")

grader_prompt = """
You are an expert programming assignment grader with 20 years of experience. Conduct a comprehensive evaluation of the submission using the multi-dimensional rubric below.

Submission Metadata
- Language: {language}
- Lines of Code: {lines_of_code}
- Number of Files: {num_files}
- Program Output: {code_run_output}
- Special Requirements or Notes: {rubrics}

Code Submission
{code}

## Retrieved Examples for Context
Use the examples below to **ensure grading consistency**, align with prior **grading tone**, and **guide formatting and phrasing** of your feedback. These represent previous submissions and their corresponding evaluations:
{similar_examples}

---
**EvaluationFramework**

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

**IMPORTANT GRADING PRINCIPLES**
1. If the submission fulfills ALL requirements specified in the marking scheme PDF, it MUST receive full marks (100%).
2. Only deduct marks if there are specific deficiencies or missing requirements.
3. The marking scheme PDF is the primary reference for grading criteria.
4. Additional deductions can only be made for issues not covered in the marking scheme if they significantly impact code quality or functionality.

**Scoring Guidelines**
- 100: Perfect implementation meeting ALL marking scheme requirements
- 90-99: Minor deviations from requirements with excellent implementation
- 80-89: Meets most requirements with some minor issues
- 70-79: Meets basic requirements but needs improvements
- 60-69: Partial implementation with significant gaps
- <60: Incomplete or non-functional submission

**Output Schema**
Return a JSON object with the following structure:
{{
  "marks": <0-100>,
  "feedback": str
}}

**Evaluation Process**
1. First verify if ALL marking scheme requirements are met
2. If all requirements are met, award full marks (100)
3. If not, identify specific missing requirements or deficiencies
4. Cross-validate program output with implementation
5. Compare against language best practices
6. Consider similar past submissions and their feedback
7. Generate scores using weighted rubric components

**Special Instructions**
- Award full marks (100) if ALL marking scheme requirements are met
- Only deduct marks for specific, documented deficiencies
- Use the marking scheme PDF as the primary grading reference
- Consider additional deductions only for significant issues not covered in the marking scheme
- Penalize security risks severely (-10-20 points)
- Reward innovative but appropriate solutions (+5-10 bonus)
- Flag academic integrity concerns explicitly
- Consider lines/file ratios in quality assessment
- Use retrieved examples to ensure consistent grading standards

Now evaluate this submission:
"""


class GradeFeedback(BaseModel):
    marks: int = Field(..., ge=0, le=100)
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
            # Format similar examples for the prompt
            if "similar_examples" in input:
                examples_text = "\n\n".join([
                    f"Example {i+1}:\nContent: {ex['content']}\nMetadata: {ex['metadata']}"
                    for i, ex in enumerate(input["similar_examples"])
                ])
                input["similar_examples"] = examples_text

            # Get response from LLM
            print("Invoking LLM chain...")
            response = self.chain.invoke(input)
            print(f"Raw LLM response: {response}")
            
            # Extract content from response
            if hasattr(response, 'content'):
                content = response.content
            elif hasattr(response, 'text'):
                content = response.text
            else:
                content = str(response)
            print(f"Extracted content: {content}")
            
            # Clean up the content
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            # Parse JSON from response
            try:
                # First try to parse as proper JSON
                result = json.loads(content)
            except json.JSONDecodeError:
                try:
                    # If that fails, try to parse as Python dictionary string
                    import ast
                    result = ast.literal_eval(content)
                except (ValueError, SyntaxError):
                    # If that fails, try to extract JSON from the content
                    json_match = re.search(r'\{[\s\S]*\}', content)
                    if json_match:
                        json_str = json_match.group(0)
                        try:
                            result = json.loads(json_str)
                        except json.JSONDecodeError:
                            try:
                                result = ast.literal_eval(json_str)
                            except (ValueError, SyntaxError):
                                raise ValueError("Could not parse response as JSON or Python dictionary")
                    else:
                        raise ValueError("No valid JSON or dictionary found in response")
            
            print(f"Parsed JSON: {result}")
            
            # Validate result
            if not isinstance(result, dict):
                raise ValueError("Response is not a dictionary")
            
            if 'marks' not in result or 'feedback' not in result:
                raise ValueError("Response missing required fields (marks, feedback)")
            
            # Ensure marks is an integer
            try:
                result['marks'] = int(result['marks'])
            except (ValueError, TypeError):
                raise ValueError("Marks must be a number")
            
            # Ensure feedback is a string
            if not isinstance(result['feedback'], str):
                result['feedback'] = str(result['feedback'])
                
            return result
        except Exception as e:
            print(f"Error invoking LLM: {str(e)}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            return {"marks": 0, "feedback": f"Error during grading: {str(e)}"}

    def generate_feedback(self, code, rubrics, similar_examples):
        # First select the most relevant stored feedback
        most_relevant = self.select_most_relevant_feedback(similar_examples)
        # Then have the LLM adapt it to the current submission
        adapted_feedback = self.llm.adapt_feedback(most_relevant, code, rubrics)
        return adapted_feedback


class GraderLLM(LLM):
    def __init__(self):
        model_name = GRADER_MODEL_NAME
        system_prompt = """
        **Role**: You are CodeGrader-ULTRA, an expert AI teaching assistant for programming courses.
        Your task is to rigorously evaluate student submissions by combining software engineering best practices with pedagogical expertise.
        Use the provided similar examples to ensure consistent grading standards and provide more contextual feedback.
        
        IMPORTANT: Return your response as a valid JSON object with 'marks' (number) and 'feedback' (string) fields.
        Do not include any markdown formatting around the JSON.
        Example response:
        {{
          "marks": 85,
          "feedback": "Your code demonstrates good understanding of the concepts..."
        }}
        """
        parser = JsonOutputParser(pydantic_object=GradeFeedback)
        super().__init__(model_name, grader_prompt, system_prompt, parser)