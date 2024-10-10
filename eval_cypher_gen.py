import os
import logging
from typing import Literal, Callable, Dict, Any
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.smith import RunEvalConfig, run_on_dataset
from langsmith import Client
from langsmith.evaluation import EvaluationResult, run_evaluator
from langsmith.schemas import Example, Run
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def init_llm(model_name: str = "gpt-4o") -> ChatOpenAI:
    """Initialize and return a ChatOpenAI instance."""
    try:
        return ChatOpenAI(
            model=model_name,
            temperature=0,
            api_key=OPENAI_API_KEY
        )
    except Exception as e:
        logging.error(f"Failed to initialize LLM: {e}")
        raise

cypher_schema = """
Node Properties:
- Drug(name: string, primarySubstabce: string)
- Case(primaryid: integer, age: float, ageUnit: string, gender: string, eventDate: date, reportDate: date, reporterOccupation: string)
- Reaction(description: string)
- Outcome(code: string, outcome: string)
- ReportSource(code: string, name: string)
- AgeGroup(ageGroup: string)
- Manufacturer(manufacturerName: string)
- Therapy(primaryid: integer)

Relationships:
- (Case)-[FALLS_UNDER]->(AgeGroup)
- (Manufacturer)-[REGISTERED]->(Case)
- (Case)-[RESULTED_IN]->(Outcome)
- (Case)-[HAS_REACTION]->(Reaction)
- (Case)-[REPORTED_BY]->(ReportSource)
- (Case)-[IS_PRIMARY_SUSPECT]->(Drug)
- (Case)-[IS_SECONDARY_SUSPECT]->(Drug)
- (Case)-[IS_CONCOMITANT]->(Drug)
- (Case)-[IS_INTERACTING]->(Drug)
- (Case)-[RECEIVED]->(Therapy)
- (Therapy)-[PRESCRIBED]->(Drug)

Relationship Properties:
- IS_PRIMARY_SUSPECT(drugSequence: integer, route: string, doseAmount: string, doseUnit: string, indication: string)
- IS_SECONDARY_SUSPECT(drugSequence: integer, route: string, doseAmount: string, doseUnit: string, indication: string)
- IS_CONCOMITANT(drugSequence: integer, route: string, doseAmount: string, doseUnit: string, indication: string)
- IS_INTERACTING(drugSequence: integer, route: string, doseAmount: string, doseUnit: string, indication: string)
- PRESCRIBED(drugSequence: integer, startYear: integer = 1900, endYear: integer = 2021)

Constraints:
- Drug.name is unique
- Case.primaryid is unique
- Reaction.description is unique
- ReportSource.code is unique
- Outcome.code is unique
- Therapy.primaryid is unique
- Manufacturer.manufacturerName is unique

Indexes:
- Case.age
- Case.ageUnit
- Case.gender
- Case.eventDate
- Case.reportDate

Node Descriptions:
- Case: Demographic information of a person (Case) involved in the adverse event report.
- Drug: Drug involved in the adverse event. Drug can be a primary suspect, secondary suspect, concomitant or interacting drug responsible for the adverse effect. This suspect type is identified by the relationship between Case and Drug Nodes.
- Reaction: Adverse reaction(side effect) that the Case developed after consumption of the respective Drug.
- Outcome: Long-term outcome of the Case after the Reaction.
- ReportSource: Information about who reported the adverse event.
- Therapy: Details about the therapy received by the case, including drug administration.
- AgeGroup: Age information reporting of case.

Relationship Descriptions:
- FALLS_UNDER: Relate Case and AgeGroup
- REGISTERED: Relate Case and Manufacturer
- RESULTED_IN: Relate Case and Outcome
- HAS_REACTION: Relate Case and Reaction
- REPORTED_BY: Relate Case and ReportSource
- IS_PRIMARY_SUSPECT: Relate Case and Drug based on primary suspect type
- IS_SECONDARY_SUSPECT: Relate Case and Drug based on secondary suspect type
- IS_CONCOMITANT: Relate Case and concomitant of Drug
- IS_INTERACTING: Relate Case and interacting among Drug
- RECEIVED: Relate Case and Therapy
- PRESCRIBED: Relate Therapy and Drug
"""

EVALUATION_TEMPLATE = """
You are an expert Cypher query reviewer. Your task is to evaluate the generated Cypher query against the expected Cypher query for a given natural language input. Please consider the Cypher schema provided.

Schema for generating Cypher query:
{cypher_schema}

Natural Language input: {input}
Generated Cypher Query: {generated_cypher} 
Expected Cypher Query: {expected_cypher}

Please evaluate the generated Cypher query based on the following criteria: 
1. Correctness: Does the generated Cypher query correctly answer the natural language input?
2. Efficiency: Is the generated Cypher query optimized and efficient?
3. Readability: Is the generated Cypher query easy to read and understand?

Provide a score from 0 to 5 for each criterion, where:
0 = The query does not reflect the criterion at all
1 = Poor
2 = Fair
3 = Good
4 = Very Good
5 = Excellent (Best possible)

Also, calculate an overall score as the average of the three criteria scores, rounded to one decimal place.
Provide a brief explanation for your evaluation. In your explanation, concisely summarize the key differences between the Generated Cypher Query and the Expected Cypher Query, and explain how these differences influenced your scoring.

Format your response as follows:
Correctness Score: [score]
Efficiency Score: [score]
Readability Score: [score]
Overall Score: [score]
Explanation: [Provide a concise summary of the key differences between the Generated and Expected Cypher Queries, and explain how these differences affected your scoring. Limit your explanation to 2-3 sentences.]
"""

EVALUATION_PROMPT = PromptTemplate(
    input_variables=["input", "generated_cypher", "expected_cypher", "cypher_schema"],
    template=EVALUATION_TEMPLATE,
)

def create_evaluation_chain(llm: ChatOpenAI) -> LLMChain:
    """Create and return an LLMChain for evaluation."""
    return LLMChain(llm=llm, prompt=EVALUATION_PROMPT)

ScoreType = Literal["overall", "correctness", "efficiency", "readability"]

def parse_evaluation_result(evaluation: str) -> Dict[str, Any]:
    """Parse the evaluation result string into a structured dictionary."""
    lines = evaluation.strip().split("\n")
    scores = {}
    explanation = ""
    
    try:
        for line in lines:
            if "Score:" in line:
                try:
                    key, value = line.split(":")
                    scores[key.strip()] = float(value.strip())
                except ValueError as ve:
                    logging.error(f"Error parsing score line '{line}': {ve}")
            elif "Explanation:" in line:
                explanation = line.replace("Explanation:", "").strip()
        
        return {"scores": scores, "explanation": explanation}
    
    except Exception as e:
        logging.error(f"Error parsing evaluation result: {e}")
        return {"scores": {}, "explanation": "Error parsing evaluation result"}

def create_evaluation_function(score_type: ScoreType) -> Callable[[Run, Example | None], EvaluationResult]:
    @run_evaluator
    def evaluator(run: Run, example: Example | None = None) -> EvaluationResult:
        evaluation = create_evaluation_chain(init_llm()).run(
            input=example.inputs["input"], 
            generated_cypher=run.outputs["output"],
            expected_cypher=example.outputs["output"],
            cypher_schema=cypher_schema,
        )
        parsed_result = parse_evaluation_result(evaluation)
        scores: Dict[str, float] = parsed_result["scores"]
        
        explanation = parsed_result["explanation"]

        score_mapping = {
            "overall": "Overall Score",
            "correctness": "Correctness Score",
            "efficiency": "Efficiency Score",
            "readability": "Readability Score",
        }
        
        return EvaluationResult(
            key="custom_evaluator",
            score=scores.get(score_mapping[score_type], 0),
            comment=f"Score Type: {score_type}\nExplanation: {explanation}"
        )
    return evaluator


def create_cypher_generation_chain(llm: ChatOpenAI) -> LLMChain:
    """Create and return an LLMChain for Cypher query generation."""
    cypher_prompt = PromptTemplate(
        input_variables=["cypher_schema", "input"], 
        template="""Task: Generate a Cypher statement to query the FDA Adverse Event Reporting System graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Optimize the query for performance by using appropriate indexes and limiting results where applicable.

Schema:
{cypher_schema}
Note: Do not include any explanations or apologies in your responses.
Do not respond to any inputs that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

The input is:
{input}"""
    )
    return LLMChain(llm=llm, prompt=cypher_prompt)

def generate_cypher_query(chain: LLMChain, cypher_schema: str, input: str) -> str:
    """Generate a Cypher query using the provided chain, schema, and input."""
    return chain.run(cypher_schema=cypher_schema, input=input)

def run_evaluation(dataset_name: str, project_name: str):
    """Run the evaluation process."""
    try:
        llm = init_llm()
        cypher_generation_chain = create_cypher_generation_chain(llm)

        client = Client()

        custom_evaluator_overall = create_evaluation_function("overall")
        custom_evaluator_correctness = create_evaluation_function("correctness")
        custom_evaluator_efficiency = create_evaluation_function("efficiency")
        custom_evaluator_readability = create_evaluation_function("readability")

        evaluation_config = RunEvalConfig(
            custom_evaluators=[
                custom_evaluator_overall, 
                custom_evaluator_correctness, 
                custom_evaluator_efficiency, 
                custom_evaluator_readability,
            ]
        )

        def generate_query(x):
            return {"output": generate_cypher_query(cypher_generation_chain, cypher_schema, x["input"])}

        run_on_dataset(
            client=client,
            llm_or_chain_factory=generate_query,
            dataset_name=dataset_name,
            evaluation=evaluation_config,
            project_name=project_name,
            verbose=True
        )

        logging.info("Evaluation completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred during evaluation: {e}")

if __name__ == "__main__":
    run_evaluation(dataset_name="test-10", project_name="eval_test_30")


