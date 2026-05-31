import logging
from typing import List, Optional, Dict, Any, Literal
import pandas as pd
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

from app.config import settings
from app.utils.security import is_code_safe
from app.services.executor import apply_transformation_plan

logger = logging.getLogger(__name__)

# -------------------------------------------------
# Models & State Schemas
# -------------------------------------------------

class IntentStructure(TypedDict):
    intent: str
    output_type: Literal["text", "csv"]


class RequiredColsStructure(TypedDict):
    req_cols: List[str]


class TransformationPlan(BaseModel):
    steps: List[str]


class ExceptionFixPlan(BaseModel):
    step: str


class AgentState(TypedDict):
    instruction: str
    file_path: str
    output_mode: Literal["csv", "text"]
    intent: str
    cols: List[str]
    req_cols: List[str]
    text_output_transformation_plan: Optional[List[str]]
    text_output: str
    sample_data: Dict[str, Any]
    transformation_plan: Optional[List[str]]


# -------------------------------------------------
# LLM Initialization
# -------------------------------------------------

llm = ChatOpenAI(
    api_key=settings.openai_api_key,
    model=settings.openai_model,
    temperature=0
)

intent_structured_llm = llm.with_structured_output(
    IntentStructure,
    method="function_calling",
)


# -------------------------------------------------
# Helper Correction Callback
# -------------------------------------------------

def fix_step_callback(failed_step: str, instruction: str, exception: Exception) -> str:
    """
    Callback invoked by the executor when a step fails.
    Uses ChatOpenAI to generate a corrected version of the failed pandas step.
    """
    logger.info(f"Self-correction callback triggered for step: '{failed_step}' due to: {exception}")
    
    structured_llm = llm.with_structured_output(
        ExceptionFixPlan,
        method="function_calling",
    )
    
    fix_prompt = f"""
    A pandas transformation step failed.
    
    Instruction:
    {instruction}
    
    Failed step:
    {failed_step}
    
    Error:
    {str(exception)}
    
    Provide corrected pandas step only.
    Return one line starting with df = ...
    """
    
    response = structured_llm.invoke(fix_prompt)
    fixed_step = response.step.strip()
    
    # Pre-validate security of the fixed step
    is_safe, error_msg = is_code_safe(fixed_step)
    if not is_safe:
        raise ValueError(f"LLM proposed unsafe code fix: {error_msg}")
        
    return fixed_step


# -------------------------------------------------
# Graph Nodes
# -------------------------------------------------

def inspect_dataset_node(state: AgentState) -> AgentState:
    """Reads dataset, updates state with all column names and metadata sample."""
    logger.info(f"Inspecting dataset at {state['file_path']}")
    df = pd.read_csv(state["file_path"])
    state["cols"] = list(df.columns)
    state["sample_data"] = df.head(2).to_dict(orient="records")
    return state


def column_selection_node(state: AgentState) -> AgentState:
    """Asks LLM to select relevant columns for answering the user instruction."""
    logger.info("Running column selection...")
    structured_llm = llm.with_structured_output(
        RequiredColsStructure,
        method="function_calling",
    )

    response = structured_llm.invoke(
        f"""
        From below user instruction/question and the columns names of file,
        Instruction:
        {state['instruction']}
        
        Columns:
        {state['cols']}
        
        Return required columns only which will be used to solve the user's question/instruction.
        """
    )

    state["req_cols"] = response["req_cols"]
    logger.info(f"Selected columns: {state['req_cols']}")
    return state


def find_intent_node(state: AgentState) -> Dict[str, Any]:
    """Identifies the user's intent and required output format (text or csv)."""
    logger.info("Determining request intent and output format...")
    prompt = f"""
    From the instruction below, decide:
    1. intent
    2. output_type: text or csv
    
    Instruction:
    {state['instruction']}
    """
    resp = intent_structured_llm.invoke(prompt)
    logger.info(f"Determined intent: '{resp['intent']}' (mode: {resp['output_type']})")
    return {
        "output_mode": resp["output_type"],
        "intent": resp["intent"],
    }


def text_executor_node(state: AgentState) -> AgentState:
    """Executes safe analysis steps for requests expecting textual output."""
    logger.info("Running text executor workflow...")
    df = pd.read_csv(state["file_path"])
    cols = state['cols']
    req_cols = state['req_cols']
    
    structured_llm = llm.with_structured_output(
        TransformationPlan,
        method="function_calling",
    )
    
    response = structured_llm.invoke(
        f"""
        From given user's questions, all columns of file and required columns of file below
        Question:
        {state['instruction']}
        
        All Columns: {cols}
        Required columns: {req_cols}
        
        Generate pandas transformation steps which will be enough to get a dataset df so that user's question can be answered.
        
        Return JSON ONLY in this format:
        {{
          "steps": [
            "df = ...",
            "df = ..."
          ]
        }}.
        """
    )

    state['text_output_transformation_plan'] = response.steps
    
    # Run the plan with sandbox executor and LLM self-correction callback
    transformed_df = apply_transformation_plan(
        df, 
        response.steps, 
        state['instruction'], 
        fix_step_callback
    )
    
    # Prompt the LLM to format the resulting DataFrame into a natural language response
    text_response = llm.invoke(f"""
        From given pandas df as a string and user question below
        df: {transformed_df.to_string()}
        Question: {state['instruction']}
        ,
        provide an answer for that question. Just a brief answer nothing else
    """)
    
    state['text_output'] = text_response.content
    logger.info("Generated text output successfully.")
    return state


def plan_generator_node(state: AgentState) -> AgentState:
    """Generates the pandas execution plan for CSV transformation requests."""
    logger.info("Generating transformation plan for CSV output...")
    structured_llm = llm.with_structured_output(
        TransformationPlan,
        method="function_calling",
    )

    response = structured_llm.invoke(
        f"""
        Generate pandas transformation steps.
        
        Return JSON ONLY in this format:
        {{
          "steps": [
            "df = ...",
            "df = ..."
          ]
        }}
        
        Columns: {state['cols']}
        Sample: {state['sample_data']}
        Instruction: {state['instruction']}
        """
    )

    state["transformation_plan"] = response.steps
    logger.info(f"Generated transformation plan steps: {response.steps}")
    return state


def route_by_output(state: AgentState) -> Literal["text", "csv"]:
    """Conditional router based on state's output mode."""
    return state["output_mode"]


# -------------------------------------------------
# Graph Compilation
# -------------------------------------------------

graph = StateGraph(AgentState)

graph.add_node("inspect_dataset", inspect_dataset_node)
graph.add_node("column_selection", column_selection_node)
graph.add_node("find_intent", find_intent_node)
graph.add_node("text", text_executor_node)
graph.add_node("csv", plan_generator_node)

graph.add_edge(START, "inspect_dataset")
graph.add_edge("inspect_dataset", "column_selection")
graph.add_edge("column_selection", "find_intent")

graph.add_conditional_edges(
    "find_intent",
    route_by_output,
    {
        "text": "text",
        "csv": "csv",
    },
)

graph.add_edge("text", END)
graph.add_edge("csv", END)

workflow = graph.compile()
