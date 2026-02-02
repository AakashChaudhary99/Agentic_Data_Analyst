from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import TypedDict, List, Optional, Dict, Any, Literal
import pandas as pd
import tempfile
import shutil
import io
import os
from contextlib import asynccontextmanager

# -------------------------------------------------
# FastAPI setup
# -------------------------------------------------

llm = ChatOpenAI(model="gpt-4.1-mini")

# -------------------------------------------------
# State & Schemas
# -------------------------------------------------

class IntentStructure(TypedDict):
    intent: str
    output_type: Literal["text", "csv"]

class RequiredColsStructure(TypedDict):
    req_cols: List[str]

class TransformationPlan(BaseModel):
    steps: List[str]

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
# Structured LLMs (FORCED function_calling)
# -------------------------------------------------

intent_structured_llm = llm.with_structured_output(
    IntentStructure,
    method="function_calling",
)

# -------------------------------------------------
# Graph nodes
# -------------------------------------------------

def inspect_dataset_node(state: AgentState) -> AgentState:
    df = pd.read_csv(state["file_path"])
    state["cols"] = list(df.columns)
    state["sample_data"] = df.head(2).to_dict(orient="records")
    return state

def column_selection_node(state: AgentState) -> AgentState:
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
    return state

def find_intent(state: AgentState):
    prompt = f"""
From the instruction below, decide:
1. intent
2. output_type: text or csv

Instruction:
{state['instruction']}
"""
    resp = intent_structured_llm.invoke(prompt)
    return {
        "output_mode": resp["output_type"],
        "intent": resp["intent"],
    }

def text_executor_node(state: AgentState) -> AgentState:
    df = pd.read_csv(state["file_path"])
    # df = df[state["req_cols"]]
    cols = state['cols']
    req_cols = state['req_cols']
    # print("SUmmary", summary)
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

    # state["text_output"] = response.content
    state['text_output_transformation_plan'] = response.steps
    # print("RESP", response.steps)
    transformed_df = apply_transformation_plan_safe(df, response.steps)
    text_response = llm.invoke(f"""
        From given pandas df as a string and user question below
        df: {transformed_df.to_string()}
        Question: {state['instruction']}
        ,
        provide an answer for that question. Just a brief answer nothing else
    """)
    state['text_output'] = text_response.content
    return state

def plan_generator_node(state: AgentState) -> AgentState:
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
    return state

def route_by_output(state: AgentState) -> Literal["text", "csv"]:
    return state["output_mode"]

# -------------------------------------------------
# Graph build
# -------------------------------------------------

graph = StateGraph(AgentState)

graph.add_node("inspect_dataset", inspect_dataset_node)
graph.add_node("column_selection", column_selection_node)
graph.add_node("find_intent", find_intent)
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

workflow = None
# -------------------------------------------------
# Helpers
# -------------------------------------------------

def apply_transformation_plan_safe(df, plan):
    for step in plan:
        exec(step)
    return df


# -------------------------------------------------
# APIs
# -------------------------------------------------
@asynccontextmanager
async def lifespan(app:FastAPI):
    global workflow
    workflow = graph.compile()
    yield
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def health_check():
    print("HEALTH CHECK HIT")
    return {"Message": "API is healthy"}

@app.post("/processing")
def processing_file(
    file: UploadFile = File(...),
    instruction: str = Form(...),
):
    print("Inside processing")
    global workflow
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp_path = tmp.name
        shutil.copyfileobj(file.file, tmp)

    op = workflow.invoke(
        {"instruction": instruction, "file_path": tmp_path}
    )
    print("OP", op)
    if op["output_mode"] == "text":
        return op["text_output"]

    df = pd.read_csv(tmp_path)
    new_df = apply_transformation_plan_safe(
        df, op["transformation_plan"]
    )

    buffer = io.StringIO()
    new_df.to_csv(buffer, index=False)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=output.csv"
        },
    )
