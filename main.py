from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
import tempfile
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import shutil
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse
import io

app = FastAPI()
from typing import TypedDict, List, Optional, Dict, Any, Literal

import os
os.environ["OPENAI_API_KEY"] = ""

app.add_middleware(
  CORSMiddleware,
  allow_origins=["http://localhost:5173"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"]
)

llm = ChatOpenAI()


class IntentStructure(TypedDict):
    intent: str
    output_type: Literal["text", "csv"]

class RequiredColsStructure(TypedDict):
    req_cols: List[str]

intent_structured_llm = llm.with_structured_output(IntentStructure)

class TransformationStep(BaseModel):
    operation: str = Field(
        description="Generic dataframe operation name, e.g. groupby, filter, order by, etc"
    )
    params: Dict[str, Any] = Field(
        description="Parameters required to execute the operation"
    )


class TransformationPlan(BaseModel):
    steps: List[str]

class AgentState(TypedDict):
    instruction: str
    file_path: str
    output_mode: Literal["csv", 'text']
    intent: str
    cols: List[str]
    req_cols: List[str]
    text_output: str
    sample_data: Dict[str, any]
    transformation_plan: Optional[List[str]]

def find_intent(state: AgentState):
    instruction = state['instruction']
    prompt = f"""
        From this given instruction or question by the user: {instruction} for a csv file, Decide the 
        1. User intent
        2. Output type (If we solve this instruction or question, would the output be in the form of text or transformation is required?
        if this instruction or question does not require transformation then text or else csv)
    """

    resp = intent_structured_llm.invoke(prompt)
    return {"output_mode": resp['output_type'], "intent": resp['intent']}

def inspect_dataset_node(state: AgentState) -> AgentState:
    df = pd.read_csv(state["file_path"])

    state["cols"] = list(df.columns)
    state["sample_data"] = df.head(2).to_dict(orient="records")

    return state

def column_selection_node(state: AgentState) -> AgentState:
    structured_llm = llm.with_structured_output(RequiredColsStructure)
    response = structured_llm.invoke(f"""
    For the given instruction and columns name,
    Instruction:
    {state['instruction']}

    Dataset columns:
    {state['cols']}

    Return only required columns as a list which will be used to solve above instruction.
    """)

    state["req_cols"] = response['req_cols']
    return state

def text_executor_node(state: AgentState) -> AgentState:
    df = pd.read_csv(state["file_path"])
    df = df[state["required_columns"]]

    summary = df.describe().to_string()

    response = llm.invoke(f"""
    Instruction:
    {state['instruction']}

    Data summary:
    {summary}

    Provide a concise insight.
    """)

    state["text_output"] = response["text"]
    return state

def plan_generator_node(state: AgentState) -> AgentState:
    structured_llm = llm.with_structured_output(TransformationPlan)
    response = structured_llm.invoke(f"""
    You are a senior Python data engineer.

Your task is to generate EXACT pandas code to solve the user instruction.

IMPORTANT RULES (DO NOT BREAK):
- Use ONLY pandas operations
- Assume a pandas DataFrame named `df` already exists
- Do NOT import anything
- Do NOT read or write files
- Do NOT use print statements
- Do NOT explain anything
- Do NOT return markdown
- Do NOT return plain text

OUTPUT FORMAT (STRICT):
- Return a list of all operation required step by step
- Each element must be a SINGLE pandas statement as a STRING
- Each statement must be executable in order
- Each statement must reassign to `df` if it modifies the dataframe


DATASET SCHEMA:
Columns: {state['cols']}

Sample rows:
{state['sample_data']}

USER INSTRUCTION:
{state['instruction']}

RETURN ONLY THIS (NO EXTRA TEXT):
[
  "df = ...",
  "df = ..."
]

    """)

    state["transformation_plan"] = response.steps
    return state


graph = StateGraph(AgentState)

# add nodes
graph.add_node('find_intent', find_intent)
graph.add_node("inspect_dataset", inspect_dataset_node)
graph.add_node("column_selection_node", column_selection_node)
graph.add_node("text_exec", text_executor_node)
graph.add_node('plan_generator_node', plan_generator_node)

graph.add_edge(START, 'find_intent')
graph.add_edge('find_intent', 'inspect_dataset')
graph.add_edge('inspect_dataset', 'column_selection_node')
graph.add_edge('column_selection_node', 'plan_generator_node')
graph.add_edge('plan_generator_node', END)

workflow = graph.compile()

def apply_transformation_plan_safe(df, plan):
    safe_globals = {
        "__builtins__": {},  # blocks open, import, eval, etc.
    }
    safe_locals = {
        "df": df
    }

    for step in plan:
        exec(step)

    return df

@app.get('/')
def health_check():
    return {"Message": "API is healthy"}

@app.post('/processing')
async def processing_file(file: UploadFile=File(...), instruction:str=Form(...)):
    try:
        print("Received", instruction)
        global workflow
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
            tmp_path = tmp.name
            shutil.copyfileobj(file.file, tmp)

        op = workflow.invoke({"instruction": instruction, 'file_path': tmp_path})
        df = pd.read_csv(tmp_path)
        transformations = op['transformation_plan']
        new_df = apply_transformation_plan_safe(df, transformations)
        print(new_df)
        buffer = io.StringIO()
        new_df.to_csv(buffer, index=False)
        buffer.seek(0)
        print(buffer)
        return StreamingResponse(
            buffer,
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=output.csv"
            }
        )

        # return {"status":"ok", 'data': op}

    except Exception as e:
        print(e)
