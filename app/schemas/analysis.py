from pydantic import BaseModel
from typing import Optional, List

class HealthCheckResponse(BaseModel):
    status: str
    message: str
    environment: str


class TransformationStep(BaseModel):
    step_number: int
    code: str


class TextAnalysisResponse(BaseModel):
    result: str
    steps_executed: List[str]
