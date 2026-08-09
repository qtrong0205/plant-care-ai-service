from pydantic import BaseModel
from typing import List

class TopKPrediction(BaseModel):
    label: str
    confidence: float

class ClassifyResponse(BaseModel):
    predicted_label: str
    confidence: float
    top_k: List[TopKPrediction]
    status: str   