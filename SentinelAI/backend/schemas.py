from pydantic import BaseModel
from typing import Optional

class AnalyzeRequest(BaseModel):
    request_type: str
    data: dict

class AnalyzeResponse(BaseModel):
    decision: str          # ALLOW | WARNING | BLOCK
    layer: Optional[str]
    confidence: float
