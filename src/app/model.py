from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict

class BBox(BaseModel):
    x: int
    y: int
    width: int
    height: int

class ValidateRequest(BaseModel):
    image_base64: str
    bbox: BBox
    class_name: str
    validators: Optional[List[Literal['openai', 'gemini', 'nvidia']]] = None

class ValidatorResult(BaseModel):
    label: str
    confidence: int = Field(..., ge=1, le=10)
    validated: bool

class ValidateResponse(BaseModel):
    __root__: Dict[str, ValidatorResult]