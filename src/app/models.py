from pydantic import BaseModel, Field ,RootModel
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
    task: Literal["lpr", "age", "gender"]   #restrict to only these options
    validators: Optional[List[Literal['openai', 'gemini', 'nvidia']]] = None

class ValidatorResult(BaseModel):
    label: str
    confidence: int = Field(..., ge=1, le=10)
    validated: bool

class ValidateResponse(RootModel[Dict[str, ValidatorResult]]):
    root: Dict[str, ValidatorResult]