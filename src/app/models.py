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
    task: Literal["lpr", "lpd", "age", "gender"]   #restrict to only these options
    validator: Literal["openai", "gemini", "nvidia"]   #restrict to only these options

class ValidatorResult(BaseModel):
    label: str
    confidence: int = Field(..., ge=1, le=100)

class ValidateResponse(RootModel[Dict[str, ValidatorResult]]):
    root: Dict[str, ValidatorResult]