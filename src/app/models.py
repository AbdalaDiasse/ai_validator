from pydantic import BaseModel, Field ,RootModel
from typing import List, Literal, Optional, Dict


class BBox(BaseModel):
    x: int
    y: int
    width: int
    height: int

class IngestRequest(BaseModel):
    track_id: str
    frame_id: str
    ts: Optional[str] = None
    frame_w: int
    frame_h: int
    bbox: Optional[BBox]=None  # [x1,y1,x2,y2] in frame coords
    image_base64: str

class RecognizeResult(BaseModel):
    label: str
    confidence: int = Field(ge=0, le=100)

class RecognizeResponse(BaseModel):
    track_id: str
    result: Optional[RecognizeResult] = None
    status: str  # "pending" | "done" | "none"

class HealthzResponse(BaseModel):
    status: str = "ok"


class ValidateRequest(BaseModel):
    image_base64: str
    # bbox: BBox
    bbox: Optional[BBox] = None   # <-- bbox is now optional
    task: Literal["lpr", "lpd", "age", "gender"]   #restrict to only these options
    validator: Literal["openai", "gemini", "nvidia"]   #restrict to only these options

class ValidatorResult(BaseModel):
    label: str
    confidence: int = Field(..., ge=1, le=100)
    detail: Optional[str] = None  # Additional info if needed

class ValidateResponse(RootModel[Dict[str, ValidatorResult]]):
    root: Dict[str, ValidatorResult]