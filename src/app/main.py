import os
from fastapi import FastAPI, HTTPException
from pydantic_settings import BaseSettings  # updated for pydantic v2+ (install pydantic-settings)
from typing import List
from src.app.models import ValidateRequest, ValidateResponse, ValidatorResult
from src.app.utils.image_utils import decode_image, crop_image
from src.app.validators.openai_validator import OpenAIValidator
from src.app.validators.gemini_validator import GeminiValidator
from src.app.validators.nvidia_validator import NvidiaValidator
from src.app.settings import settings

app = FastAPI(title='VLM Validator Service')

VALIDATORS = {
    'openai': OpenAIValidator(),
    'gemini': GeminiValidator(),
    'nvidia': NvidiaValidator()
}

@app.post('/validate', response_model=ValidateResponse)
def validate(req: ValidateRequest):
    print("Received validation request:", req)
    try:
        img = decode_image(req.image_base64)
        crop = crop_image(img, req.bbox)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    names = req.validators or list(VALIDATORS.keys())
    result = {}
    for name in names:
        if name not in VALIDATORS:
            raise HTTPException(status_code=400, detail=f"Unknown validator: {name}")
        val = VALIDATORS[name].validate(crop,req.task)
        # print(val)
        conf10 = max(1, min(int(val["detections"]['confidence'] ), 100))
        result[name] = ValidatorResult(label=val["detections"]['label'], confidence=conf10)

    return ValidateResponse.parse_obj(result)