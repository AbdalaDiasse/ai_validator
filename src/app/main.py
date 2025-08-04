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
    try:
        # ✅ Step 1: Decode and crop image safely
        try:
            img = decode_image(req.image_base64)
            crop = crop_image(img, req.bbox)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Image processing error: {str(e)}")

        name = req.validator
        if name not in VALIDATORS:
            raise HTTPException(status_code=400, detail=f"Unknown validator: {name}")

        # ✅ Step 2: Call the validator
        try:
            val = VALIDATORS[name].validate(crop, req.task)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Validator execution error: {str(e)}")

        # ✅ Step 3: Ensure validator returned a proper structure
        if not isinstance(val, dict) or not val.get("success", True):
            error_msg = val.get("error", "Validator returned an error") if isinstance(val, dict) else "Invalid validator response"
            raise HTTPException(status_code=500, detail=error_msg)

        detections = val.get("detections", [])
        if not isinstance(detections, list):
            raise HTTPException(status_code=500, detail="Validator returned detections in invalid format")

        # ✅ Step 4: Handle empty detection list
        if len(detections) == 0:
            print("No License platen detected")
            return ValidateResponse({name: ValidatorResult(label="0", confidence=1)})


        # ✅ Step 5: Process first detection safely (or loop if needed)
        first_det = detections[0]
        print("first_det:", first_det)
        print([k in first_det for k in ["label", "confidence"]])
        if not all(k in first_det for k in ["label", "confidence"]):
            raise HTTPException(status_code=500, detail="Malformed detection result from validator")

        try:
            conf10 = max(1, min(int(first_det["confidence"]), 100))
        except (ValueError, TypeError):
            raise HTTPException(status_code=500, detail="Invalid confidence value in detection result")

        # ✅ Step 6: Build and return response
        result = {
            name: ValidatorResult(
                label=first_det["label"],
                confidence=conf10
            )
        }
        return ValidateResponse({name: ValidatorResult(label=first_det["label"], confidence=conf10)})
        # return ValidateResponse(result=result)

    except HTTPException:
        raise  # let FastAPI handle it normally
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {str(e)}")

