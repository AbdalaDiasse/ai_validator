from fastapi import APIRouter, HTTPException
from anyio import to_thread
from src.app.models import ValidateRequest, ValidateResponse, ValidatorResult
from src.app.utils.image_utils import decode_image, crop_image
from src.app.validators.openai_validator import OpenAIValidator
from src.app.validators.gemini_validator import GeminiValidator
from src.app.validators.nvidia_validator import NvidiaValidator

router = APIRouter()
VALIDATORS = {
    "openai": OpenAIValidator(),
    "gemini": GeminiValidator(),
    "nvidia": NvidiaValidator(),
}

@router.post("/validate", response_model=ValidateResponse)
async def validate(req: ValidateRequest):
    try:
        # decode/crop on a worker thread if it's CPU-heavy (Pillow)
        try:
            img = await to_thread.run_sync(decode_image, req.image_base64)
            crop = await to_thread.run_sync(crop_image, img, req.bbox) if req.bbox is not None else img
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Image processing error: {str(e)}")

        name = req.validator
        if name not in VALIDATORS:
            print(f"Unknown validator: {name}") 
            raise HTTPException(status_code=400, detail=f"Unknown validator: {name}")

        # If validator is sync, offload to thread; if you implement .validate_async, call that instead.
        try:
            # val = await to_thread.run_sync(VALIDATORS[name].validate, crop, req.task)
            val = await VALIDATORS[name].validate(crop, req.task)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Validator execution error: {str(e)}")

        if not isinstance(val, dict) or not val.get("success", True):
            err = val.get("error", "Validator returned an error") if isinstance(val, dict) else "Invalid validator response"
            raise HTTPException(status_code=500, detail=err)

        detections = val.get("detections", [])
        if not isinstance(detections, list):
            raise HTTPException(status_code=500, detail="Validator returned detections in invalid format")

        if len(detections) == 0:
            return ValidateResponse({name: ValidatorResult(label="0", confidence=1, detail="No License plate detected")})

        det = detections[0]
        if not all(k in det for k in ["label", "confidence"]):
            raise HTTPException(status_code=500, detail="Malformed detection result from validator")

        try:
            conf10 = max(1, min(int(det["confidence"]), 100))
        except (ValueError, TypeError):
            raise HTTPException(status_code=500, detail="Invalid confidence value in detection result")

        return ValidateResponse({name: ValidatorResult(label=det["label"], confidence=conf10 , detail="License plate detected")})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {str(e)}")
