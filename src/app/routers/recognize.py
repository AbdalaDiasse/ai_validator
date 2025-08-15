from fastapi import APIRouter, HTTPException, Request
from src.app.models import IngestRequest, RecognizeResponse
from fastapi.responses import JSONResponse

router = APIRouter()

@router.post("/ingest", response_model=None)
async def ingest(req: Request, body: IngestRequest):
    service = req.app.state.get_service()
    try:
        await service.ingest(body)
        return JSONResponse({"status":"accepted"})
    except Exception as e:
        raise HTTPException(500, f"Ingestion error: {e}")

@router.get("/results/{track_id}", response_model=RecognizeResponse)
async def get_result(req: Request, track_id: str):
    service = req.app.state.get_service()
    r = service.results.get(track_id)
    if not r:
        return RecognizeResponse(track_id=track_id, status="none", result=None)
    if r["status"] == "done" and r["result"]:
        return RecognizeResponse(track_id=track_id, status="done", result=r["result"])
    return RecognizeResponse(track_id=track_id, status=r["status"], result=None)
