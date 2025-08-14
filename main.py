# main.py
import logging
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from starlette.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from src.app.routers.validate import router as validate_router
from src.app.routers.recognize import router as recognize_router
from src.app.middleware.request_id import request_id_middleware
from src.app.utils.http import get_async_client
from src.app.utils.logging import setup_logging
from src.app.services.pattern_a import PatternAService
from src.app.services.pattern_b import PatternBService
from src.app.models import HealthzResponse
log = logging.getLogger("uvicorn.error")
SERVICE = None
mode = "pattern_a"  # or "pattern_a", set based on your configuration
def create_app() -> FastAPI:
    setup_logging()
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        get_async_client()
        global SERVICE
        if mode == "pattern_b":
            SERVICE = PatternBService()
        else:
            SERVICE = PatternAService()
        await SERVICE.start()
        logging.getLogger(__name__).info("Startup complete")
        yield
        await SERVICE.stop()
        # Shutdown
        client = get_async_client()
        await client.aclose()
        logging.getLogger(__name__).info("Shutdown complete")

    app = FastAPI(title="VLM Validator Service",lifespan=lifespan)

    app.middleware("http")(request_id_middleware(app))
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],            # tighten in prod
        allow_methods=["POST", "GET", "OPTIONS"],
        allow_headers=["*"],
    )

    app.include_router(validate_router)
    app.include_router(recognize_router)

    @app.get("/healthz", response_model=HealthzResponse)
    async def healthz() :
        return {"status": "ok"}
    
    
    # @app.get("/healthz", response_class=PlainTextResponse)
    # async def healthz() -> str:
    #     return "ok"
    
    def get_service():
        return SERVICE

    # expose getter to other modules
    app.state.get_service = get_service

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
