import asyncio
from typing import Dict, List
from anyio import to_thread
from PIL import Image
from src.app.utils.image_utils import decode_image, crop_image
from src.app.utils.trackbuf import TrackBuffer, crop_score
from src.app.validators.gemini_validator import GeminiValidator
from src.app.settings import settings

class TrackActor:
    def __init__(self, track_id: str, top_n: int, idle_ms: int, validator: GeminiValidator, done_cb):
        self.track_id = track_id
        self.tb = TrackBuffer(track_id, top_n, idle_ms)
        self.validator = validator
        self.done_cb = done_cb
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task = asyncio.create_task(self._run())

    async def push(self, crop: Image.Image, frame_w: int, frame_h: int):
        score = await to_thread.run_sync(crop_score, crop, frame_w, frame_h)
        await self._queue.put(("add", score, crop))

    async def _run(self):
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(self._queue.get(), timeout=self.tb.idle)
                except asyncio.TimeoutError:
                    # idle → flush if any
                    if self.tb.heap:
                        await self._flush()
                        return
                    continue

                kind = msg[0]
                if kind == "add":
                    _, score, crop = msg
                    self.tb.add(score, {"image": crop})
        except asyncio.CancelledError:
            pass

    async def _flush(self):
        crops = self.tb.best()
        images = [c["image"] for c in crops]
        results = await self.validator.batch_recognize(images)
        # choose best
        arr = [r for r in results if r.get("label")]
        best = max(arr, key=lambda x: x.get("confidence",0)) if arr else None
        await self.done_cb(self.track_id, best)

    async def stop(self):
        if not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

class PatternBService:
    def __init__(self):
        self.top_n = settings.top_n
        self.idle_ms = settings.idle_ms
        self.validator = GeminiValidator()
        self.actors: Dict[str, TrackActor] = {}
        self.results: Dict[str, dict] = {}

    async def start(self): pass
    async def stop(self):
        for a in list(self.actors.values()):
            await a.stop()

    async def ingest(self, req: dict):
        img = await to_thread.run_sync(decode_image, req["image_base64"])
        crop = await to_thread.run_sync(crop_image, img, req["bbox"])
        actor = self.actors.get(req["track_id"])
        if actor is None:
            actor = self.actors[req["track_id"]] = TrackActor(
                req["track_id"], self.top_n, self.idle_ms, self.validator, self._on_done
            )
        self.results.setdefault(req["track_id"], {"status":"pending", "result":None})
        await actor.push(crop, req["frame_w"], req["frame_h"])

    async def _on_done(self, track_id: str, best: dict | None):
        if best:
            self.results[track_id] = {"status":"done", "result":{"label":best["label"], "confidence":best["confidence"]}}
        else:
            self.results[track_id] = {"status":"done", "result":None}
        # cleanup actor
        actor = self.actors.pop(track_id, None)
        if actor: await actor.stop()
