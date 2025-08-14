import asyncio
from typing import Dict, List
from anyio import to_thread
from PIL import Image
from src.app.utils.image_utils import decode_image, crop_image
from src.app.utils.trackbuf import TrackBuffer, BatchBuilder, crop_score
from src.app.validators.gemini_validator import GeminiValidator
from src.app.settings import settings

class PatternAService:
    def __init__(self):
        self.top_n = settings.top_n
        self.idle_ms = settings.idle_ms
        self.buffers: Dict[str, TrackBuffer] = {}
        self.results: Dict[str, dict] = {}
        self.batcher = BatchBuilder(settings.max_imgs, settings.max_wait_ms)
        self.validator = GeminiValidator()
        self.sem = asyncio.Semaphore(settings.max_parallel_calls)
        self._task = None
        self._stop = asyncio.Event()

    async def start(self):
        self._stop.clear()
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        self._stop.set()
        if self._task:
            await self._task

    async def ingest(self, req: dict):
        # decode & crop off-thread
        img = await to_thread.run_sync(decode_image, req["image_base64"])
        crop = await to_thread.run_sync(crop_image, img, req["bbox"]) if req["bbox"] else img
        score = await to_thread.run_sync(crop_score, crop, req["frame_w"], req["frame_h"])
        tb = self.buffers.get(req["track_id"])
        if tb is None:
            tb = self.buffers[req["track_id"]] = TrackBuffer(req["track_id"], self.top_n, self.idle_ms)
        payload = {"image": crop}
        tb.add(score, payload)
        # mark pending for this track
        self.results.setdefault(req["track_id"], {"status":"pending", "result":None})

    async def _loop(self):
        # periodic: flush ready tracks into batcher, call gemini on batches
        try:
            while not self._stop.is_set():
                # move ready tracks to batcher
                ready_ids = [tid for tid, tb in list(self.buffers.items()) if tb.ready()]
                for tid in ready_ids:
                    crops = self.buffers[tid].best()
                    del self.buffers[tid]
                    if crops:
                        self.batcher.enqueue(tid, crops)

                # build next batch
                batch, map_index = self.batcher.next_batch()
                if batch:
                    # call gemini in a controlled concurrency
                    asyncio.create_task(self._process_batch(batch, map_index))
                await asyncio.sleep(0.02)
        except asyncio.CancelledError:
            pass

    async def _process_batch(self, batch: List[dict], map_index: List[tuple]):
        async with self.sem:
            # gather images in order
            images = [item["image"] for item in batch]
            results = await self.validator.batch_recognize(images)  # list aligned to images
            # regroup by track & vote (mode; tie-break by confidence)
            per_track: Dict[str, List[dict]] = {}
            for (tid, _local_idx), r in zip(map_index, results):
                per_track.setdefault(tid, []).append(r)

            for tid, arr in per_track.items():
                # filter blanks
                arr = [r for r in arr if r.get("label")]
                if not arr:
                    self.results[tid] = {"status":"done", "result": None}
                    continue
                # pick best by confidence (you can implement majority vote here)
                best = max(arr, key=lambda x: x.get("confidence",0))
                self.results[tid] = {
                    "status":"done",
                    "result":{"label": best["label"], "confidence": best["confidence"]}
                }
