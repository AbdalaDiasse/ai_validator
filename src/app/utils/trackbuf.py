import time, heapq
from typing import List, Tuple, Any, Deque
from collections import deque
import numpy as np
from PIL import Image, ImageStat
try:
    import cv2
except ImportError:
    cv2 = None

def variance_of_laplacian(gray_np: np.ndarray) -> float:
    if cv2 is None:
        gx = np.abs(np.gradient(gray_np, axis=1))
        gy = np.abs(np.gradient(gray_np, axis=0))
        return float(np.var(gx + gy))
    return float(cv2.Laplacian(gray_np, cv2.CV_64F).var())

def crop_score(crop: Image.Image, frame_w: int, frame_h: int) -> float:
    gray = crop.convert("L")
    arr = np.array(gray, dtype=np.float32)
    cw, ch = gray.size
    # features
    sharp = variance_of_laplacian(arr)
    area_frac = (cw * ch) / (frame_w * frame_h)
    aspect = cw / max(1, ch)
    aspect_penalty = min(abs(aspect - 4.0) / 4.0, 1.0)
    brightness = ImageStat.Stat(gray).mean[0] / 255.0
    brightness_penalty = min(abs(brightness - 0.6) / 0.6, 1.0)

    sharp_norm = min(sharp / 200.0, 1.0)
    score = (
        0.55 * sharp_norm +
        0.25 * min(area_frac / 0.02, 1.0) +
        0.10 * (1.0 - aspect_penalty) +
        0.10 * (1.0 - brightness_penalty)
    )
    return float(score)

class TrackBuffer:
    def __init__(self, track_id: str, top_n: int, idle_ms: int):
        self.track_id = track_id
        self.N = top_n
        self.heap: List[Tuple[float, Any]] = []
        self.last_ts = time.time()
        self.idle = idle_ms / 1000.0

    def add(self, score: float, payload: dict):
        self.last_ts = time.time()
        if len(self.heap) < self.N:
            heapq.heappush(self.heap, (score, payload))
        else:
            if score > self.heap[0][0]:
                heapq.heapreplace(self.heap, (score, payload))

    def ready(self) -> bool:
        return (time.time() - self.last_ts) > self.idle

    def best(self) -> List[dict]:
        return [p for _, p in sorted(self.heap, key=lambda x: -x[0])]

class BatchBuilder:
    def __init__(self, max_imgs: int, max_wait_ms: int):
        self.max_imgs = max_imgs
        self.max_wait = max_wait_ms / 1000.0
        self.queue: Deque[Tuple[str, List[dict]]] = deque()

    def enqueue(self, track_id: str, crops: List[dict]):
        self.queue.append((track_id, crops))

    def next_batch(self):
        import time as _time
        start = _time.time()
        batch, map_index = [], []  # [(track_id, local_idx)]
        while self.queue and len(batch) < self.max_imgs:
            tid, crops = self.queue[0]
            take = min(len(crops), self.max_imgs - len(batch))
            for i in range(take):
                batch.append(crops[i])
                map_index.append((tid, i))
            self.queue.popleft()
            if (_time.time() - start) >= self.max_wait:
                break
        return batch, map_index
