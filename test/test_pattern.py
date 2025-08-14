import argparse
import asyncio
import base64
import os
from pathlib import Path
from typing import List, Tuple
import httpx
from PIL import Image


def img_to_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def default_bbox(img: Image.Image) -> dict:
    """A simple bbox covering the center third of the image (demo only)."""
    w, h = img.size
    x1 = w // 3
    x2 = 2 * w // 3
    y1 = h // 3
    y2 = 2 * h // 3
    bbox = {"x": 1211, "y": 743, "width": 109, "height": 38}
    return bbox  # Convert to BBox and back to dict for consistency


async def health_check(client: httpx.AsyncClient, base_url: str) -> None:
    r = await client.get(f"{base_url.rstrip('/')}/healthz")
    r.raise_for_status()
    print("Health:", r.text if r.headers.get("content-type","").startswith("text/") else r.json())


async def ingest_crop(
    client: httpx.AsyncClient,
    base_url: str,
    *,
    track_id: str,
    frame_id: str,
    img_path: Path,
    bbox: dict,
    end_of_track: bool,
) -> None:
    # read dims for payload
    with Image.open(img_path) as im:
        w, h = im.size
    payload = {
        "track_id": track_id,
        "frame_id": frame_id,
        "frame_w": w,
        "frame_h": h,
        "bbox": bbox,
        "image_base64": img_to_b64(img_path),
        # NOTE: your current recognize router didn’t include this field,
        # but if you add it, the service can flush immediately (Pattern A fast-path).
        "end_of_track": end_of_track,
    }
    r = await client.post(f"{base_url.rstrip('/')}/ingest", json=payload)
    if r.status_code >= 300:
        print(f"[{track_id}:{frame_id}] ingest FAILED:", r.text)
    else:
        try:
            print(f"[{track_id}:{frame_id}] ingest →", r.json())
        except Exception:
            print(f"[{track_id}:{frame_id}] ingest →", r.text)


async def get_result(client: httpx.AsyncClient, base_url: str, track_id: str, wait_ms: int) -> dict:
    r = await client.get(f"{base_url.rstrip('/')}/results/{track_id}", params={"wait_ms": str(wait_ms)})
    r.raise_for_status()
    data = r.json()
    print(f"[{track_id}] result:", data)
    return data


async def test_pattern_a(
    client: httpx.AsyncClient,
    base_url: str,
    images: List[Path],
    top_n: int,
    poll_wait_ms: int,
) -> None:
    """
    Sends N crops for one track, then end_of_track=true, then long-polls results.
    Repeat for multiple tracks (one after another).
    """
    track_idx = 0
    for i in range(0, len(images), top_n):
        track_idx += 1
        tid = f"trackA_{track_idx:03d}"
        batch = images[i : i + top_n]
        if not batch:
            break

        # send all N crops; last one marks end_of_track
        for j, img_path in enumerate(batch, 1):
            with Image.open(img_path) as im:
                bbox = default_bbox(im)
                print(bbox["x"], bbox["y"], bbox["width"], bbox["height"])
            await ingest_crop(
                client,
                base_url,
                track_id=tid,
                frame_id=f"f{j}",
                img_path=img_path,
                bbox=bbox,
                end_of_track=(j == len(batch)),  # flush now
            )

        # wait for result (long-poll)
        await get_result(client, base_url, tid, poll_wait_ms)


async def test_pattern_b(
    client: httpx.AsyncClient,
    base_url: str,
    images: List[Path],
    tracks: int,
    crops_per_track: int,
    interleave_delay_ms: int,
    poll_wait_ms: int,
) -> None:
    """
    Interleaves multiple tracks without end_of_track; relies on idle timeout for flush.
    """
    # build per-track slices
    per_track_imgs = [images[i * crops_per_track : (i + 1) * crops_per_track] for i in range(tracks)]
    tids = [f"trackB_{i+1:03d}" for i in range(tracks)]

    # interleave sending
    for k in range(crops_per_track):
        tasks = []
        for t_index, tid in enumerate(tids):
            imgs = per_track_imgs[t_index]
            if k < len(imgs):
                img_path = imgs[k]
                with Image.open(img_path) as im:
                    bbox = default_bbox(im)
                tasks.append(
                    ingest_crop(
                        client,
                        base_url,
                        track_id=tid,
                        frame_id=f"f{k+1}",
                        img_path=img_path,
                        bbox=bbox,
                        end_of_track=False,  # rely on idle timeout
                    )
                )
        if tasks:
            await asyncio.gather(*tasks)
        # slight delay to let batches form and to simulate staggered frames
        await asyncio.sleep(max(0.0, interleave_delay_ms / 1000.0))

    # after sending, long-poll each track’s result
    await asyncio.gather(*(get_result(client, base_url, tid, poll_wait_ms) for tid in tids))


def collect_images(images_dir: Path, max_imgs: int | None) -> List[Path]:
    imgs = [p for p in sorted(images_dir.iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if max_imgs:
        imgs = imgs[:max_imgs]
    if not imgs:
        raise SystemExit(f"No images found in: {images_dir}")
    return imgs


async def amain():
    ap = argparse.ArgumentParser(description="Test Pattern A and B via API")
    ap.add_argument("--base-url", default="http://localhost:8080", help="Service base URL, e.g. https://...a.run.app")
    ap.add_argument("--images-dir", default="/home/tr_user/surveye/ai_validator/data/track", help="Folder of test images")
    ap.add_argument("--pattern", choices=["A", "B"], default="A", help="Which pattern to exercise")
    ap.add_argument("--top-n", type=int, default=2, help="N crops per track (Pattern A)")
    ap.add_argument("--tracks", type=int, default=3, help="# of tracks to simulate (Pattern B)")
    ap.add_argument("--crops-per-track", type=int, default=3, help="Crops per track (Pattern B)")
    ap.add_argument("--max-images", type=int, default=0, help="Limit total images (0 = all)")
    ap.add_argument("--poll-wait-ms", type=int, default=5000, help="Long-poll wait for results")
    ap.add_argument("--interleave-delay-ms", type=int, default=100, help="Delay between interleaved sends (Pattern B)")
    args = ap.parse_args()

    images = collect_images(Path(args.images_dir), args.max_images or None)
    print("Images", images)
    timeout = httpx.Timeout(timeout=60.0, connect=5.0)  # read/write/pool inherit 60s
    async with httpx.AsyncClient(timeout=timeout) as client:
        await health_check(client, args.base_url)

        if args.pattern.upper() == "A":
            # send sequential tracks; last crop of each track sets end_of_track
            await test_pattern_a(
                client,
                args.base_url,
                images,
                top_n=args.top_n,
                poll_wait_ms=args.poll_wait_ms,
            )
        else:
            # interleave multiple tracks and rely on idle timeout flush
            # e.g., with tracks=3 and crops_per_track=3, it sends 9 images interleaved
            await test_pattern_b(
                client,
                args.base_url,
                images,
                tracks=args.tracks,
                crops_per_track=args.crops_per_track,
                interleave_delay_ms=args.interleave_delay_ms,
                poll_wait_ms=args.poll_wait_ms,
            )


if __name__ == "__main__":
    asyncio.run(amain())
