# test_nvidia_attributes.py
import base64
import json
import requests
from pathlib import Path
import os
project_root = Path(__file__).resolve().parent.parent
print("Project root:", project_root)
IMAGE_PATH = os.path.join(project_root, "data","attributes", "car.jpeg")

# URL = "http://localhost:8080/validate/attributes"
URL = "https://ai-validator-654942414948.europe-west9.run.app/validate/attributes"
def send_request(detection_type="body", bbox=None):
    with open(IMAGE_PATH, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "image_base64": b64,
        "detection_type": detection_type
    }
    if bbox:
        payload["bbox"] = bbox  # example: {"x":100, "y":50, "width":200, "height":200}

    api_key = os.environ.get("VALIDATOR_API_KEY")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    r = requests.post(URL, json=payload, headers=headers, timeout=120)
    print("status:", r.status_code)
    try:
        print(json.dumps(r.json(), indent=2))
    except Exception:
        print(r.text)

if __name__ == "__main__":
    # test the body extractor
    send_request("car")
    # test the car extractor with an optional bbox
    # send_request("face")