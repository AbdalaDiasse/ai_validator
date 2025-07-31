import base64
import json
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

# ==== Configuration ====
API_URL = "http://localhost:8000/validate"
IMAGE_PATH = "/home/tr_user/surveye/ai_validator/data/trafic2.jpg"

# Bounding box (example)
bbox = {"x": 1211, "y": 743, "width": 109, "height": 38}
class_name = "AA 757 GZ"

# ==== 1. Read and encode image to base64 ====
with open(IMAGE_PATH, "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

# ==== 2. Prepare payload ====
payload = {
    "image_base64": image_base64,
    "bbox": bbox,
    "task": "lpr",  # License Plate Recognition
    "validators": ["gemini"]
}

# ==== 3. Send request ====
response = requests.post(API_URL, headers={"Content-Type": "application/json"}, data=json.dumps(payload))

if response.status_code != 200:
    print("❌ Request failed:", response.text)
    exit()

result = response.json()
print("✅ API Response:", json.dumps(result, indent=2))

# ==== 4. Draw bounding box & label ====
# Load the original image
img = Image.open(IMAGE_PATH).convert("RGB")
draw = ImageDraw.Draw(img)

# Draw rectangle
x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
draw.rectangle([x, y, x + w, y + h], outline="red", width=3)

# Draw label text from one of the validator results
label = list(result.values())[0]["label"]  # get label from first validator
# Load a TTF font with desired size (you can use any .ttf available on your system)
try:
    font = ImageFont.truetype("arial.ttf", 10)  # font size = 24
except IOError:
    font = ImageFont.load_default()  # fallback if font not found
draw.text((x, y - 30), label, font=font, fill="yellow")

# ==== 5. Show image ====
img.save("output.jpg")
