# VLM Validator Service

This project provides a FastAPI-based service for validating outputs from detection models (e.g., license plate, age, gender) using multiple AI validators (OpenAI, Gemini, Nvidia).

## Features

- Validate detection results using multiple AI backends
- Supports image cropping and bounding box handling
- Extensible validator architecture

## Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd ai_validator
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   - Copy `.env.example` to `.env` and fill in your API keys (OpenAI, Gemini, Nvidia, etc.).

4. **Start the API server**
   ```bash
   uvicorn src.app.main:app --host 0.0.0.0 --port 8000
   ```

## API Usage

### Endpoint

`POST /validate`

### Request Body

```json
{
  "image_base64": "<base64-encoded-image>",
  "bbox": { "x": 1211, "y": 743, "width": 109, "height": 38 },
  "task": "lpr",  
  "validator": "gemini" 
}
```

### Example Python Request

```python
import base64
import json
import requests

API_URL = "http://localhost:8000/validate"
IMAGE_PATH = "data/trafic2.jpg"

with open(IMAGE_PATH, "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "image_base64": image_base64,
    "bbox": {"x": 1211, "y": 743, "width": 109, "height": 38},
    "task": "lpr",
    "validator": "gemini"
}

response = requests.post(API_URL, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
print(response.json())
```

### Response

```json
{
  "gemini": {
    "label": "AA 757 GZ",
    "confidence": 95
  }
}
```

## Testing

Run unit tests with:

```bash
pytest --maxfail=1 --disable-warnings -q
```

## Project Structure

- `src/app/main.py` – FastAPI app and endpoint
- `src/app/validators/` – Validator implementations
- `src/app/models.py` – Pydantic models
- `src/app/settings.py` – Configuration loader
- `test/` – Test scripts and sample requests

##