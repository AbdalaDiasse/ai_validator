# VLM Validator Service

This project provides a FastAPI-based service for validating outputs from detection models (e.g., license plate, age, gender) using multiple AI validators (OpenAI, Gemini, Nvidia).

## Features

- Validate detection results using multiple AI backends
- Supports image cropping and bounding box handling
- Extensible validator architecture

## Setup locally

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

## Docker

1. **build image **
   ```bash
   git clone <your-repo-url>
   cd ai_validator
   docker build -t vlm-validator:dev .
   ```

2. **Run container **
   ```bash
   docker run -it --rm -p 8080:8080 -v .env:/app/.env  --name ai-validator ai-validator:dev
   ```


## Google Cloud Plateform

1. **Install gcloud**
   ```bash
   # Add the Google Cloud SDK distribution URI as a package source
   sudo apt-get install apt-transport-https ca-certificates gnupg curl -y
   echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
   | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list

   # Import the Google Cloud public key
   curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
   | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg

   # Update and install the CLI
   sudo apt-get update && sudo apt-get install google-cloud-cli -y
   # Once installed run to initialize gcloud
   gcloud init
   ```
2. **Make sure the following services are activated**
   ```bash
   gcloud services enable \
      artifactregistry.googleapis.com \
      run.googleapis.com \
      cloudbuild.googleapis.com \
      secretmanager.googleapis.com
   ```
3. **Push the image to  Registry**
   ```bash
   gcloud auth configure-docker us-central1-docker.pkg.dev

   # We have two Options 
   # Option 1 :
   docker tag ai-validator:dev us-central1-docker.pkg.dev/surveye-468818/serveye/ai-validator:dev
   docker push us-central1-docker.pkg.dev/surveye-468818/serveye/ai-validator:dev 

   # Option 2 :
   # if we want to call also  use builds submit which will build directly to cloud build
   cd <PROJECT_DIR> # Make sure you have Dockefile in it
   gcloud builds submit --tag us-central1-docker.pkg.dev/surveye-468818/serveye/ai-validator:dev

   ```

4. **Store gemini API key as secrete**
   ```bash
      echo -n "<YOUR_GEMINI_API_KEY>" | \
      gcloud secrets create APP_GOOGLE_API_KEY --data-file=-
   ```
5. **Deploy to cloud run**

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