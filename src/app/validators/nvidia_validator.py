import os
import requests
from PIL import Image
import io
from src.app.validators.base_validator import BaseValidator

class NvidiaValidator(BaseValidator):
    def __init__(self):
        self.endpoint = os.getenv('NVIDIA_NIM_ENDPOINT')
        self.api_key = os.getenv('NVIDIA_NIM_API_KEY')

    def validate(self, image: Image.Image, class_name: str) -> dict:
        buf = io.BytesIO()
        image.save(buf, format='JPEG')
        buf.seek(0)
        files = {'file': ('crop.jpg', buf, 'image/jpeg')}
        data = {'class_name': class_name}
        headers = {'Authorization': f'Bearer {self.api_key}'}
        r = requests.post(self.endpoint, files=files, data=data, headers=headers)
        r.raise_for_status()
        resp = r.json()
        # Expecting {'label': '...', 'confidence': 0.82}
        return resp