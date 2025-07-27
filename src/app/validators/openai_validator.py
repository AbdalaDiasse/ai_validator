import os
import io
from PIL import Image
import openai
from src.app.validators.base_validator import BaseValidator

openai.api_key = os.getenv('OPENAI_API_KEY')

class OpenAIValidator(BaseValidator):
    def validate(self, image: Image.Image, class_name: str) -> dict:
        # Prepare image bytes
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        buf.seek(0)

        # Call OpenAI Vision API via ChatCompletion with attachment
        response = openai.ChatCompletion.create(
            model='gpt-4o',
            messages=[
                {'role': 'system', 'content': f"You are a vision model that validates '{class_name}' detection. Provide the actual label and a confidence between 0 and 1."},
                {'role': 'user', 'content': 'Validate the following image.', 'image': {'data': buf.read(), 'mimetype': 'image/png'}}
            ]
        )
        text = response.choices[0].message.content.strip().split()
        # Example parse: "label: ABC123 confidence: 0.85"
        label = text[1]
        confidence = float(text[-1])
        return {'label': label, 'confidence': confidence}