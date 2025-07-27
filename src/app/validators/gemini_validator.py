# import os
# from PIL import Image
# import io
# import google.generativeai as gemini
# import json
import os
import cv2
import ultralytics
from google import genai
from google.genai import types
from PIL import Image
from ultralytics.utils.downloads import safe_download
from ultralytics.utils.plotting import Annotator, colors
import base_validator
# from src.app.validators.base_validator import BaseValidator

gemini.configure(api_key=os.getenv('GOOGLE_API_KEY'))

class GeminiValidator(BaseValidator):
    def validate(self, image: Image.Image, class_name: str) -> dict:
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        buf.seek(0)
        img_bytes = buf.read()

        response = gemini.chat.create(
            model='gemini-1.5',
            prompt=f"Validate the detected class '{class_name}'. Return label and confidence between 0 and 1.",
            images=[{'image_bytes': img_bytes, 'mime_type': 'image/png'}]
        )
        content = response.last.user_response_text.strip().split()
        label = content[1]
        confidence = float(content[-1])
        return {'label': label, 'confidence': confidence}
    
    def object_detection(self, image: Image.Image) -> list:
        """Detect objects in the image using Gemini's object detection capabilities."""

        prompt = """Detect the 2d bounding boxes of objects in image."""

        # Fixed, plotting function depends on this.
        output_prompt = "Return just box_2d and labels, no additional text."

        image, w, h = read_image("gemini-image1.jpg")  # Read img, extract width, height

        results = self.inference(image, prompt + output_prompt)  # Perform inference

        cln_results = json.loads(clean_results(results))  # Clean results, list convert

        annotator = Annotator(image)  # initialize Ultralytics annotator

        for idx, item in enumerate(cln_results):
            # By default, gemini model return output with y coordinates first.
            # Scale normalized box coordinates (0–1000) to image dimensions
            y1, x1, y2, x2 = item["box_2d"]  # bbox post processing,
            y1 = y1 / 1000 * h
            x1 = x1 / 1000 * w
            y2 = y2 / 1000 * h
            x2 = x2 / 1000 * w

            if x1 > x2:
                x1, x2 = x2, x1  # Swap x-coordinates if needed
            if y1 > y2:
                y1, y2 = y2, y1  # Swap y-coordinates if needed

            annotator.box_label([x1, y1, x2, y2], label=item["label"], color=colors(idx, True))

        Image.fromarray(annotator.result())  # display the output