# import os
# from PIL import Image
# import io
# import google.generativeai as gemini
# import json
import os
import cv2
import json
import ultralytics
from google import genai
from google.genai import types
from PIL import Image
from ultralytics.utils.downloads import safe_download
from ultralytics.utils.plotting import Annotator, colors
from src.app.validators.base_validator import BaseValidator
from src.app.settings import settings
google_api_key = settings.google_api_key


class GeminiValidator(BaseValidator):
    def __init__(self):
        super().__init__(base_model="gemini-2.5-pro")
        self.client = genai.Client(api_key=google_api_key)
        

    def validate(self, image: Image.Image, class_name: str, task: str) -> dict:
        """
        Dispatch validation based on the task: 'lpr', 'gender', or 'age'.
        """
        if task == "lpr":
            return self._validate_lpr(image)
        # elif task == "gender":
        #     return self._validate_gender(image, class_name)
        # elif task == "age":
        #     return self._validate_age(image, class_name)
        else:
            raise ValueError(f"❌ Unsupported task: {task}. Allowed: lpr, gender, age")

    # LPR: Detect and OCR license plates
    def _validate_lpr(self, image: Image.Image) -> dict:
        prompt = """
        Detect 2D bounding boxes for all visible license plates.
        """
        output_prompt = """
        Return only 'box_2d' and 'label' (plate number from OCR). 
        Ignore unclear plates. No extra text.
        """
        h, w = image.shape[:2]
        results = self.inference(image, prompt + output_prompt)
        cln_results = json.loads(self.clean_results(results))

        annotator = Annotator(image)
        for idx, item in enumerate(cln_results):
            y1, x1, y2, x2 = item["box_2d"]
            y1, x1, y2, x2 = (y1 / 1000 * h, x1 / 1000 * w, y2 / 1000 * h, x2 / 1000 * w)
            annotator.box_label([x1, y1, x2, y2], label=item["label"], color=colors(idx, True))

        return {"detections": cln_results, "annotated_image": annotator.result()}

    # # Gender Classification
    # def _validate_gender(self, image: Image.Image, class_name: str) -> dict:
    #     return self._run_gemini_validator(image, class_name, task_type="gender")

    # # Age Estimation
    # def _validate_age(self, image: Image.Image, class_name: str) -> dict:
    #     return self._run_gemini_validator(image, class_name, task_type="age")

    # # Shared Gemini Validation Logic for gender/age
    # def _run_gemini_validator(self, image: Image.Image, class_name: str, task_type: str) -> dict:
    #     buf = io.BytesIO()
    #     image.save(buf, format='PNG')
    #     img_bytes = buf.getvalue()

    #     response = gemini.chat.create(
    #         model='gemini-1.5',
    #         prompt=f"Validate the detected class '{class_name}' for {task_type}. "
    #                f"Return label (predicted {task_type}) and confidence (0-1).",
    #         images=[{'image_bytes': img_bytes, 'mime_type': 'image/png'}]
    #     )

    #     content = response.last.user_response_text.strip().split()
    #     label = content[1]
    #     confidence = float(content[-1])
        
        
        return {"label": label, "confidence": confidence}
