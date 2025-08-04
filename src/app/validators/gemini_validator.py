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
        # super().__init__(base_model="gemini-2.5-pro")
        self.client = genai.Client(api_key=google_api_key)
        self.base_model ="gemini-2.5-pro"

    def validate(self, image: Image.Image,task: str) -> dict:
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
            raise ValueError(f" Unsupported task: {task}. Allowed: lpr, gender, age")
        
    def _validate_lpr(self, image: Image.Image) -> dict:
        prompt = """
        Detect 2D bounding boxes for all visible license plates.
        """
        output_prompt = """
        Return only 'box_2d', 'label' (plate number from OCR), and 'confidence' (1-100).
        Ignore unclear plates. 
        No extra text.
        Use OCR to extract the plate number.
        Make sure there is no space in the plate number, and no special characters, only numbers and letters.
        """

        try:
            # Step 1: Run inference
            results = self.run_inference(self.client, image, self.base_model, prompt + output_prompt)

            # Step 2: Validate response structure
            if not isinstance(results, dict) or not results.get("success", True):
                return {"success": False, "error": results.get("error", "Unknown LLM error"), "detections": []}

            raw_output = results.get("output")
            if not raw_output:
                return {"success": False, "error": "Empty response from LLM", "detections": []}

            # Step 3: Parse JSON safely
            try:
                cln_results = json.loads(self.clean_results(raw_output))
            except (json.JSONDecodeError, TypeError) as e:
                return {"success": False, "error": f"Invalid JSON from LLM: {e}", "detections": []}

            # Step 4: Validate detection list format
            if not isinstance(cln_results, list):
                return {"success": False, "error": "LLM returned non-list result", "detections": []}

            # ✅ Handle case: No detections
            if len(cln_results) == 0:
                return {"success": True, "detections": [], "annotated_image": image}

            # Step 5: Validate detection objects
            if not all(isinstance(item, dict) and {"box_2d", "label", "confidence"}.issubset(item.keys()) for item in cln_results):
                return {"success": False, "error": "Malformed detection objects from LLM", "detections": []}

            # Step 6: Annotate image with detections
            h, w = image.size
            annotator = Annotator(image)
            for idx, item in enumerate(cln_results):
                y1, x1, y2, x2 = item["box_2d"]
                y1, x1, y2, x2 = (y1 / 1000 * h, x1 / 1000 * w, y2 / 1000 * h, x2 / 1000 * w)
                annotator.box_label([x1, y1, x2, y2], label=item["label"], color=colors(idx, True))

            # ✅ Return successful result
            return {"success": True, "detections": cln_results, "annotated_image": annotator.result()}

        except Exception as e:
            # ✅ Catch unexpected runtime errors
            return {"success": False, "error": f"Unexpected error: {str(e)}", "detections": []}
    
    
    # def _validate_lpr(self, image: Image.Image) -> dict:
    #     prompt = """
    #     Detect 2D bounding boxes for all visible license plates.
    #     """
    #     output_prompt = """
    #     Return only 'box_2d', 'label' (plate number from OCR), and 'confidence' (1-100).
    #     Ignore unclear plates. 
    #     No extra text.
    #     Use OCR to extract the plate number.
    #     Make sure there is no space in the plate number, and no special characters, only numbers and letters.
    #     """

    #     try:
    #         # Call inference
    #         results = self.run_inference(self.client, image, self.base_model, prompt + output_prompt)

    #         # ✅ Ensure the inference call returned a proper structure
    #         if not isinstance(results, dict) or not results.get("success", True):
    #             return {"success": False, "error": results.get("error", "Unknown LLM error")}

    #         raw_output = results.get("output")
    #         if not raw_output:
    #             return {"success": False, "error": "Empty response from LLM"}

    #         # ✅ Try to parse the LLM output as JSON
    #         try:
    #             cln_results = json.loads(self.clean_results(raw_output))
    #         except (json.JSONDecodeError, TypeError) as e:
    #             return {"success": False, "error": f"Invalid JSON from LLM: {e}"}

    #         # ✅ Validate required keys in the detections
    #         if not isinstance(cln_results, list) or not all(
    #             isinstance(item, dict) and {"box_2d", "label", "confidence"}.issubset(item.keys())
    #             for item in cln_results
    #         ):
    #             return {"success": False, "error": "Malformed detection objects from LLM"}

    #         # ✅ Process results only if valid
    #         h, w = image.size
    #         annotator = Annotator(image)
    #         for idx, item in enumerate(cln_results):
    #             y1, x1, y2, x2 = item["box_2d"]
    #             y1, x1, y2, x2 = (y1 / 1000 * h, x1 / 1000 * w, y2 / 1000 * h, x2 / 1000 * w)
    #             annotator.box_label([x1, y1, x2, y2], label=item["label"], color=colors(idx, True))

    #         return {"success": True, "detections": cln_results, "annotated_image": annotator.result()}

    #     except Exception as e:
    #         # ✅ Catch any unexpected runtime errors
    #         return {"success": False, "error": f"Unexpected error: {str(e)}"}
    
    # def _validate_lpr(self, image: Image.Image) -> dict:
    #     prompt = """
    #     Detect 2D bounding boxes for all visible license plates.
    #     """
        
    #     output_prompt = """
    #     Return only 'box_2d',  'label' (plate number from OCR) , and 'confidence' (1-100).
    #     Ignore unclear plates. 
    #     No extra text.
    #     Use OCR to extract the plate number.
    #     Make sure there in no space in the plate number, and no special characters, only number and letters only
           
    #     """
        
    #     h, w = image.size
    #     results = self.run_inference(self.client,image,self.base_model, prompt + output_prompt)
    #     print(results)
    #     cln_results = json.loads(self.clean_results(results["output"]))
    #     print("Cleaned results:", cln_results)
    #     annotator = Annotator(image)
    #     for idx, item in enumerate(cln_results):
    #         y1, x1, y2, x2 = item["box_2d"]
    #         y1, x1, y2, x2 = (y1 / 1000 * h, x1 / 1000 * w, y2 / 1000 * h, x2 / 1000 * w)
            
    #         # if x1 > x2:
    #         #     x1, x2 = x2, x1  # Swap x-coordinates if needed
    #         # if y1 > y2:
    #         #     y1, y2 = y2, y1  # Swap y-coordinates if needed
            
    #         annotator.box_label([x1, y1, x2, y2], label=item["label"], color=colors(idx, True))

    #     return {"detections": cln_results[0], "annotated_image": annotator.result()}
    
    # LPR: Detect and OCR license plates
    def _validate_lpd(self, image: Image.Image) -> dict:
        prompt = """
        Detect 2D bounding boxes for all visible license plates.
        """
        
        output_prompt = """
        Return only 'box_2d',  'label' (plate number from OCR) , and 'confidence' (1-100).
        Ignore unclear plates. 
        No extra text.
        Use OCR to extract the plate number.
        Make sure there in no space in the plate number, and no special characters, only number and letters only
           
        """
        
        h, w = image.size
        results = self.inference(image, prompt + output_prompt)
        
        cln_results = json.loads(self.clean_results(results))
        print("Cleaned results:", cln_results)
        annotator = Annotator(image)
        for idx, item in enumerate(cln_results):
            y1, x1, y2, x2 = item["box_2d"]
            y1, x1, y2, x2 = (y1 / 1000 * h, x1 / 1000 * w, y2 / 1000 * h, x2 / 1000 * w)
            
            # if x1 > x2:
            #     x1, x2 = x2, x1  # Swap x-coordinates if needed
            # if y1 > y2:
            #     y1, y2 = y2, y1  # Swap y-coordinates if needed
            
            annotator.box_label([x1, y1, x2, y2], label=item["label"], color=colors(idx, True))

        return {"detections": cln_results[0], "annotated_image": annotator.result()}

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
