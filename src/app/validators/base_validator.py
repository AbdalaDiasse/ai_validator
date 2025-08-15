from abc import ABC, abstractmethod
from PIL import Image
from google import genai
from google.genai import types
from google import genai
from google.genai.types import GenerateContentResponse
from google.api_core.exceptions import GoogleAPIError
import json
import re

PLATE_REGEX = re.compile(r"^[A-Z]{2}[0-9]{3,4}[A-Z]{1,2}$")
class BaseValidator(ABC):
    def __init__(self,base_model):
        """
        Base class for image validators.
        Subclasses should implement the validate method.
        """
        self.base_model = base_model
        


    def clean_plate(self,label: str) -> str | None:
        """Keep only A-Z0-9, enforce Senegalese plate regex."""
        clean = re.sub(r'[^A-Z0-9]', '', label.upper())
        return clean if PLATE_REGEX.match(clean) else None


    def clean_results(self, results):
        """
        Accepts:
        - str containing JSON (with/without ```json fences)
        - dict with 'results' or 'detections' list
        - list of detection dicts
        Returns: list of {"index": int, "label": str, "confidence": int}
        """
        try:
            # 1) Normalize to a Python object
            if isinstance(results, str):
                s = results.strip()
                if s.startswith("```"):
                    # strip leading/trailing code fences like ```json ... ```
                    s = s.lstrip("`")
                    # remove an optional 'json' language tag
                    if s.startswith("json"):
                        s = s[4:]
                    s = s.rstrip("`").strip()
                obj = json.loads(s)
            else:
                obj = results  # could already be dict/list from SDK

            # 2) Extract a list
            if isinstance(obj, dict):
                items = obj.get("results")
                if items is None:
                    items = obj.get("detections")
                if items is None:
                    # Some models might return {"label":..., "confidence":...}
                    # normalize that to a single-item list
                    items = [obj] if {"label","confidence"} <= set(obj.keys()) else []
            elif isinstance(obj, list):
                items = obj
            else:
                items = []

            # 3) Validate and normalize entries
            out = []
            for i, item in enumerate(items):
                if not isinstance(item, dict):
                    continue
                label = self.clean_plate(item.get("label", ""))
                if not label:
                    continue
                try:
                    conf = int(item.get("confidence", 1))
                except Exception:
                    conf = 1
                conf = max(1, min(conf, 100))
                idx = item.get("index", i)
                out.append({"index": idx, "label": label, "confidence": conf})
            print("Cleaned results:", out)
            return out
        except Exception as e:
            print(f"❌ Failed to parse/clean results: {e}")
            return []


    # def clean_results(self, results: str):
    #     """Clean raw model output, enforce JSON and Senegal plate constraints."""
    #     try:
    #         # Step 1: Remove markdown fences
    #         print("Raw results:", results)
    #         cleaned = results.strip().removeprefix("```json").removesuffix("```").strip()
    #         print("Cleaned results:", cleaned)
    #         # Step 2: Attempt to load as JSON
    #         parsed = json.loads(cleaned)
    #         print("Parsed results:", type(parsed))
    #         print("item 0:", parsed[0])
    #         # Step 3: Validate labels with regex
    #         valid_outputs = []
    #         for item in parsed:
    #             label = item.get("label", "")
    #             valid_label = self.clean_plate(label)
    #             if valid_label:
    #                 # Replace label with cleaned version
    #                 item["label"] = valid_label
    #                 valid_outputs.append(item)
    #         print("Valid outputs:", valid_outputs)
    #         return valid_outputs

    #     except Exception as e:
    #         print(f"❌ Failed to parse/clean results: {e}")
    #         return []
    
    # def clean_results(self,results):
    #     """Clean the results for visualization."""
    #     return results.strip().removeprefix("```json").removesuffix("```").strip()
    
    @abstractmethod
    def validate(self, image: Image.Image, class_name: str) -> dict:
        """
        Validate the given cropped image for the predicted class.
        Returns a dict: { 'label': str, 'confidence': float }
        Confidence should be between 0.0 and 1.0.
        """
        pass
    
    async def run_inference(self,client: genai.Client, image ,model_name: str, prompt: str,temp=0.5):
        try:
            # Perform the inference call
            response: GenerateContentResponse = await client.aio.models.generate_content(
                model=model_name,
                contents=[prompt, image],  # Provide both the text prompt and image as input,
                config=types.GenerateContentConfig(
                    temperature=temp,  # Controls creativity vs. determinism in output
                ),
            )

            # Check if response has error
            if hasattr(response, "error") and response.error:
                return {"success": False, "error": response.error.message}

            # Check if candidates exist
            if not response.candidates:
                return {"success": False, "error": "No candidates returned by the model"}

            # Extract text output
            output_text = response.candidates[0].content.parts[0].text
            return {"success": True, "output": output_text}

        except GoogleAPIError as e:
            # Catch Google API errors
            return {"success": False, "error": f"API error: {str(e)}"}

        except Exception as e:
            # Catch unexpected errors
            return {"success": False, "error": f"Unexpected error: {str(e)}"}
    
    def inference2(self,image, prompt, temp=0.5):
        """
        Performs inference using Google Gemini 2.5 Pro Experimental model.

        Args:
            image (str or genai.types.Blob): The image input, either as a base64-encoded string or Blob object.
            prompt (str): A text prompt to guide the model's response.
            temp (float, optional): Sampling temperature for response randomness. Default is 0.5.

        Returns:
            str: The text response generated by the Gemini model based on the prompt and image.
        """
        response = self.client.models.generate_content(
            # model="gemini-2.5-flash-preview-05-20",  # or "gemini-2.5-pro-exp-03-25"
            model="gemini-2.5-pro",
            # model="gemini-2.0-flash",
            # model="gemini-2.5-flash",
            contents=[prompt, image],  # Provide both the text prompt and image as input
            config=types.GenerateContentConfig(
                temperature=temp,  # Controls creativity vs. determinism in output
            ),
        )
        
        print("gemini response: ", response)

        return response.text  # Return the generated textual response 