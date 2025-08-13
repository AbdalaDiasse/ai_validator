from pydantic_settings import BaseSettings
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, '.env')

class Settings(BaseSettings):
    openai_api_key: str
    google_api_key: str
    nvidia_nim_endpoint: str
    nvidia_nim_api_key: str

    class Config:
        env_file = '.env'

settings = Settings()