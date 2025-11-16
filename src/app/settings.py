from pydantic_settings import BaseSettings
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, '.env')
print(ENV_PATH)

class Settings(BaseSettings):
    openai_api_key: str
    google_api_key: str
    nvidia_nim_endpoint: str
    nvidia_nim_api_key: str
    validator_api_key: str = os.getenv("VALIDATOR_API_KEY", "your-secret-key")

    class Config:
        env_file = '.env'

settings = Settings()