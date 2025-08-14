from pydantic_settings import BaseSettings
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, '.env')

class Settings(BaseSettings):
    openai_api_key: str
    google_api_key: str
    nvidia_nim_endpoint: str
    nvidia_nim_api_key: str
    top_n: int = 3
    idle_ms: int = 500
    mode: str = "pattern_a"  # or "pattern_b"
    max_wait_ms: int = 1000
    max_imgs: int = 8
    max_parallel_calls: int = 4
    

    class Config:
        env_file = '.env'

settings = Settings()