# ai_validator
VLM based module used to validated the output of the B4H detection , the validator consist of :
1. Age validator
2. Gender validator
3. License plate Validator
Model used
1. Gemini



## Setup

1. Copy `.env.example` to `.env` and fill in your API keys.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt


##Source en:

source /home/tr_user/miniconda3/bin/activate gemini

## To test :
pytest --maxfail=1 --disable-warnings -q