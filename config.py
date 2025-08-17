import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///metrics.db')
    
    # Model fallback chain - ordered from most powerful to least
    MODELS = [
        'gemini-2.0-flash',
        'gemini-2.5-flash',
        'gemini-2.5-flash-lite',
    ]
    
    # Default model to try first
    MODEL_NAME = MODELS[0]
    
    # Validate API key
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set in environment variables")