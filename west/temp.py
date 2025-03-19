from google import generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

print("Available Gemini Models:")
for model in genai.list_models():
    print(f"Model Name: {model.name}")
    print(f"  Description: {model.description}")
    # Try to access supported generation methods
    if hasattr(model, 'supported_generation_methods'):
        print(f"  Supported Methods: {model.supported_generation_methods}")
    elif hasattr(model, 'supported_methods'):
        print(f"  Supported Methods: {model.supported_methods}")
    else:
        print("  Could not determine supported methods.")
    print("---")