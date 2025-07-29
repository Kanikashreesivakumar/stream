from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()  # Loads variables from .env

gemini_api_key = os.getenv("GEMINI_API_KEY")
print("Gemini API Key:", gemini_api_key)

# Try a simple Gemini API call
genai.configure(api_key=gemini_api_key)

try:
    # List available models
    models = genai.list_models()
    print("Available models:")
    for m in models:
        print(f"- {m.name} (supported methods: {m.supported_generation_methods})")

    # Pick a supported model (replace with a valid model name from the list)
    # Example: 'models/gemini-1.0-pro' or similar
    model_name = "models/gemini-2.5-pro"
    model = genai.GenerativeModel(model_name)
    response = model.generate_content("Say hello!")
    print("Gemini API response:", response.text)
except Exception as e:
    print("Error communicating with Gemini API:", e)

