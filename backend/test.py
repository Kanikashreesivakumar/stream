from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()  

gemini_api_key = os.getenv("GEMINI_API_KEY")
print("Gemini API Key:", gemini_api_key)


genai.configure(api_key=gemini_api_key)

try:
    
    models = genai.list_models()
    print("Available models:")
    for m in models:
        print(f"- {m.name} (supported methods: {m.supported_generation_methods})")

   
    model_name = "models/gemini-2.5-pro"
    model = genai.GenerativeModel(model_name)
    response = model.generate_content("Say hello!")
    print("Gemini API response:", response.text)
except Exception as e:
    print("Error communicating with Gemini API:", e)

