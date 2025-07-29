from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()  # Loads variables from .env

gemini_api_key = os.getenv("GEMINI_API_KEY")
print("Gemini API Key:", gemini_api_key)

# Try a simple Gemini API call
genai.configure(api_key=gemini_api_key)

try:
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content("Say hello!")
    print("Gemini API response:", response.text)
except Exception as e:
    print("Error communicating with Gemini API:", e)

