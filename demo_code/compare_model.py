import os
import ollama
import google.generativeai as genai
from dotenv import load_dotenv

# Load Gemini API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def get_gemini_response(prompt: str) -> str:
    try:
        model = genai.GenerativeModel(model_name='models/gemini-1.5-pro-latest')
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"[Gemini Error] {e}"

def get_mistral_response(prompt: str) -> str:
    try:
        response = ollama.chat(model='mistral', messages=[
            {'role': 'user', 'content': prompt}
        ])
        return response['message']['content'].strip()
    except Exception as e:
        return f"[Mistral Error] {e}"

def main():
    prompt = input("Enter your question: ")

    print("\n🧠 Getting response from Gemini...")
    gemini_response = get_gemini_response(prompt)

    print("\n🖥 Getting response from Mistral...")
    mistral_response = get_mistral_response(prompt)

    print("\n================= 🔍 COMPARISON =================")
    print(f"\n📘 Gemini:\n{gemini_response}")
    print(f"\n💻 Mistral:\n{mistral_response}")
    print("\n=================================================")

if __name__ == "__main__":
    main()
