import google.generativeai as genai
import os
from dotenv import load_dotenv
from PIL import Image

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def describe_screenshot(image_path):
    model = genai.GenerativeModel("models/gemini-2.5-flash")
    img = Image.open(image_path)
    prompt = "Look at this page from an AWS whitepaper. Does it contain the 'AWS Shared Responsibility Model' diagram? If so, describe it in detail. If not, what is on this page?"
    
    response = model.generate_content([prompt, img])
    print(response.text)

if __name__ == "__main__":
    describe_screenshot("page_17_screenshot.png")
