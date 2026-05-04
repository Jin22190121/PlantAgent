import google.generativeai as genai
import os

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

models = genai.list_models()

for m in models:
    print(m.name)