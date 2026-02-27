"""Simple script to verify OpenAI API key is working. Create .env and put API key there."""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key or api_key == "your-api-key-here":
    print("Error: Set OPENAI_API_KEY in .env file")
    exit(1)

client = OpenAI(api_key=api_key)

try:
    response = client.chat.completions.create(
        temperature=0.0,
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say 'API key works!' and nothing else."}],
        max_tokens=10,
    )
    print("Success:", response.choices[0].message.content.strip())
except Exception as e:
    print("Error:", e)
