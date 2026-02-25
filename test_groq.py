import os
import httpx

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    # Try reading from .env
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
try:
    response = httpx.get("https://api.groq.com/openai/v1/models", headers=headers)
    models = response.json().get("data", [])
    vision_models = [m["id"] for m in models if "vision" in m["id"].lower()]
    print("Vision models available:", vision_models)
    
    # Let's test calling llama-3.2-11b-vision-preview without an image just to see if it works
    resp = httpx.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json={
        "model": "llama-3.2-11b-vision-preview",
        "messages": [{"role": "user", "content": "hello"}]
    })
    print("Response status:", resp.status_code)
except Exception as e:
    print("Error:", e)
