import requests
import os

API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "qwen/qwen-2.5-72b-instruct" 

def extract_door_schedule(ocr_text):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    system_prompt = (
    "You are a helpful assistant. The following text was extracted via OCR from a scanned architectural document. "
    "Your task is to extract ONLY the DOOR SCHEDULE table, if it exists. "
    "Do NOT include the WINDOW SCHEDULE. "
    "For each door, calculate the door area (in square meters) based on the width and height in inches. "
    "Then estimate material (based on FINISH/REMARKS) and assign a price using the following sample rates:\n"
    "- wood: $120/sqm + $50 install\n"
    "- glass: $225/sqm + $100 install\n"
    "- metal: $180/sqm + $75 install\n"
    "- aluminum: $150/sqm + $60 install\n"
    "- pvc: $90/sqm + $40 install\n"
    "Return the DOOR SCHEDULE table in Markdown format, and include a new column 'EST. COST'.")

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ocr_text}
        ]
    }

    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None

