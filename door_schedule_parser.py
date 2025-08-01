import os
import re
import requests
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
MODEL = "qwen/qwen-2.5-72b-instruct"

def calculate_area_sqm(width_cm: float, height_cm: float) -> float:
    return round((width_cm / 100) * (height_cm / 100), 3)

class TavilyPriceSearcher:
    def __init__(self, region: str = "usa"):
        from tavily import TavilyClient
        self.api_key = TAVILY_API_KEY
        self.client = TavilyClient(api_key=self.api_key)
        self.region = region
        self.price_cache = {}

    def search_material_prices(self, material: str) -> dict:
        key = f"{material.lower()}_{self.region.lower()}"
        if key in self.price_cache:
            return self.price_cache[key]

        query = f"{material} door price per square meter + installation price in USD {self.region} 2025"
        response = self.client.search(query=query, search_depth="advanced", max_results=5)

        sqm_prices, install_prices = [], []
        for result in response.get("results", []):
            content = result.get("content", "").lower()
            sqm_prices += [float(m[0]) for m in re.findall(r'(\d{2,4})\s*\$\s*per\s*(?:sqm|square meter|מ״ר)', content)]
            install_prices += [float(m) for m in re.findall(r'installation[:\s]+\$(\d{2,4})', content)]

        avg_price = round(sum(sqm_prices) / len(sqm_prices), 2) if sqm_prices else 150
        avg_install = round(sum(install_prices) / len(install_prices), 2) if install_prices else 60

        result = {
            "price_per_sqm": avg_price,
            "installation": avg_install,
            "timestamp": datetime.now().isoformat()
        }
        self.price_cache[key] = result
        return result

def extract_door_schedule_json(ocr_text: str) -> list:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    system_prompt = (
        "You are a helpful assistant. The following text was extracted via OCR from a scanned architectural document. "
        "Your task is to extract ONLY the DOOR SCHEDULE table, if it exists. "
        "Do NOT include the WINDOW SCHEDULE. "
        "Return the door schedule as a JSON array of objects, each object containing the fields: "
        "'door_id' (string), 'count' (integer), 'width_cm' (number), 'height_cm' (number), "
        "'operation' (string), 'finish' (string), 'remarks' (string). "
        "Assume that all dimensions are in centimeters (cm), unless explicitly stated otherwise."
    )

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ocr_text}
        ]
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        content = response.json()["choices"][0]["message"]["content"]
        try:
            json_match = re.search(r"```json(.*?)```", content, re.DOTALL)
            if json_match:
                json_text = json_match.group(1).strip()
            else:
                json_match = re.search(r"(\[.*\])", content, re.DOTALL)
                if json_match:
                    json_text = json_match.group(1).strip()
                else:
                    raise ValueError("No JSON found in LLM response")

            data = json.loads(json_text)
            return data
        except Exception as e:
            print("Error decoding JSON from LLM response:")
            print(content)
            print("Exception:", e)
            return []
    else:
        print(f"Error {response.status_code}: {response.text}")
        return []

def calculate_costs_and_augment(data: list, price_searcher: TavilyPriceSearcher) -> list:
    for door in data:
        try:
            width_cm = float(door["width_cm"])
            height_cm = float(door["height_cm"])
            count = int(door.get("count", 1))
            finish = door.get("finish", "").lower()

            area = calculate_area_sqm(width_cm, height_cm)
            pricing = price_searcher.search_material_prices(finish)
            total_cost = round(count * (area * pricing["price_per_sqm"] + pricing["installation"]), 2)

            door["area_sqm"] = area
            door["price_per_sqm"] = pricing["price_per_sqm"]
            door["installation_cost"] = pricing["installation"]
            door["total_cost"] = total_cost
        except Exception as e:
            print(f"Error processing door {door.get('door_id', 'unknown')}: {e}")
            door["area_sqm"] = None
            door["total_cost"] = None
    return data

if __name__ == "__main__":
    with open("ocr_text.txt", "r", encoding="utf-8") as f:
        ocr_text = f.read()

    doors_json = extract_door_schedule_json(ocr_text)
    if not doors_json:
        print("No door schedule data extracted.")
    else:
        searcher = TavilyPriceSearcher()
        doors_with_costs = calculate_costs_and_augment(doors_json, searcher)
        print(json.dumps(doors_with_costs, indent=2, ensure_ascii=False))