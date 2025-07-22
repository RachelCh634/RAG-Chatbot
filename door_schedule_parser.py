import re
from typing import List, Dict, Optional
import logging

MATERIAL_PRICES: Dict[str, Dict[str, float]] = {
    "wood": {"price_per_sqm": 800, "installation": 200},
    "wooden": {"price_per_sqm": 800, "installation": 200},
    "metal": {"price_per_sqm": 1200, "installation": 300},
    "steel": {"price_per_sqm": 1200, "installation": 300},
    "glass": {"price_per_sqm": 1500, "installation": 400},
    "aluminum": {"price_per_sqm": 1000, "installation": 250},
    "pvc": {"price_per_sqm": 600, "installation": 150},
    "composite": {"price_per_sqm": 900, "installation": 220},
    "default": {"price_per_sqm": 1000, "installation": 250}
}

LABOR_COST_PERCENTAGE: float = 0.25

def calculate_door_area(size_str: str) -> float:
    """
    Calculates the door area in square meters from a dimension string (cm), for example: "90x210" or "90×210".
    Returns 0.0 if unable to interpret.
    """
    try:
        size_str = size_str.replace('×', 'x').replace('X', 'x').strip()
        match = re.match(r'^\s*(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*$', size_str)
        if not match:
            logging.warning(f"Could not parse size string: '{size_str}'")
            return 0.0

        width_cm, height_cm = map(float, match.groups())
        area_sqm = (width_cm / 100) * (height_cm / 100)
        return round(area_sqm, 3)
    except Exception as e:
        logging.error(f"Error calculating door area for '{size_str}': {e}")
        return 0.0

def get_material_pricing(material: str) -> Dict[str, float]:
    """
    Returns material and installation prices by material type.
    If not found - returns default prices.
    """
    material_key = material.lower().strip()
    if material_key in MATERIAL_PRICES:
        return MATERIAL_PRICES[material_key]

    for key in MATERIAL_PRICES.keys():
        if key in material_key or material_key in key:
            return MATERIAL_PRICES[key]

    logging.warning(f"Material '{material}' not found, using default pricing")
    return MATERIAL_PRICES["default"]

def calculate_door_cost(door_info: Dict) -> Dict[str, float]:
    """
    Calculates cost breakdown by door area and material.
    Returns a dictionary with material, labor, installation, and total cost.
    """
    try:
        area_sqm = door_info.get("area_sqm", 0)
        material = door_info.get("material", "default")
        
        if area_sqm <= 0:
            return {
                "material_cost": 0,
                "labor_cost": 0,
                "installation_cost": 0,
                "total_cost": 0,
                "error": "Invalid area"
            }

        pricing = get_material_pricing(material)
        material_cost = area_sqm * pricing["price_per_sqm"]
        labor_cost = material_cost * LABOR_COST_PERCENTAGE
        installation_cost = pricing["installation"]
        total_cost = material_cost + labor_cost + installation_cost

        result = {
            "material_cost": round(material_cost, 2),
            "labor_cost": round(labor_cost, 2),
            "installation_cost": round(installation_cost, 2),
            "total_cost": round(total_cost, 2),
            "price_per_sqm": pricing["price_per_sqm"]
        }
        return result
    except Exception as e:
        logging.error(f"Error calculating cost for door {door_info}: {e}")
        return {
            "material_cost": 0,
            "labor_cost": 0,
            "installation_cost": 0,
            "total_cost": 0,
            "error": str(e)
        }

def parse_door_schedule(text: str) -> List[Dict]:
    """
    Extracts a list of doors from raw OCR text or a report.
    Searches for a "Door Schedule" section and parses door rows.
    Returns a list of dictionaries with information and doors including cost calculations.
    """
    doors: List[Dict] = []
    try:
        schedule_patterns = [r'Door\s+Schedule', r'DOOR\s+SCHEDULE', r'דלתות.*לוח', r'לוח.*דלתות']
        schedule_start: Optional[int] = None
        for i, pattern in enumerate(schedule_patterns):
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                schedule_start = match.end()
                break
            else:
                print(f"Pattern {i} not found: '{pattern}'")

        if schedule_start is None:
            door_mentions = re.findall(r'\b[Dd]oor\b', text)
            print(f"Found {len(door_mentions)} mentions of 'door' in text")
            logging.warning("Door Schedule section not found in text")
            return doors

        relevant_text = text[schedule_start:]
        
        lines = [line.strip() for line in relevant_text.splitlines() if line.strip()]

        door_row_patterns = [
            re.compile(r'^(D[-_]?\d+)\s*\|\s*([\d×x*X\s]+)\s*\|\s*([\w\s\-א-ת]+?)\s*\|\s*(.+)?$', re.IGNORECASE),
            re.compile(r'^([Ddד][-_]?\d+)\s*\|\s*([\d×x*X\s]+)\s*\|\s*([\w\s\-א-ת]+?)\s*\|\s*(.+)?$', re.IGNORECASE),
            re.compile(r'^(D[-_]?\d+)\s+([\d×x*X\s]+)\s+([\w\s\-א-ת]+?)(?:\s+(.+))?$', re.IGNORECASE),
            re.compile(r'^([Ddד][-_]?\d+)\s+([\d×x*X\s]+)\s+([\w\s\-א-ת]+?)(?:\s+(.+))?$', re.IGNORECASE)
        ]

        for line_num, line in enumerate(lines):
            
            if (re.match(r'^[=\-_\s]*$', line) or 
                'door' in line.lower() or 
                'schedule' in line.lower() or 
                'door id' in line.lower() or
                ('size' in line.lower() and 'material' in line.lower())):
                continue

            door_found = False
            for pattern_num, pattern in enumerate(door_row_patterns):
                match = pattern.match(line)
                if match:
                    door_id = match.group(1).strip()
                    size = match.group(2).strip()
                    material = match.group(3).strip()
                    notes = match.group(4).strip() if match.group(4) else ""

                    area = calculate_door_area(size)
                    if area > 0:
                        door_info = {
                            "door_id": door_id,
                            "size": size,
                            "material": material,
                            "notes": notes,
                            "area_sqm": area
                        }
                        cost_breakdown = calculate_door_cost(door_info)
                        door_info["cost_breakdown"] = cost_breakdown

                        doors.append(door_info)
                    else:
                        print(f"Skipped door {door_id} due to invalid area")
                    door_found = True
                    break
                else:
                    print(f"Pattern {pattern_num} did not match")
                    
            if not door_found and len(line) > 10:
                logging.info(f"Could not parse potential door line: {line}")

    except Exception as e:
        logging.error(f"Error parsing door schedule: {e}")
    
    return doors

def generate_door_summary(doors: List[Dict]) -> Dict:
    """
    Creates a total summary of doors with costs and totals by material. 
    """
    if not doors:
        return {"total_doors": 0, "total_cost": 0, "total_area": 0}

    total_cost = sum(door.get("cost_breakdown", {}).get("total_cost", 0) for door in doors)
    total_area = sum(door.get("area_sqm", 0) for door in doors)

    material_summary: Dict[str, Dict[str, float]] = {}
    for door in doors:
        material = door.get("material", "unknown")
        if material not in material_summary:
            material_summary[material] = {"count": 0, "total_cost": 0, "total_area": 0}

        material_summary[material]["count"] += 1
        material_summary[material]["total_cost"] += door.get("cost_breakdown", {}).get("total_cost", 0)
        material_summary[material]["total_area"] += door.get("area_sqm", 0)

    summary = {
        "total_doors": len(doors),
        "total_cost": round(total_cost, 2),
        "total_area": round(total_area, 3),
        "average_cost_per_door": round(total_cost / len(doors), 2),
        "average_cost_per_sqm": round(total_cost / total_area, 2) if total_area > 0 else 0,
        "material_breakdown": material_summary
    }
    return summary

def create_door_embeddings_text(doors: List[Dict]) -> List[str]:
    """
    Converts door information into texts suitable for indexing in a RAG or vector DB.
    """
    door_texts: List[str] = []
    for i, door in enumerate(doors):
        cost = door.get("cost_breakdown", {})
        text = (
            f"Door {door['door_id']}: "
            f"Size {door['size']}, "
            f"Material {door['material']}, "
            f"Area {door['area_sqm']} square meters, "
            f"Total cost {cost.get('total_cost', 0)} USD , "
            f"Material cost {cost.get('material_cost', 0)} USD , "
            f"Labor cost {cost.get('labor_cost', 0)} USD , "
            f"Installation cost {cost.get('installation_cost', 0)} USD "
        )
        if door.get('notes'):
            text += f", Notes: {door['notes']}"
        door_texts.append(text)
    
    return door_texts