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
    """Calculate door area - assumes measurements are in centimeters"""
    try:
        size_str = size_str.replace('×', 'x').replace('X', 'x').replace('"', '').strip()
        
        match = re.search(r'(\d+(?:\.\d+)?)\s*["\']?\s*[xX×]\s*(\d+(?:\.\d+)?)\s*["\']?', size_str)
        if not match:
            logging.warning(f"Could not parse size string: '{size_str}'")
            return 0.0

        width, height = map(float, match.groups())
        
        if width < 50 and height < 50:
            width_cm = width * 2.54
            height_cm = height * 2.54
        else:
            width_cm, height_cm = width, height

        area_sqm = (width_cm / 100) * (height_cm / 100)
        return round(area_sqm, 3)
    except Exception as e:
        logging.error(f"Error calculating door area for '{size_str}': {e}")
        return 0.0

def get_material_pricing(material: str) -> Dict[str, float]:
    """Returns material and installation prices by material type."""
    material_key = material.lower().strip()
    if material_key in MATERIAL_PRICES:
        return MATERIAL_PRICES[material_key]

    for key in MATERIAL_PRICES.keys():
        if key in material_key or material_key in key:
            return MATERIAL_PRICES[key]

    logging.warning(f"Material '{material}' not found, using default pricing")
    return MATERIAL_PRICES["default"]

def calculate_door_cost(door_info: Dict) -> Dict[str, float]:
    """Calculates cost breakdown by door area and material."""
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

def is_door_line(line: str) -> bool:
    line_lower = line.lower()
    door_keywords = ['door', 'entry', 'swing', 'pocket', 'bathroom', 'bedroom', 'closet']
    window_keywords = ['window', 'sliding', 'single hung', 'vinyl', 'gl', 'glazing', 'egress']
    has_window_keywords = any(keyword in line_lower for keyword in window_keywords)
    if has_window_keywords:
        print(f"Skipping window line: {line[:50]}...")
        return False
    
    has_door_keywords = any(keyword in line_lower for keyword in door_keywords)
    
    has_door_id = bool(re.search(r'\b(0[1-9]|[1-9]\d*|D\d+)\b', line))
    
    return has_door_keywords or has_door_id

def parse_door_schedule(text: str) -> List[Dict]:
    """
    Extract doors only (not windows) from OCR text.
    """
    doors: List[Dict] = []
    try:
        schedule_patterns = [r'Door\s+Schedule', r'DOOR\s+SCHEDULE']
        schedule_start: Optional[int] = None
        
        for pattern in schedule_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                schedule_start = match.end()
                break

        if schedule_start is None:
            print("Door Schedule section not found, searching entire text")
            relevant_text = text
        else:
            window_match = re.search(r'Window\s+Schedule|WINDOW\s+SCHEDULE', text[schedule_start:], re.IGNORECASE)
            if window_match:
                relevant_text = text[schedule_start:schedule_start + window_match.start()]
            else:
                relevant_text = text[schedule_start:]
        
        lines = [line.strip() for line in relevant_text.splitlines() if line.strip()]

        door_patterns = [
            re.compile(r'^(\d+)\s+([\d"\'×x\s]+)\s+(swing|pocket)\s*(.*)$', re.IGNORECASE),
            re.compile(r'^(0[1-4])\s+([\d"\'×x\s]+)\s+(swing|pocket)\s*(.*)$', re.IGNORECASE),
            re.compile(r'^(\w+)\s+([\d"\'×x\s]+)\s+(\w+)\s*(.*)$', re.IGNORECASE)
        ]

        for line_num, line in enumerate(lines):
            if (re.match(r'^[=\-_\s]*$', line) or 
                len(line) < 5 or
                'count' in line.lower() or
                'size' in line.lower() and 'operation' in line.lower()):
                continue

            if not is_door_line(line):
                continue

            door_found = False
            for pattern_num, pattern in enumerate(door_patterns):
                match = pattern.match(line)
                if match:
                    print(f"Pattern {pattern_num} matched: {line}")
                    
                    door_id = match.group(1).strip()
                    size = match.group(2).strip()
                    operation = match.group(3).strip()
                    extra = match.group(4).strip() if len(match.groups()) > 3 else ""

                    material = "wood"  
                    if "wd" in extra.lower():
                        material = "wooden"
                    elif "metal" in extra.lower():
                        material = "metal"
                    elif "glass" in extra.lower():
                        material = "glass"

                    area = calculate_door_area(size)
                    if area > 0:
                        door_info = {
                            "door_id": door_id,
                            "size": size,
                            "operation": operation,
                            "material": material,
                            "notes": extra,
                            "area_sqm": area
                        }
                        cost_breakdown = calculate_door_cost(door_info)
                        door_info["cost_breakdown"] = cost_breakdown

                        doors.append(door_info)
                        print(f"Added door: {door_id} - {size} - {material}")
                    
                    door_found = True
                    break

            if not door_found:
                print(f"No pattern matched for line: {line}")

    except Exception as e:
        logging.error(f"Error parsing door schedule: {e}")
    
    print(f"Found {len(doors)} doors total")
    return doors

def generate_door_summary(doors: List[Dict]) -> Dict:
    """Creates a total summary of doors with costs and totals by material."""
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
    """Converts door information into texts suitable for indexing in a RAG or vector DB."""
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