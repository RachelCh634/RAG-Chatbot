import re
from typing import List, Dict, Any
import logging
from datetime import datetime
import os

try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    print("Warning: tavily-python not installed. Install with: pip install tavily-python")

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

class TavilyPriceSearcher:
    """Handle price searches using Tavily API"""
    
    def __init__(self, api_key: str = None, region: str = "israel"):
        if api_key is None:
            api_key = os.getenv('TAVILY_API_KEY')
            if api_key:
                print("Using Tavily API key from environment variable")
            else:
                print("No Tavily API key found in environment variable TAVILY_API_KEY")
        
        self.api_key = api_key
        self.region = region
        self.client = None
        self.price_cache = {} 
        
        if TAVILY_AVAILABLE and api_key:
            try:
                self.client = TavilyClient(api_key=api_key)
                print("Tavily client initialized successfully")
            except Exception as e:
                print(f"Failed to initialize Tavily client: {e}")
                self.client = None
        else:
            if not api_key:
                print("No Tavily API key provided - using fallback prices")
            else:
                print("Tavily not available - using fallback prices")

    def search_material_prices(self, material: str, door_type: str = "door") -> Dict[str, Any]:
        if not self.client:
            return {"error": "Tavily client not available", "fallback": True}
        
        cache_key = f"{material}_{door_type}_{self.region}"
        if cache_key in self.price_cache:
            print(f"Using cached price for {material}")
            return self.price_cache[cache_key]
        
        try:
            if self.region.lower() == "israel":
                query = f"{material} door prices Israel 2025 cost per square meter installation NIS shekel"
            else:
                query = f"{material} door prices {self.region} 2025 cost per square meter installation"
            
            print(f"🔍 Searching Tavily for: {query}")
            
            response = self.client.search(
                query=query,
                search_depth="advanced",
                max_results=5,
                include_raw_content=True
            )
            
            price_info = self._extract_price_info(response, material)
            price_info["search_query"] = query
            price_info["search_timestamp"] = datetime.now().isoformat()
            price_info["source"] = "tavily"
            
            self.price_cache[cache_key] = price_info
            
            return price_info
            
        except Exception as e:
            print(f"Tavily search failed for {material}: {e}")
            return {"error": str(e), "fallback": True}

    def _extract_price_info(self, search_response: Dict, material: str) -> Dict[str, Any]:
        extracted_info = {
            "material": material,
            "prices_found": [],
            "average_price_per_sqm": None,
            "average_installation": None,
            "sources": [],
            "confidence": "low"
        }
        
        if not search_response.get("results"):
            return extracted_info
        
        prices_per_sqm = []
        installation_costs = []
        
        for result in search_response["results"]:
            content = result.get("content", "").lower()
            url = result.get("url", "")
            title = result.get("title", "")
            
            price_patterns = [
                r'[₪\$]?(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:per\s*(?:sq\.?m|square\s*meter|m²|מ״ר))',
                r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*[₪\$]\s*(?:per\s*(?:sq\.?m|square\s*meter|m²|מ״ר))',
                r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:ש״ח|שקל)\s*(?:למ״ר|למטר)',
            ]
            
            for pattern in price_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    try:
                        price = float(match.group(1).replace(',', ''))
                        if 100 <= price <= 5000:
                            prices_per_sqm.append(price)
                            extracted_info["prices_found"].append({
                                "price": price,
                                "source": url,
                                "title": title[:100]
                            })
                    except ValueError:
                        continue
            
            install_patterns = [
                r'installation[:\s]+[₪\$]?(\d+(?:,\d{3})*(?:\.\d{2})?)',
                r'התקנה[:\s]+(\d+(?:,\d{3})*(?:\.\d{2})?)\s*[₪שש״ח]',
            ]
            
            for pattern in install_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    try:
                        cost = float(match.group(1).replace(',', ''))
                        if 50 <= cost <= 1000:
                            installation_costs.append(cost)
                    except ValueError:
                        continue
            
            extracted_info["sources"].append({
                "url": url,
                "title": title,
                "relevance_score": result.get("score", 0)
            })
        
        if prices_per_sqm:
            extracted_info["average_price_per_sqm"] = round(sum(prices_per_sqm) / len(prices_per_sqm), 2)
            extracted_info["confidence"] = "high" if len(prices_per_sqm) >= 3 else "medium"
        
        if installation_costs:
            extracted_info["average_installation"] = round(sum(installation_costs) / len(installation_costs), 2)
        
        print(f"Extracted {len(prices_per_sqm)} prices for {material}")
        
        return extracted_info

    def get_enhanced_pricing(self, material: str) -> Dict[str, float]:
        search_result = self.search_material_prices(material)
        
        if (search_result.get("average_price_per_sqm") and 
            not search_result.get("error") and 
            search_result.get("confidence") in ["medium", "high"]):
            
            pricing = {
                "price_per_sqm": search_result["average_price_per_sqm"],
                "installation": search_result.get("average_installation", 250),
                "source": "tavily_enhanced",
                "confidence": search_result["confidence"],
                "last_updated": search_result.get("search_timestamp"),
                "search_sources": len(search_result.get("sources", [])),
                "detailed_sources": search_result.get("sources", []),
                "prices_found": search_result.get("prices_found", []),
                "search_query": search_result.get("search_query", "")
            }
            
            print(f"Using Tavily pricing for {material}: ₪{pricing['price_per_sqm']}/sqm")
            return pricing
        
        else:
            print(f"Using fallback pricing for {material}")
            fallback_pricing = get_material_pricing(material)
            fallback_pricing["source"] = "fallback"
            fallback_pricing["confidence"] = "default"
            fallback_pricing["detailed_sources"] = []
            fallback_pricing["prices_found"] = []
            fallback_pricing["search_query"] = ""
            return fallback_pricing

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
    """Returns material and installation prices by material type (fallback)."""
    material_key = material.lower().strip()
    if material_key in MATERIAL_PRICES:
        return MATERIAL_PRICES[material_key]

    for key in MATERIAL_PRICES.keys():
        if key in material_key or material_key in key:
            return MATERIAL_PRICES[key]

    logging.warning(f"Material '{material}' not found, using default pricing")
    return MATERIAL_PRICES["default"]

def calculate_door_cost_enhanced(door_info: Dict, price_searcher: TavilyPriceSearcher = None) -> Dict[str, float]:
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

        if price_searcher:
            pricing = price_searcher.get_enhanced_pricing(material)
        else:
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
            "price_per_sqm": pricing["price_per_sqm"],
            "pricing_source": pricing.get("source", "fallback"),
            "pricing_confidence": pricing.get("confidence", "default"),
            "pricing_sources_detail": pricing.get("detailed_sources", []),
            "found_prices": pricing.get("prices_found", []),
            "search_query_used": pricing.get("search_query", "")
        }
        
        if pricing.get("last_updated"):
            result["pricing_last_updated"] = pricing["last_updated"]
        if pricing.get("search_sources"):
            result["search_sources_count"] = pricing["search_sources"]
            
        return result
        
    except Exception as e:
        logging.error(f"Error calculating cost for {door_info}: {e}")
        return {
            "material_cost": 0,
            "labor_cost": 0,
            "installation_cost": 0,
            "total_cost": 0,
            "error": str(e)
        }

def extract_door_section_only(text: str) -> str:
    door_start_patterns = [
        r'Door\s+Schedule',
        r'DOOR\s+SCHEDULE',
        r'דלתות',
        r'לוח\s+דלתות'
    ]
    
    door_section_start = None
    for pattern in door_start_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            door_section_start = match.start()
            print(f"Found door section start at position {door_section_start}")
            break
    
    if door_section_start is None:
        print("No door section found, processing entire text")
        return text
    
    door_section_end_patterns = [
        r'Window\s+Schedule',
        r'WINDOW\s+SCHEDULE', 
        r'חלונות',
        r'לוח\s+חלונות',
        r'Hardware\s+Schedule',
        r'HARDWARE\s+SCHEDULE',
        r'Finish\s+Schedule',
        r'FINISH\s+SCHEDULE',
        r'Room\s+Schedule',
        r'ROOM\s+SCHEDULE'
    ]
    
    door_section_end = len(text)  
    
    for pattern in door_section_end_patterns:
        match = re.search(pattern, text[door_section_start:], re.IGNORECASE)
        if match:
            door_section_end = door_section_start + match.start()
            print(f"Found door section end at position {door_section_end} (pattern: {pattern})")
            break
    
    door_only_text = text[door_section_start:door_section_end]
    print(f"Extracted door section: {len(door_only_text)} characters")
    
    return door_only_text

def is_door_line(line: str) -> bool:
    line_lower = line.lower().strip()
    
    if len(line_lower) < 3:
        return False
    
    definite_window_keywords = [
        'window', 'sliding window', 'single hung', 'double hung', 
        'vinyl window', 'glazing', 'egress window', 'casement',
        'awning window', 'חלון', 'חלונות'
    ]
    
    for keyword in definite_window_keywords:
        if keyword in line_lower:
            print(f"Rejecting window line: {line[:50]}...")
            return False
    
    door_keywords = [
        'door', 'entry', 'swing', 'pocket', 'sliding door',
        'bathroom door', 'bedroom door', 'closet door', 'דלת'
    ]
    
    has_door_id = bool(re.search(r'\b(0[1-9]|[1-9]\d*|D\d+)\b', line))
    
    has_door_keywords = any(keyword in line_lower for keyword in door_keywords)
    
    has_dimensions = bool(re.search(r'\d+["\']?\s*[xX×]\s*\d+["\']?', line))    

    is_door = (has_door_id or has_door_keywords) and has_dimensions
    
    if is_door:
        print(f"Accepting door line: {line[:50]}...")
    
    return is_door

def parse_door_schedule_enhanced(text: str, tavily_api_key: str = None, region: str = "israel") -> Dict[str, Any]:
    """
    Enhanced door schedule parser with better window filtering and price source tracking
    """
    price_searcher = None
    if TAVILY_AVAILABLE:
        price_searcher = TavilyPriceSearcher(tavily_api_key, region)
    
    doors: List[Dict] = []
    
    try:
        door_only_text = extract_door_section_only(text)
        
        lines = [line.strip() for line in door_only_text.splitlines() if line.strip()]
        
        print(f"Processing {len(lines)} lines from door section only")

        door_patterns = [
            re.compile(r'^(\d+)\s+([\d"\'×x\s]+)\s+(swing|pocket|sliding)\s*(.*)$', re.IGNORECASE),
            re.compile(r'^(0[1-9])\s+([\d"\'×x\s]+)\s+(swing|pocket|sliding)\s*(.*)$', re.IGNORECASE),
            re.compile(r'^(D\d+)\s+([\d"\'×x\s]+)\s+(\w+)\s*(.*)$', re.IGNORECASE),
            re.compile(r'^(\w+)\s+([\d"\'×x\s]+)\s+(\w+)\s*(.*)$', re.IGNORECASE)
        ]

        for line_num, line in enumerate(lines):
            if (re.match(r'^[=\-_\s]*$', line) or 
                len(line) < 5 or
                'count' in line.lower() or
                ('size' in line.lower() and 'operation' in line.lower()) or
                'door schedule' in line.lower()):
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
                    if "wd" in extra.lower() or "wood" in extra.lower():
                        material = "wooden"
                    elif "metal" in extra.lower() or "steel" in extra.lower():
                        material = "metal"
                    elif "glass" in extra.lower():
                        material = "glass"
                    elif "aluminum" in extra.lower() or "aluminium" in extra.lower():
                        material = "aluminum"
                    elif "pvc" in extra.lower():
                        material = "pvc"

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
                        
                        cost_breakdown = calculate_door_cost_enhanced(door_info, price_searcher)
                        door_info["cost_breakdown"] = cost_breakdown

                        doors.append(door_info)
                        
                        pricing_info = "enhanced" if price_searcher else "standard"
                        print(f"Added door: {door_id} - {size} - {material} ({pricing_info} pricing)")
                    
                    door_found = True
                    break

            if not door_found:
                print(f"No pattern matched for line: {line}")

    except Exception as e:
        logging.error(f"Error parsing door schedule: {e}")
    
    print(f"Found {len(doors)} doors total (windows completely excluded)")
    
    summary = generate_enhanced_door_summary(doors, price_searcher is not None)
    
    return {
        "doors": doors,
        "summary": summary,
        "enhanced_pricing": price_searcher is not None,
        "processed_timestamp": datetime.now().isoformat(),
        "windows_excluded": True,  
        "door_section_only": True  
    }

def generate_enhanced_door_summary(doors: List[Dict], is_enhanced: bool = False) -> Dict:
    """Creates enhanced summary with detailed pricing source information"""
    if not doors:
        return {"total_doors": 0, "total_cost": 0, "total_area": 0}

    total_cost = sum(door.get("cost_breakdown", {}).get("total_cost", 0) for door in doors)
    total_area = sum(door.get("area_sqm", 0) for door in doors)

    material_summary: Dict[str, Dict[str, Any]] = {}
    pricing_sources = {}
    all_pricing_sources = [] 
    
    for door in doors:
        material = door.get("material", "unknown")
        cost_breakdown = door.get("cost_breakdown", {})
        
        if material not in material_summary:
            material_summary[material] = {
                "count": 0, 
                "total_cost": 0, 
                "total_area": 0,
                "avg_price_per_sqm": 0
            }

        material_summary[material]["count"] += 1
        material_summary[material]["total_cost"] += cost_breakdown.get("total_cost", 0)
        material_summary[material]["total_area"] += door.get("area_sqm", 0)
        material_summary[material]["avg_price_per_sqm"] = cost_breakdown.get("price_per_sqm", 0)
        
        pricing_source = cost_breakdown.get("pricing_source", "unknown")
        if pricing_source not in pricing_sources:
            pricing_sources[pricing_source] = 0
        pricing_sources[pricing_source] += 1
        
        detailed_sources = cost_breakdown.get("pricing_sources_detail", [])
        if detailed_sources:
            all_pricing_sources.extend(detailed_sources)

    summary = {
        "total_doors": len(doors),
        "total_cost": round(total_cost, 2),
        "total_area": round(total_area, 3),
        "average_cost_per_door": round(total_cost / len(doors), 2),
        "average_cost_per_sqm": round(total_cost / total_area, 2) if total_area > 0 else 0,
        "material_breakdown": material_summary,
        "pricing_enhanced": is_enhanced,
        "pricing_sources": pricing_sources,
        "detailed_pricing_sources": all_pricing_sources,  
        "unique_source_urls": list(set([source.get("url", "") for source in all_pricing_sources if source.get("url")]))
    }
    
    return summary

def create_enhanced_door_embeddings_text(doors: List[Dict]) -> List[str]:
    """Enhanced version with pricing source information"""
    door_texts: List[str] = []
    for i, door in enumerate(doors):
        cost = door.get("cost_breakdown", {})
        
        text = (
            f"Door {door['door_id']}: "
            f"Size {door['size']}, "
            f"Material {door['material']}, "
            f"Area {door['area_sqm']} square meters, "
            f"Total cost {cost.get('total_cost', 0)} currency units, "
            f"Material cost {cost.get('material_cost', 0)}, "
            f"Labor cost {cost.get('labor_cost', 0)}, "
            f"Installation cost {cost.get('installation_cost', 0)}"
        )
        
        if cost.get("pricing_source"):
            text += f", Pricing source: {cost['pricing_source']}"
        if cost.get("pricing_confidence"):
            text += f", Confidence: {cost['pricing_confidence']}"
        if cost.get("search_query_used"):
            text += f", Search query: {cost['search_query_used']}"
        
        if door.get('notes'):
            text += f", Notes: {door['notes']}"
            
        door_texts.append(text)
    
    return door_texts