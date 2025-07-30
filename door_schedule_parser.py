import re
from typing import List, Dict, Any, Optional
import os

try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False

DEFAULT_PRICES = {
    "wood": {"price_per_sqm": 120, "installation": 50},
    "metal": {"price_per_sqm": 180, "installation": 75},
    "glass": {"price_per_sqm": 225, "installation": 100},
    "default": {"price_per_sqm": 150, "installation": 60}
}

class DynamicDoorParser:
    def __init__(self, tavily_api_key: str = None):
        """Initialize parser with optional Tavily integration for real-time pricing"""
        self.tavily_client = None
        if TAVILY_AVAILABLE and (tavily_api_key or os.getenv('TAVILY_API_KEY')):
            try:
                self.tavily_client = TavilyClient(api_key=tavily_api_key or os.getenv('TAVILY_API_KEY'))
            except:
                pass
    
    def clean_text(self, text: str) -> str:
        """Clean OCR text by removing HTML tags and normalizing spaces"""
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def find_door_table_section(self, text: str) -> str:
        """Dynamically locate door table section regardless of exact format"""
        text = self.clean_text(text)
        
        door_indicators = [
            r'door\s*schedule', r'door\s*table', r'door\s*list',
            r'doors?', r'entry', r'bedroom', r'bathroom'
        ]
        
        best_section = ""
        max_door_density = 0
        
        for indicator in door_indicators:
            matches = list(re.finditer(indicator, text, re.IGNORECASE))
            for match in matches:
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 2000)
                section = text[start:end]
                
                door_count = len(re.findall(r'\d+"\s*x\s*\d+"', section))
                if door_count > max_door_density:
                    max_door_density = door_count
                    best_section = section
        
        return best_section if best_section else text[:3000]
    
    def extract_table_rows(self, section: str) -> List[str]:
        """Extract potential table rows from text section"""
        lines = section.split('\n')
        table_rows = []
        
        for line in lines:
            line = line.strip()
            if not line or len(line) < 5:
                continue
            
            has_dimensions = bool(re.search(r'\d+"\s*x\s*\d+"', line))
            has_numbers = bool(re.search(r'\b\d+\b', line))
            has_door_words = bool(re.search(r'swing|pocket|sliding|door|entry|bedroom|bathroom', line, re.IGNORECASE))
            
            if (has_dimensions and has_numbers) or (has_door_words and has_dimensions):
                table_rows.append(line)
        
        return table_rows
    
    def parse_door_row(self, row: str, row_index: int) -> Optional[Dict[str, Any]]:
        """Parse individual table row to extract door information"""
        try:
            parts = row.split()
            if len(parts) < 3:
                return None
            
            door_data = {"row_index": row_index, "original": row}
            
            for i, part in enumerate(parts):
                if re.match(r'^\d{1,2}$', part) and not door_data.get("id"):
                    door_data["id"] = part
                
                elif re.match(r'^\d+$', part) and part != door_data.get("id"):
                    door_data["count"] = int(part)
                
                elif re.search(r'\d+"\s*x\s*\d+"', part):
                    door_data["size"] = part
                    dim_match = re.search(r'(\d+)"\s*x\s*(\d+)"', part)
                    if dim_match:
                        width, height = float(dim_match.group(1)), float(dim_match.group(2))
                        door_data["width"] = width
                        door_data["height"] = height
                        door_data["area_sqm"] = round((width * height * 2.54 * 2.54) / 10000, 4)
                
                elif re.search(r'swing|pocket|sliding', part, re.IGNORECASE):
                    door_data["operation"] = part.lower()
            
            remaining_text = " ".join([p for p in parts if p not in [
                door_data.get("id"), str(door_data.get("count", "")), 
                door_data.get("size", ""), door_data.get("operation", "")
            ]])
            
            door_data["material"] = self._detect_material(remaining_text)
            door_data["remarks"] = remaining_text.strip()
            
            return door_data if door_data.get("size") else None
            
        except Exception:
            return None
    
    def _detect_material(self, text: str) -> str:
        """Detect door material from text content"""
        text_lower = text.lower()
        if any(word in text_lower for word in ["glass", "gl"]):
            return "glass"
        elif any(word in text_lower for word in ["metal", "steel", "aluminum"]):
            return "metal"
        else:
            return "wood"
    
    def get_material_price(self, material: str) -> Dict[str, float]:
        """Get material pricing from Tavily or use defaults"""
        if self.tavily_client:
            try:
                query = f"{material} door price per square meter 2025"
                response = self.tavily_client.search(query=query, max_results=3)
                
                prices = []
                for result in response.get("results", []):
                    content = result.get("content", "").lower()
                    price_match = re.search(r'[₪\$]?(\d+(?:\.\d{2})?)\s*(?:per\s*(?:sq\.?m|m²))', content)
                    if price_match:
                        price = float(price_match.group(1))
                        if 50 <= price <= 1000:
                            prices.append(price)
                
                if prices:
                    return {
                        "price_per_sqm": round(sum(prices) / len(prices), 2),
                        "installation": 60,
                        "source": "tavily"
                    }
            except:
                pass
        
        return DEFAULT_PRICES.get(material, DEFAULT_PRICES["default"])
    
    def calculate_costs(self, doors: List[Dict]) -> List[Dict]:
        """Calculate costs for all doors using dynamic pricing"""
        for door in doors:
            if not door.get("area_sqm"):
                continue
                
            pricing = self.get_material_price(door["material"])
            material_cost = door["area_sqm"] * pricing["price_per_sqm"]
            labor_cost = material_cost * 0.25
            installation_cost = pricing["installation"]
            
            door["costs"] = {
                "material": round(material_cost, 2),
                "labor": round(labor_cost, 2),
                "installation": round(installation_cost, 2),
                "total": round(material_cost + labor_cost + installation_cost, 2),
                "source": pricing.get("source", "default")
            }
        
        return doors
    
    def parse_doors(self, text: str) -> Dict[str, Any]:
        """Main function to parse doors from any table format"""
        section = self.find_door_table_section(text)
        rows = self.extract_table_rows(section)
        
        doors = []
        for i, row in enumerate(rows):
            door = self.parse_door_row(row, i)
            if door:
                doors.append(door)
        
        doors = self.calculate_costs(doors)
        total_cost = sum(door.get("costs", {}).get("total", 0) for door in doors)
        
        return {
            "doors": doors,
            "summary": {
                "total_doors": len(doors),
                "total_cost": round(total_cost, 2),
                "by_material": self._group_by_material(doors)
            }
        }
    
    def _group_by_material(self, doors: List[Dict]) -> Dict[str, Dict]:
        """Group doors by material type for summary"""
        grouped = {}
        for door in doors:
            material = door["material"]
            if material not in grouped:
                grouped[material] = {"count": 0, "total_cost": 0}
            grouped[material]["count"] += door.get("count", 1)
            grouped[material]["total_cost"] += door.get("costs", {}).get("total", 0)
        return grouped


def parse_doors_dynamic(text: str, tavily_api_key: str = None) -> Dict[str, Any]:
    """Parse doors from any table format with optional Tavily pricing"""
    parser = DynamicDoorParser(tavily_api_key)
    return parser.parse_doors(text)

def extract_doors_fast(text: str) -> List[Dict[str, Any]]:
    """Quick door extraction without pricing calculations"""
    parser = DynamicDoorParser()
    result = parser.parse_doors(text)
    return result.get("doors", [])