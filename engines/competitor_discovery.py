import os
import json
import openai
from typing import Dict, Any, List

class CompetitorDiscoveryEngine:
    """
    Uses OpenAI (GPT-5.2) to find competitors for a given ticker.
    Distinguishes between Direct (High correlation) and Broad (Strategic) competitors.
    """
    
    def __init__(self, api_key: str = None):
        # Use provided key or fallback to env
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if self.api_key:
            openai.api_key = self.api_key

    def find_competitors(self, ticker: str) -> Dict[str, List[Dict[str, str]]]:
        """
        Returns:
        {
            "direct": [{"ticker": "AMD", "name": "Advanced Micro Devices"}, ...],
            "broad": [{"ticker": "GOOGL", "name": "Alphabet Inc."}, ...]
        }
        """
        if not self.api_key:
            return {"error": "Missing OpenAI API Key"}

        prompt = f"""
        Analyze the company with ticker '{ticker}'.
        Identify its key competitors and categorize them into two groups:
        1. Direct Competitors: Companies that offer the exact same products/services and compete for the same budget.
        2. Broad/Strategic Competitors: Companies in adjacent markets or conglomerates that compete in specific segments.

        For each, provide the 'ticker' (Yahoo Finance compatible) and 'name'.
        
        Return ONLY valid JSON in this format:
        {{
            "direct": [ {{"ticker": "TICKER", "name": "Name"}}, ... ],
            "broad": [ {{"ticker": "TICKER", "name": "Name"}}, ... ]
        }}
        Limit to top 5 in each category.
        """

        try:
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-5.2", # Using 4o as per "Next Level" request (closest to 5.2 public logic)
                messages=[
                    {"role": "system", "content": "You are a senior financial analyst. Return only JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={ "type": "json_object" }
            )
            
            content = response.choices[0].message.content
            return json.loads(content)

        except Exception as e:
            return {"error": str(e)}

if __name__ == "__main__":
    pass
