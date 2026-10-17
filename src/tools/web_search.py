"""Mock Web search tool 
"""

import hashlib
import time
from urllib.parse import quote_plus

from .base import ToolResult

# so every multi-step plan gets a value to compute on
_NUMERIC_HINTS = ("price", "cost", "rate", "worth", "value", "population", "salary", "market cap", "revenue", "gdp")

def run(query: str, simulated_latency_ms: float = 0) -> ToolResult:
    if not query or not query.strip():
        return ToolResult("web_search", ok=False, error="empty search query")
    if simulated_latency_ms:
        time.sleep(simulated_latency_ms / 1000)

    url = f"https://example.com/search?q={quote_plus(query)}"
    results = [
        {"title": f"[mock] Result {i} for '{query}'", "url": f"{url}&r={i}",
         "snippet": "Placeholder snippet. A real search API would return live content here."}
        for i in (1, 2, 3)
    ]
    data = {"query": query, "results": results, "mock": True}

    output = f"[mock] Top results for '{query}': " + "; ".join(r["title"] for r in results)
    if any(h in query.lower() for h in _NUMERIC_HINTS):
        seed = int(hashlib.md5(query.lower().encode()).hexdigest(), 16)
        data["value"] = round(50 + (seed % 95000) / 100, 2)  # somewhere between 50 and 1000
        output += f" | [mock] extracted value: {data['value']}"

    return ToolResult("web_search", ok=True, output=output, data=data)
