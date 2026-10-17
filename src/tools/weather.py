"""Mock Weather Tool 
"""

import hashlib
import random
import time

from .base import ToolResult

_CONDITIONS = ["sunny", "partly cloudy", "cloudy", "light rain", "heavy rain", "thunderstorms", "windy", "hazy"]


def run(location: str, simulated_latency_ms: float = 0) -> ToolResult:
    if not location or not location.strip():
        return ToolResult("weather", ok=False, error="no location given")
    if simulated_latency_ms:
        time.sleep(simulated_latency_ms / 1000)  # mimic a real API round-trip (see registry.py)

    seed = int(hashlib.md5(location.strip().lower().encode()).hexdigest(), 16)
    rng = random.Random(seed)
    temp = rng.randint(8, 34)  
    condition = rng.choice(_CONDITIONS)
    humidity = rng.randint(30, 95)
    rain_chance = rng.randint(60, 95) if "rain" in condition or "thunder" in condition else rng.randint(0, 40)

    return ToolResult(
        "weather", ok=True,
        output=(f"[mock] Weather in {location}: {condition}, {temp}°C, "
                f"humidity {humidity}%, {rain_chance}% chance of rain"),
        data={"location": location, "condition": condition, "temp_c": temp,
              "humidity": humidity, "rain_chance": rain_chance, "mock": True},
    )
