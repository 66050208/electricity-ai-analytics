import requests


def estimate_bill(kwh, rate=4.20):
    # Simple average-rate estimate for a student project; not an official bill calculator.
    return max(0.0, float(kwh)) * float(rate)


def appliance_estimate(watts, hours_per_day, days, rate=4.20):
    kwh = (float(watts) / 1000.0) * float(hours_per_day) * int(days)
    return kwh, estimate_bill(kwh, rate)


def get_weather(latitude=13.7563, longitude=100.5018):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {"latitude": latitude, "longitude": longitude, "current":"temperature_2m,relative_humidity_2m", "timezone":"auto"}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        current = r.json().get("current", {})
        return {"temperature": current.get("temperature_2m", "-"), "humidity": current.get("relative_humidity_2m", "-")}
    except Exception:
        return {"temperature":"-", "humidity":"-"}
