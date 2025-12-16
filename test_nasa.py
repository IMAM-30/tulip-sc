import requests
from datetime import datetime, timedelta

# Test untuk Parepare
lat, lon = -4.0096, 119.6236
end = datetime.utcnow()
start = end - timedelta(days=7)

url = "https://power.larc.nasa.gov/api/temporal/daily/point"
params = {
    "parameters": "PRECTOTCORR,T2M_MIN,T2M_MAX,RH2M,WS2M,WD2M,PS,ALLSKY_SFC_SW_DWN",
    "community": "AG",
    "latitude": lat,
    "longitude": lon,
    "start": start.strftime("%Y%m%d"),
    "end": end.strftime("%Y%m%d"),
    "format": "JSON"
}

print("Testing NASA API...")
print(f"URL: {url}")
print(f"Params: {params}")
print("-" * 50)

try:
    response = requests.get(url, params=params, timeout=30)
    print(f"Status: {response.status_code}")
    print(f"Headers: {dict(response.headers)}")
    
    if response.status_code == 200:
        data = response.json()
        print("\n✅ API Success!")
        
        # Cek struktur
        if "properties" in data and "parameter" in data["properties"]:
            params_data = data["properties"]["parameter"]
            first_param = list(params_data.keys())[0] if params_data else None
            
            if first_param:
                dates = list(params_data[first_param].keys())
                print(f"\n📅 Available dates ({len(dates)}):")
                for d in sorted(dates, reverse=True)[:5]:  # 5 terbaru
                    print(f"  - {d}")
                    
                    # Sample values
                    sample_vals = {}
                    for f in ["PRECTOTCORR", "T2M_MAX", "RH2M"]:
                        if f in params_data:
                            sample_vals[f] = params_data[f].get(d, "N/A")
                    print(f"    Values: {sample_vals}")
        else:
            print("\n❌ Unexpected response structure")
            print(f"Response keys: {list(data.keys())}")
    else:
        print(f"\n❌ API Error: {response.text[:500]}")
        
except Exception as e:
    print(f"\n🔥 Exception: {e}")