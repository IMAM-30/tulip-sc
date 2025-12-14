import os
import json
import requests
import pandas as pd
import joblib
from datetime import datetime, timedelta

# =========================
# CONFIG
# =========================
NASA_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
FEATURES = [
    "PRECTOTCORR","T2M_MIN","T2M_MAX","RH2M",
    "WS2M","WD2M","PS","ALLSKY_SFC_SW_DWN"
]

PREDICTION_PATH = "predictions"

scaler = joblib.load("scaler.pkl")
model = joblib.load("logreg_model.pkl")

# =========================
# LOCATION DATA (PAKAI PUNYAMU)
# =========================
from locations import (
    DEFAULT_LOCATIONS,
    SULSEL,
    KECAMATAN
)

# =========================
# HELPERS
# =========================
def fetch_nasa_valid(lat, lon):
    end = datetime.utcnow()
    start = end - timedelta(days=7)

    url = (
        f"{NASA_URL}"
        f"?parameters={','.join(FEATURES)}"
        f"&community=AG"
        f"&latitude={lat}"
        f"&longitude={lon}"
        f"&start={start:%Y%m%d}"
        f"&end={end:%Y%m%d}"
        f"&format=JSON"
    )

    r = requests.get(url, timeout=30).json()
    params = r["properties"]["parameter"]

    for date in sorted(params[FEATURES[0]].keys(), reverse=True):
        values = {f: float(params[f][date]) for f in FEATURES}
        if all(v != -999 for v in values.values()):
            return date, values

    return None, None

def interpret(prob):
    if prob < 0.3:
        return {"status":"Aman","level":"rendah","warna":"green"}
    if prob < 0.6:
        return {"status":"Waspada","level":"sedang","warna":"yellow"}
    return {"status":"Berpotensi Banjir","level":"tinggi","warna":"red"}

def run_prediction(group_name, locations):
    hasil = {}

    for name,(lat,lon) in locations.items():
        date,data = fetch_nasa_valid(lat,lon)
        if not data:
            continue

        df = pd.DataFrame([data], columns=FEATURES)
        prob = float(model.predict_proba(scaler.transform(df))[0][1])

        hasil[name] = {
            "lat": lat,
            "lon": lon,
            "date": date,
            "data": data,
            "prediction": {
                "probabilitas": prob
            },
            "interpretasi": interpret(prob)
        }

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "hasil": hasil
    }

# =========================
# MAIN
# =========================
os.makedirs(PREDICTION_PATH, exist_ok=True)

groups = {
    "default": DEFAULT_LOCATIONS,
    "sulsel": SULSEL,
    "kecamatan_parepare": KECAMATAN["Parepare"],
    "kecamatan_pinrang": KECAMATAN["Pinrang"],
    "kecamatan_barru": KECAMATAN["Barru"]
}

for name, locations in groups.items():
    print(f"Updating {name}...")
    result = run_prediction(name, locations)

    with open(f"{PREDICTION_PATH}/{name}.json", "w") as f:
        json.dump(result, f, indent=2)

print("✅ Update selesai")
