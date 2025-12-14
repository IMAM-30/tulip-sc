import os, json, time, requests
import pandas as pd
import joblib
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "predictions")
os.makedirs(OUT_DIR, exist_ok=True)

scaler = joblib.load("scaler.pkl")
model = joblib.load("logreg_model.pkl")

FEATURES = [
    "PRECTOTCORR","T2M_MIN","T2M_MAX","RH2M",
    "WS2M","WD2M","PS","ALLSKY_SFC_SW_DWN"
]

NASA_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

from locations import LOCATION_INDEX   # << SEMUA LOKASI KAMU DIPAKAI

def fetch_valid(lat, lon):
    end = datetime.utcnow()
    start = end - timedelta(days=7)

    url = (
        f"{NASA_URL}?parameters={','.join(FEATURES)}"
        f"&community=AG&latitude={lat}&longitude={lon}"
        f"&start={start:%Y%m%d}&end={end:%Y%m%d}&format=JSON"
    )

    r = requests.get(url, timeout=30).json()
    params = r["properties"]["parameter"]

    for d in sorted(params[FEATURES[0]], reverse=True):
        vals = {f: float(params[f][d]) for f in FEATURES}
        if all(v != -999 for v in vals.values()):
            return d, vals
    return None, None

def interpret(prob):
    if prob < 0.3:
        return {"status":"Aman","warna":"green"}
    if prob < 0.6:
        return {"status":"Waspada","warna":"yellow"}
    return {"status":"Berpotensi Banjir","warna":"red"}

for slug, loc in LOCATION_INDEX.items():
    try:
        date, data = fetch_valid(loc["lat"], loc["lon"])
        if not data:
            continue

        df = pd.DataFrame([data], columns=FEATURES)
        prob = float(model.predict_proba(scaler.transform(df))[0][1])

        result = {
            "slug": slug,
            "location": loc["name"],
            "group": loc["group"],
            "parent": loc["parent"],
            "date": date,
            "nasa": data,
            "prediction": {
                "probabilitas": prob
            },
            "interpretasi": interpret(prob),
            "updated_at": datetime.utcnow().isoformat()
        }

        with open(os.path.join(OUT_DIR, f"{slug}.json"), "w") as f:
            json.dump(result, f, indent=2)

        time.sleep(1)  # << PENTING: JANGAN BERSAMAAN

    except Exception as e:
        print("ERROR:", slug, e)
