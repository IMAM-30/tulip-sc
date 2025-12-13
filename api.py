from flask import Flask, jsonify
import requests
import pandas as pd
import joblib
from datetime import datetime, timedelta
import os

app = Flask(__name__)

# 1. DAFTAR KOTA + KOORDINAT
LOCATIONS = {
    "Jakarta Pusat":   (-6.1862, 106.8283),
    "Jakarta Barat":   (-6.1650, 106.7890),
    "Jakarta Selatan": (-6.2550, 106.8456),
    "Jakarta Timur":   (-6.2100, 106.8950),
    "Jakarta Utara":   (-6.1200, 106.8650),
    "Makassar":        (-5.137320, 119.428238),
    "Luwu Utara":      (-3.349880, 120.378101),
    "Bogor":           (-6.59444, 106.78917),
    "Bekasi":          (-6.2349, 106.9896),
    "Cianjur":         (-6.82222, 107.13944),
    "Sukabumi":        (-6.91806, 106.92667)
}

# 2. FEATURE MODEL
FEATURES = [
    "PRECTOTCORR", "T2M_MIN", "T2M_MAX", "RH2M",
    "WS2M", "WD2M", "PS", "ALLSKY_SFC_SW_DWN"
]

# 3. LOAD MODEL
scaler = joblib.load("scaler.pkl")
model = joblib.load("logreg_model.pkl")

print("Model dan scaler berhasil dimuat")

# 4. AMBIL DATA NASA
def fetch_nasa(lat, lon):
    end = datetime.utcnow()
    start = end - timedelta(days=3)

    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point?"
        f"parameters={','.join(FEATURES)}"
        f"&community=AG"
        f"&latitude={lat}&longitude={lon}"
        f"&start={start:%Y%m%d}&end={end:%Y%m%d}"
        "&format=JSON"
    )

    r = requests.get(url, timeout=20)
    r.raise_for_status()

    d = r.json()["properties"]["parameter"]
    latest = sorted(d["PRECTOTCORR"].keys())[-1]

    return {k: float(d[k][latest]) for k in FEATURES}

# 5. PREDIKSI
def predict(data):
    df = pd.DataFrame([data], columns=FEATURES)
    scaled = scaler.transform(df)

    pred = int(model.predict(scaled)[0])
    prob = float(model.predict_proba(scaled)[0][1])

    return {
        "prediksi": pred,
        "probabilitas": prob
    }

# 6. ENDPOINT API
@app.route("/predict")
def predict_api():
    result = {}

    for city, (lat, lon) in LOCATIONS.items():
        try:
            data = fetch_nasa(lat, lon)
            result[city] = {
                "data": data,
                "prediction": predict(data)
            }
        except Exception as e:
            result[city] = {"error": str(e)}

    return jsonify(result)
