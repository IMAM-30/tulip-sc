from flask import Flask, jsonify
import requests
import pandas as pd
import joblib
from datetime import datetime, timedelta

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({
        "status": "ok",
        "service": "TULIP Smart Climate API",
        "endpoint": "/predict"
    })

# ==============================
# DATA & MODEL
# ==============================
LOCATIONS = {
    "Jakarta Pusat": (-6.1862, 106.8283),
    "Makassar": (-5.137320, 119.428238),
}

FEATURES = [
    "PRECTOTCORR", "T2M_MIN", "T2M_MAX", "RH2M",
    "WS2M", "WD2M", "PS", "ALLSKY_SFC_SW_DWN"
]

scaler = joblib.load("scaler.pkl")
model = joblib.load("logreg_model.pkl")

print("Model dan scaler berhasil dimuat")

# ==============================
# NASA FETCH
# ==============================
def fetch_nasa(lat, lon):
    end = datetime.utcnow()
    start = end - timedelta(days=3)

    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters={','.join(FEATURES)}"
        f"&community=AG&latitude={lat}&longitude={lon}"
        f"&start={start:%Y%m%d}&end={end:%Y%m%d}&format=JSON"
    )

    r = requests.get(url, timeout=20)
    d = r.json()["properties"]["parameter"]
    latest = sorted(d["PRECTOTCORR"].keys())[-1]

    return {k: float(d[k][latest]) for k in FEATURES}

def predict(data):
    df = pd.DataFrame([data])
    scaled = scaler.transform(df)
    return {
        "prediksi": int(model.predict(scaled)[0]),
        "probabilitas": float(model.predict_proba(scaled)[0][1])
    }

# ==============================
# API
# ==============================
@app.route("/predict")
def predict_api():
    result = {}
    for city, (lat, lon) in LOCATIONS.items():
        data = fetch_nasa(lat, lon)
        result[city] = {
            "data": data,
            "prediction": predict(data)
        }
    return jsonify(result)
