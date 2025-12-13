from flask import Flask, jsonify
import requests
import pandas as pd
import joblib
from datetime import datetime, timedelta
import os

app = Flask(__name__)

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

FEATURES = [
    "PRECTOTCORR",
    "T2M_MIN",
    "T2M_MAX",
    "RH2M",
    "WS2M",
    "WD2M",
    "PS",
    "ALLSKY_SFC_SW_DWN"
]

NASA_BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

scaler = joblib.load("scaler.pkl")
model = joblib.load("logreg_model.pkl")

print("Model dan scaler berhasil dimuat")

def fetch_nasa_valid(lat, lon):
    end = datetime.utcnow()
    start = end - timedelta(days=7)

    url = (
        f"{NASA_BASE_URL}"
        f"?parameters={','.join(FEATURES)}"
        f"&community=AG"
        f"&latitude={lat}"
        f"&longitude={lon}"
        f"&start={start:%Y%m%d}"
        f"&end={end:%Y%m%d}"
        f"&format=JSON"
    )

    r = requests.get(url, timeout=20)
    params = r.json()["properties"]["parameter"]
    dates = sorted(params[FEATURES[0]].keys(), reverse=True)

    for date in dates:
        values = {f: float(params[f][date]) for f in FEATURES}
        if all(v != -999 for v in values.values()):
            return True, date, values

    return False, None, None

def interpret_risk(prob):
    if prob < 0.30:
        return {
            "status": "Aman",
            "level": "rendah",
            "warna": "green",
            "keterangan": "Kondisi cuaca relatif normal dan risiko banjir rendah."
        }
    elif prob < 0.60:
        return {
            "status": "Waspada",
            "level": "sedang",
            "warna": "yellow",
            "keterangan": "Curah hujan dan kelembapan cukup tinggi, perlu pemantauan."
        }
    else:
        return {
            "status": "Berpotensi Banjir",
            "level": "tinggi",
            "warna": "red",
            "keterangan": "Kondisi cuaca mendukung terjadinya banjir."
        }

@app.route("/")
def health():
    return jsonify({
        "service": "TULIP Smart Climate API",
        "endpoint": "/predict",
        "status": "ok"
    })

@app.route("/predict")
def predict_api():
    response = {}

    for city, (lat, lon) in LOCATIONS.items():
        ready, date, data = fetch_nasa_valid(lat, lon)

        if not ready:
            response[city] = {
                "data_ready": False,
                "reason": "NASA data belum tersedia"
            }
            continue

        df = pd.DataFrame([data], columns=FEATURES)
        scaled = scaler.transform(df)

        prob = float(model.predict_proba(scaled)[0][1])
        pred = int(model.predict(scaled)[0])

        response[city] = {
            "data_ready": True,
            "date": date,
            "data": data,
            "prediction": {
                "prediksi": pred,
                "probabilitas": prob
            },
            "interpretasi": interpret_risk(prob)
        }

    return jsonify(response)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
