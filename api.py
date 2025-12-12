from flask import Flask, jsonify
from flask_socketio import SocketIO
import requests
import numpy as np
import pandas as pd
import joblib
import threading
import time
from datetime import datetime, timedelta

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

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

# 2. LOAD MODEL
FEATURES = [
    "PRECTOTCORR", "T2M_MIN", "T2M_MAX", "RH2M",
    "WS2M", "WD2M", "PS", "ALLSKY_SFC_SW_DWN"
]

scaler = joblib.load("scaler.pkl")
logreg_model = joblib.load("logreg_model.pkl")

print("Model dan Scaler berhasil dimuat.")

# 3. AMBIL DATA NASA LENGKAP
def fetch_nasa(lat, lon):
    end = datetime.utcnow()
    start = end - timedelta(days=3)

    url = (
        f"https://power.larc.nasa.gov/api/temporal/daily/point?"
        f"parameters={','.join(FEATURES)}"
        f"&community=AG"
        f"&latitude={lat}&longitude={lon}"
        f"&start={start:%Y%m%d}&end={end:%Y%m%d}"
        f"&format=JSON"
    )

    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        d = r.json()["properties"]["parameter"]

        latest = sorted(d["PRECTOTCORR"].keys())[-1]

        return {
            "PRECTOTCORR": float(d["PRECTOTCORR"][latest]),
            "T2M_MIN": float(d["T2M_MIN"][latest]),
            "T2M_MAX": float(d["T2M_MAX"][latest]),
            "RH2M": float(d["RH2M"][latest]),
            "WS2M": float(d["WS2M"][latest]),
            "WD2M": float(d["WD2M"][latest]),
            "PS": float(d["PS"][latest]),
            "ALLSKY_SFC_SW_DWN": float(d["ALLSKY_SFC_SW_DWN"][latest])
        }

    except Exception as e:
        print("NASA Error:", e)
        return None

# 4. FUNGSI PREDIKSI (LOGREG)
def predict(data):
    df = pd.DataFrame([data], columns=FEATURES)

    scaled = scaler.transform(df)
    pred = int(logreg_model.predict(scaled)[0])

    try:
        prob = float(logreg_model.predict_proba(scaled)[0][1])
    except:
        prob = None

    return {
        "prediksi": pred,
        "probabilitas": prob
    }

# 5. LOOP BACKGROUND REALTIME
def background_loop():
    while True:
        result_all = {}

        for name, (lat, lon) in LOCATIONS.items():
            data = fetch_nasa(lat, lon)

            if data:
                pred = predict(data)
                result_all[name] = {
                    "data": data,
                    "prediction": pred
                }
            else:
                result_all[name] = {"error": "NASA fetch failed"}

        socketio.emit("prediction", result_all)
        print("Realtime update terkirim.")

        time.sleep(10)

# 6. API MANUAL (opsional)
@app.route("/predict")
def predict_api():
    output = {}

    for name, (lat, lon) in LOCATIONS.items():
        data = fetch_nasa(lat, lon)

        if data:
            output[name] = {
                "data": data,
                "prediction": predict(data)
            }
        else:
            output[name] = {"error": "NASA fetch failed"}

    return jsonify(output)

# 7. RUN SERVER
if __name__ == "__main__":
    thread = threading.Thread(target=background_loop)
    thread.daemon = True
    thread.start()

    socketio.run(app, host="0.0.0.0", port=5000)
