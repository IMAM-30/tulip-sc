from flask import Flask, jsonify
from flask_cors import CORS
import requests
import pandas as pd
import joblib
from datetime import datetime, timedelta
import os

# =====================================================
# APP INIT
# =====================================================
app = Flask(__name__)
CORS(app)

# =====================================================
# DEFAULT LOCATIONS (DASHBOARD CEPAT)
# =====================================================
DEFAULT_LOCATIONS = {
    "Jakarta Pusat": (-6.1862, 106.8283),
    "Jakarta Barat": (-6.1650, 106.7890),
    "Jakarta Selatan": (-6.2550, 106.8456),
    "Jakarta Timur": (-6.2100, 106.8950),
    "Jakarta Utara": (-6.1200, 106.8650),
    "Makassar": (-5.1373, 119.4282),
    "Luwu Utara": (-3.3499, 120.3781),
    "Bogor": (-6.59444, 106.78917),
    "Bekasi": (-6.2349, 106.9896),
    "Cianjur": (-6.82222, 107.13944),
    "Sukabumi": (-6.91806, 106.92667)
}

# =====================================================
# PROVINCES (KAB/KOTA PER PROVINSI)
# =====================================================
PROVINCES = {
    "Sulawesi Selatan": {
        "Makassar": (-5.1373, 119.4282),
        "Parepare": (-4.0096, 119.6236),
        "Palopo": (-2.9929, 120.1969),
        "Bantaeng": (-5.5587, 120.0203),
        "Barru": (-4.4173, 119.6270),
        "Bone": (-4.5386, 120.3279),
        "Bulukumba": (-5.4329, 120.2051),
        "Enrekang": (-3.5631, 119.7620),
        "Gowa": (-5.3167, 119.7426),
        "Jeneponto": (-5.6770, 119.7327),
        "Kepulauan Selayar": (-6.1187, 120.4583),
        "Luwu": (-3.3000, 120.1833),
        "Luwu Timur": (-2.5094, 121.3689),
        "Maros": (-5.0055, 119.5736),
        "Pangkep": (-4.8054, 119.5572),
        "Pinrang": (-3.7850, 119.6520),
        "Sidrap": (-3.9175, 119.9833),
        "Sinjai": (-5.1241, 120.2530),
        "Soppeng": (-4.3519, 119.8866),
        "Takalar": (-5.4167, 119.5000),
        "Tana Toraja": (-3.0750, 119.7420),
        "Toraja Utara": (-2.9000, 119.8000),
        "Wajo": (-4.0226, 120.0691)
    }
}

# =====================================================
# ML CONFIG
# =====================================================
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

print("Model & scaler loaded")

# =====================================================
# UTIL FUNCTIONS
# =====================================================
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
            "warna": "green"
        }
    elif prob < 0.60:
        return {
            "status": "Waspada",
            "level": "sedang",
            "warna": "yellow"
        }
    else:
        return {
            "status": "Berpotensi Banjir",
            "level": "tinggi",
            "warna": "red"
        }


def run_prediction(locations: dict):
    result = {}

    for name, (lat, lon) in locations.items():
        ready, date, data = fetch_nasa_valid(lat, lon)

        if not ready:
            result[name] = {"data_ready": False}
            continue

        df = pd.DataFrame([data], columns=FEATURES)
        scaled = scaler.transform(df)

        prob = float(model.predict_proba(scaled)[0][1])
        pred = int(model.predict(scaled)[0])

        result[name] = {
            "data_ready": True,
            "date": date,
            "data": data,
            "prediction": {
                "prediksi": pred,
                "probabilitas": prob
            },
            "interpretasi": interpret_risk(prob)
        }

    return result

# =====================================================
# ROUTES
# =====================================================
@app.route("/")
def health():
    return jsonify({
        "service": "TULIP Climate ML API",
        "status": "ok"
    })


@app.route("/predict")
def predict_default():
    return jsonify({
        "mode": "default",
        "jumlah_lokasi": len(DEFAULT_LOCATIONS),
        "hasil": run_prediction(DEFAULT_LOCATIONS)
    })


@app.route("/predict/provinsi/<provinsi>")
def predict_province(provinsi):
    if provinsi not in PROVINCES:
        return jsonify({
            "error": "Provinsi tidak tersedia"
        }), 404

    locations = PROVINCES[provinsi]

    return jsonify({
        "mode": "provinsi",
        "provinsi": provinsi,
        "jumlah_lokasi": len(locations),
        "hasil": run_prediction(locations)
    })

# =====================================================
# ENTRY POINT
# =====================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
