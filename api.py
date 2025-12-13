from flask import Flask, jsonify
from flask_cors import CORS
import requests
import pandas as pd
import joblib
from datetime import datetime, timedelta
import os

# =========================
# APP INIT
# =========================
app = Flask(__name__)
CORS(app)

# =========================
# MODEL & SCALER
# =========================
scaler = joblib.load("scaler.pkl")
model = joblib.load("logreg_model.pkl")
print("Model dan scaler berhasil dimuat")

# =========================
# FEATURES
# =========================
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

# =========================
# DEFAULT LOCATIONS
# =========================
DEFAULT_LOCATIONS = {
    "Jakarta Pusat": (-6.1862, 106.8283),
    "Jakarta Barat": (-6.1650, 106.7890),
    "Jakarta Selatan": (-6.2550, 106.8456),
    "Jakarta Timur": (-6.2100, 106.8950),
    "Jakarta Utara": (-6.1200, 106.8650),
    "Makassar": (-5.1373, 119.4282),
    "Luwu Utara": (-3.3499, 120.3781),
    "Bogor": (-6.5944, 106.7892),
    "Bekasi": (-6.2349, 106.9896),
    "Cianjur": (-6.8222, 107.1394),
    "Sukabumi": (-6.9181, 106.9267)
}

# =========================
# SULAWESI SELATAN (KAB/KOTA)
# =========================
SULSEL = {
    "Makassar": (-5.1373, 119.4282),
    "Parepare": (-4.0096, 119.6236),
    "Palopo": (-2.9929, 120.1969),
    "Barru": (-4.4173, 119.6270),
    "Pinrang": (-3.7850, 119.6520),
    "Bone": (-4.5386, 120.3279),
    "Bulukumba": (-5.4329, 120.2051),
    "Enrekang": (-3.5631, 119.7620),
    "Gowa": (-5.3167, 119.7426),
    "Jeneponto": (-5.6770, 119.7327),
    "Luwu": (-3.3000, 120.1833),
    "Luwu Timur": (-2.5094, 121.3689),
    "Maros": (-5.0055, 119.5736),
    "Pangkep": (-4.8054, 119.5572),
    "Sidrap": (-3.9175, 119.9833),
    "Sinjai": (-5.1241, 120.2530),
    "Soppeng": (-4.3519, 119.8866),
    "Takalar": (-5.4167, 119.5000),
    "Tana Toraja": (-3.0750, 119.7420),
    "Toraja Utara": (-2.9000, 119.8000),
    "Wajo": (-4.0226, 120.0691)
}

# =========================
# KECAMATAN KHUSUS
# =========================
KECAMATAN = {

    "Parepare": {
        "Bacukiki": (-4.0173, 119.6302),
        "Bacukiki Barat": (-4.0027, 119.6106),
        "Soreang": (-4.0154, 119.6258),
        "Ujung": (-4.0096, 119.6236)
    },

    "Pinrang": {
        "Batulappa": (-3.7246, 119.6356),
        "Cempa": (-3.8425, 119.6701),
        "Duampanua": (-3.7842, 119.6239),
        "Lanrisang": (-3.7543, 119.6495),
        "Lembang": (-3.8206, 119.7012),
        "Mattiro Bulu": (-3.7715, 119.6824),
        "Paleteang": (-3.7941, 119.6721),
        "Patampanua": (-3.7098, 119.6514),
        "Suppa": (-3.8179, 119.6093),
        "Tiroang": (-3.8005, 119.6837),
        "Watang Sawitto": (-3.7814, 119.6408)
    },

    "Barru": {
        "Balusu": (-4.3512, 119.6158),
        "Barru": (-4.4173, 119.6270),
        "Mallusetasi": (-4.2651, 119.6044),
        "Pujananting": (-4.4928, 119.7724),
        "Soppeng Riaja": (-4.3075, 119.6463),
        "Tanete Rilau": (-4.4126, 119.6588),
        "Tanete Riaja": (-4.4518, 119.7094)
    }
}

# =========================
# HELPERS
# =========================
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
        return {"status": "Aman", "level": "rendah", "warna": "green"}
    elif prob < 0.60:
        return {"status": "Waspada", "level": "sedang", "warna": "yellow"}
    else:
        return {"status": "Berpotensi Banjir", "level": "tinggi", "warna": "red"}


def run_prediction(locations: dict):
    results = {}

    for name, (lat, lon) in locations.items():
        ready, date, data = fetch_nasa_valid(lat, lon)

        if not ready:
            results[name] = {"data_ready": False}
            continue

        df = pd.DataFrame([data], columns=FEATURES)
        scaled = scaler.transform(df)

        prob = float(model.predict_proba(scaled)[0][1])
        pred = int(model.predict(scaled)[0])

        results[name] = {
            "data_ready": True,
            "date": date,
            "data": data,
            "prediction": {
                "prediksi": pred,
                "probabilitas": prob
            },
            "interpretasi": interpret_risk(prob)
        }

    return results

# =========================
# ROUTES
# =========================
@app.route("/")
def health():
    return jsonify({"status": "ok", "service": "TULIP Smart Climate API"})


@app.route("/predict")
def predict_default():
    hasil = run_prediction(DEFAULT_LOCATIONS)
    return jsonify({"mode": "default", "hasil": hasil})


@app.route("/predict/sulsel")
def predict_sulsel():
    hasil = run_prediction(SULSEL)
    return jsonify({"mode": "sulawesi_selatan", "hasil": hasil})


@app.route("/predict/kecamatan/<wilayah>")
def predict_kecamatan(wilayah):
    wilayah = wilayah.replace("-", " ").title()

    if wilayah not in KECAMATAN:
        return jsonify({
            "error": "Wilayah tidak tersedia",
            "available": list(KECAMATAN.keys())
        }), 404

    hasil = run_prediction(KECAMATAN[wilayah])
    return jsonify({
        "mode": "kecamatan",
        "wilayah": wilayah,
        "hasil": hasil
    })

# =========================
# RUN
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
