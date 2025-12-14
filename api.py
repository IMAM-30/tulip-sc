from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import json
import requests

# =========================
# APP INIT
# =========================
app = Flask(__name__)
CORS(app)

# =========================
# CONFIG
# =========================
PREDICTION_PATH = "predictions"

OPENWEATHER_KEY = os.environ.get("OPENWEATHER_KEY")
OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"

# =========================
# HELPERS
# =========================
def load_prediction(name: str):
    """
    Membaca file JSON hasil prediksi NASA + ML
    """
    path = os.path.join(PREDICTION_PATH, f"{name}.json")
    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        return json.load(f)

# =========================
# ROUTES
# =========================
@app.route("/")
def health():
    return jsonify({
        "status": "ok",
        "service": "TULIP Smart Climate API",
        "mode": "cached-ml + realtime-weather"
    })

# -------------------------
# PREDIKSI (CACHE JSON)
# -------------------------
@app.route("/predict/<group>")
def predict_group(group):
    """
    group:
    - default
    - sulsel
    - kecamatan_parepare
    - kecamatan_pinrang
    - kecamatan_barru
    """
    data = load_prediction(group)

    if not data:
        return jsonify({
            "error": "Data prediksi tidak ditemukan",
            "group": group,
            "expected_file": f"predictions/{group}.json"
        }), 404

    return jsonify(data)

# -------------------------
# REALTIME WEATHER
# -------------------------
@app.route("/realtime")
def realtime_weather():
    lat = request.args.get("lat")
    lon = request.args.get("lon")

    if not lat or not lon:
        return jsonify({
            "error": "Parameter lat dan lon wajib"
        }), 400

    params = {
        "lat": lat,
        "lon": lon,
        "appid": OPENWEATHER_KEY,
        "units": "metric",
        "lang": "id"
    }

    try:
        r = requests.get(OPENWEATHER_URL, params=params, timeout=10)
        r.raise_for_status()
        return jsonify(r.json())
    except Exception as e:
        return jsonify({
            "error": "Gagal mengambil cuaca realtime",
            "detail": str(e)
        }), 500

# =========================
# RUN
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
