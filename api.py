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
    "Bogor": (-6.59444, 106.78917),
    "Bekasi": (-6.2349, 106.9896),
    "Cianjur": (-6.82222, 107.13944),
    "Sukabumi": (-6.91806, 106.92667)
}

# =========================
# PROVINCES
# =========================
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
        "Luwu Utara": (-3.3499, 120.3781),
        "Maros": (-5.0055, 119.5736),
        "Pangkajene dan Kepulauan": (-4.8054, 119.5572),
        "Pinrang": (-3.7850, 119.6520),
        "Sidenreng Rappang": (-3.9175, 119.9833),
        "Sinjai": (-5.1241, 120.2530),
        "Soppeng": (-4.3519, 119.8866),
        "Takalar": (-5.4167, 119.5000),
        "Tana Toraja": (-3.0750, 119.7420),
        "Toraja Utara": (-2.9000, 119.8000),
        "Wajo": (-4.0226, 120.0691)
    },

    "Sulawesi Barat": {
        "Mamuju": (-2.6767, 118.8885),
        "Majene": (-3.1500, 118.8667),
        "Polewali Mandar": (-3.4324, 119.3435),
        "Mamasa": (-3.0853, 119.3464),
        "Mamuju Tengah": (-2.7437, 119.3036),
        "Pasangkayu": (-1.4677, 119.4321)
    },

    "Sulawesi Tengah": {
        "Palu": (-0.8986, 119.8506),
        "Banggai": (-1.0416, 122.7713),
        "Banggai Kepulauan": (-1.6403, 123.5504),
        "Banggai Laut": (-1.5833, 123.5000),
        "Buol": (1.0465, 121.5000),
        "Donggala": (-0.6587, 119.7427),
        "Morowali": (-2.8473, 121.9794),
        "Morowali Utara": (-1.9000, 121.3000),
        "Parigi Moutong": (-0.8428, 120.1836),
        "Poso": (-1.3950, 120.7520),
        "Sigi": (-1.3850, 119.9669),
        "Tojo Una-Una": (-1.0986, 121.5736),
        "Tolitoli": (1.3081, 120.8861)
    },

    "Sulawesi Tenggara": {
        "Kendari": (-3.9985, 122.5129),
        "Baubau": (-5.4667, 122.6333),
        "Bombana": (-4.7700, 121.9100),
        "Buton": (-5.3096, 122.9886),
        "Buton Selatan": (-5.5167, 122.7500),
        "Buton Tengah": (-5.3667, 122.4833),
        "Buton Utara": (-4.7000, 123.0333),
        "Kolaka": (-4.0497, 121.5986),
        "Kolaka Timur": (-4.3000, 121.8333),
        "Kolaka Utara": (-3.4000, 121.2000),
        "Konawe": (-3.9381, 122.0837),
        "Konawe Kepulauan": (-4.0833, 123.0500),
        "Konawe Selatan": (-4.1333, 122.4500),
        "Konawe Utara": (-3.4167, 122.2000),
        "Muna": (-4.8333, 122.5000),
        "Muna Barat": (-4.8000, 122.3833),
        "Wakatobi": (-5.6500, 123.9500)
    },

    "Gorontalo": {
        "Gorontalo": (0.5435, 123.0580),
        "Bone Bolango": (0.5356, 123.3486),
        "Boalemo": (0.7000, 122.3000),
        "Pohuwato": (0.7090, 121.7000),
        "Gorontalo Utara": (0.8600, 122.9000)
    },

    "Sulawesi Utara": {
        "Manado": (1.4748, 124.8421),
        "Bitung": (1.4400, 125.1890),
        "Tomohon": (1.3250, 124.8370),
        "Kotamobagu": (0.7333, 124.3167),
        "Bolaang Mongondow": (0.7333, 124.0000),
        "Bolaang Mongondow Selatan": (0.5500, 123.9000),
        "Bolaang Mongondow Timur": (0.8000, 124.4000),
        "Bolaang Mongondow Utara": (0.9000, 123.3000),
        "Kepulauan Sangihe": (3.5500, 125.5500),
        "Kepulauan Siau Tagulandang Biaro": (2.7500, 125.4000),
        "Kepulauan Talaud": (4.3067, 126.8047),
        "Minahasa": (1.2167, 124.8333),
        "Minahasa Selatan": (1.0500, 124.5500),
        "Minahasa Tenggara": (1.0000, 124.8000),
        "Minahasa Utara": (1.5167, 124.9167)
    }

}


# =========================
# HELPERS
# =========================
def normalize_province(slug: str) -> str:
    return slug.replace("-", " ").title()

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

    for city, (lat, lon) in locations.items():
        ready, date, data = fetch_nasa_valid(lat, lon)

        if not ready:
            results[city] = {"data_ready": False}
            continue

        df = pd.DataFrame([data], columns=FEATURES)
        scaled = scaler.transform(df)

        prob = float(model.predict_proba(scaled)[0][1])
        pred = int(model.predict(scaled)[0])

        results[city] = {
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
    return jsonify({
        "mode": "default",
        "jumlah_lokasi": len(hasil),
        "hasil": hasil
    })

@app.route("/predict/provinsi/<province_slug>")
def predict_province(province_slug):
    province_name = normalize_province(province_slug)

    if province_name not in PROVINCES:
        return jsonify({
            "error": "Provinsi tidak tersedia",
            "available": list(PROVINCES.keys())
        }), 404

    hasil = run_prediction(PROVINCES[province_name])

    return jsonify({
        "mode": "provinsi",
        "provinsi": province_name,
        "jumlah_lokasi": len(hasil),
        "hasil": hasil
    })

# =========================
# RUN
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
