from flask import Flask, jsonify
from flask_cors import CORS
import requests, time, os
import pandas as pd
import joblib
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

# =========================
# LOAD MODEL
# =========================
scaler = joblib.load("scaler.pkl")
model = joblib.load("logreg_model.pkl")

FEATURES = [
    "PRECTOTCORR","T2M_MIN","T2M_MAX","RH2M",
    "WS2M","WD2M","PS","ALLSKY_SFC_SW_DWN"
]

NASA_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# =========================
# RAW LOCATION DATA (PUNYA KAMU)
# =========================
DEFAULT_LOCATIONS = {
    "Jakarta Pusat": (-6.1862,106.8283),
    "Jakarta Barat": (-6.1650,106.7890),
    "Jakarta Selatan": (-6.2550,106.8456),
    "Jakarta Timur": (-6.2100,106.8950),
    "Jakarta Utara": (-6.1200,106.8650),
    "Makassar": (-5.1373,119.4282),
    "Luwu Utara": (-3.3499,120.3781),
    "Bogor": (-6.5944,106.7892),
    "Bekasi": (-6.2349,106.9896),
    "Cianjur": (-6.8222,107.1394),
    "Sukabumi": (-6.9181,106.9267)
}

SULSEL = {
    "Makassar": (-5.1373,119.4282),
    "Parepare": (-4.0096,119.6236),
    "Palopo": (-2.9929,120.1969),
    "Barru": (-4.4173,119.6270),
    "Pinrang": (-3.7850,119.6520),
    "Bone": (-4.5386,120.3279),
    "Bulukumba": (-5.4329,120.2051),
    "Enrekang": (-3.5631,119.7620),
    "Gowa": (-5.3167,119.7426),
    "Jeneponto": (-5.6770,119.7327),
    "Luwu": (-3.3000,120.1833),
    "Luwu Timur": (-2.5094,121.3689),
    "Maros": (-5.0055,119.5736),
    "Pangkep": (-4.8054,119.5572),
    "Sidrap": (-3.9175,119.9833),
    "Sinjai": (-5.1241,120.2530),
    "Soppeng": (-4.3519,119.8866),
    "Takalar": (-5.4167,119.5000),
    "Tana Toraja": (-3.0750,119.7420),
    "Toraja Utara": (-2.9000,119.8000),
    "Wajo": (-4.0226,120.0691)
}

KECAMATAN = {
    "Parepare": {
        "Bacukiki": (-4.0173,119.6302),
        "Bacukiki Barat": (-4.0027,119.6106),
        "Soreang": (-4.0154,119.6258),
        "Ujung": (-4.0096,119.6236)
    },
    "Pinrang": {
        "Batulappa": (-3.7246,119.6356),
        "Cempa": (-3.8425,119.6701),
        "Duampanua": (-3.7842,119.6239),
        "Lanrisang": (-3.7543,119.6495),
        "Watang Sawitto": (-3.7814,119.6408)
    },
    "Barru": {
        "Balusu": (-4.3512,119.6158),
        "Barru": (-4.4173,119.6270),
        "Mallusetasi": (-4.2651,119.6044),
        "Tanete Rilau": (-4.4126,119.6588)
    }
}

# =========================
# LOCATION INDEX
# =========================
LOCATION_INDEX = {}

def slugify(name):
    return name.lower().replace(" ", "-")

def register(group, parent, data):
    for name,(lat,lon) in data.items():
        LOCATION_INDEX[slugify(name)] = {
            "name": name,
            "group": group,
            "parent": parent,
            "lat": lat,
            "lon": lon
        }

register("default","Nasional",DEFAULT_LOCATIONS)
register("sulsel","Sulawesi Selatan",SULSEL)
for kab,data in KECAMATAN.items():
    register("kecamatan",kab,data)

# =========================
# CACHE
# =========================
CACHE = {}
TTL = 86400

# =========================
# HELPERS
# =========================
def fetch_nasa(lat,lon):
    end = datetime.utcnow()
    start = end - timedelta(days=7)
    url = f"{NASA_URL}?parameters={','.join(FEATURES)}&community=AG&latitude={lat}&longitude={lon}&start={start:%Y%m%d}&end={end:%Y%m%d}&format=JSON"
    r = requests.get(url,timeout=20).json()
    params = r["properties"]["parameter"]
    for d in sorted(params[FEATURES[0]],reverse=True):
        vals = {f:float(params[f][d]) for f in FEATURES}
        if all(v!=-999 for v in vals.values()):
            return d, vals
    return None,None

def interpret(p):
    if p<0.3: return {"status":"Aman","warna":"green"}
    if p<0.6: return {"status":"Waspada","warna":"yellow"}
    return {"status":"Berpotensi Banjir","warna":"red"}

def predict(slug):
    now=time.time()
    if slug in CACHE and now-CACHE[slug]["ts"]<TTL:
        return CACHE[slug]["data"]

    loc=LOCATION_INDEX[slug]
    date,data=fetch_nasa(loc["lat"],loc["lon"])
    df=pd.DataFrame([data],columns=FEATURES)
    prob=float(model.predict_proba(scaler.transform(df))[0][1])
    res={
        "location":loc["name"],
        "group":loc["group"],
        "parent":loc["parent"],
        "date":date,
        "nasa":data,
        "prediction":{"prob":prob},
        "interpretasi":interpret(prob)
    }
    CACHE[slug]={"ts":now,"data":res}
    return res

# =========================
# ROUTES
# =========================
@app.route("/")
def health():
    return jsonify({"status":"ok","service":"TULIP API"})

@app.route("/locations")
def locations():
    return LOCATION_INDEX

@app.route("/predict/<slug>")
def predict_route(slug):
    if slug not in LOCATION_INDEX:
        return jsonify({"error":"Lokasi tidak ditemukan"}),404
    return jsonify(predict(slug))

if __name__=="__main__":
    app.run(host="0.0.0.0",port=int(os.environ.get("PORT",8080)))
