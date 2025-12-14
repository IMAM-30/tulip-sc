import os, json, time, requests
import pandas as pd
import joblib
from datetime import datetime, timedelta

# =========================
# CONFIG
# =========================
PREDICTION_PATH = "predictions"
os.makedirs(PREDICTION_PATH, exist_ok=True)

FEATURES = [
    "PRECTOTCORR","T2M_MIN","T2M_MAX","RH2M",
    "WS2M","WD2M","PS","ALLSKY_SFC_SW_DWN"
]

NASA_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# =========================
# LOAD MODEL
# =========================
scaler = joblib.load("scaler.pkl")
model = joblib.load("logreg_model.pkl")
print("✅ Model & scaler loaded")

# =========================
# LOCATIONS (FULL)
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
# HELPERS
# =========================
def slugify(name):
    return name.lower().replace(" ", "-")

def fetch_nasa(lat, lon):
    end = datetime.utcnow()
    start = end - timedelta(days=7)

    url = (
        f"{NASA_URL}?parameters={','.join(FEATURES)}"
        f"&community=AG"
        f"&latitude={lat}&longitude={lon}"
        f"&start={start:%Y%m%d}&end={end:%Y%m%d}&format=JSON"
    )

    r = requests.get(url, timeout=20).json()
    params = r["properties"]["parameter"]

    for d in sorted(params[FEATURES[0]], reverse=True):
        vals = {f: float(params[f][d]) for f in FEATURES}
        if all(v != -999 for v in vals.values()):
            return d, vals
    return None, None

def interpret(prob):
    if prob < 0.3:
        return {"status":"Aman","level":"rendah","warna":"green"}
    elif prob < 0.6:
        return {"status":"Waspada","level":"sedang","warna":"yellow"}
    return {"status":"Berpotensi Banjir","level":"tinggi","warna":"red"}

def save_prediction(slug, data):
    path = os.path.join(PREDICTION_PATH, f"{slug}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# =========================
# MAIN UPDATE
# =========================
def process_location(name, lat, lon, group, parent):
    print(f"⏳ Processing {name}")
    date, data = fetch_nasa(lat, lon)
    if not data:
        print(f"⚠️ No valid NASA data for {name}")
        return

    df = pd.DataFrame([data], columns=FEATURES)
    prob = float(model.predict_proba(scaler.transform(df))[0][1])

    result = {
        "location": name,
        "slug": slugify(name),
        "group": group,
        "parent": parent,
        "date": date,
        "nasa": data,
        "prediction": {
            "probabilitas": prob
        },
        "interpretasi": interpret(prob),
        "updated_at": datetime.utcnow().isoformat()
    }

    save_prediction(slugify(name), result)
    print(f"✅ Saved {name}")

# =========================
# EXECUTE ALL
# =========================
if __name__ == "__main__":

    for name,(lat,lon) in DEFAULT_LOCATIONS.items():
        process_location(name, lat, lon, "default", "Nasional")

    for name,(lat,lon) in SULSEL.items():
        process_location(name, lat, lon, "sulsel", "Sulawesi Selatan")

    for kab, kecs in KECAMATAN.items():
        for name,(lat,lon) in kecs.items():
            process_location(name, lat, lon, "kecamatan", kab)

    print("🎉 ALL PREDICTIONS UPDATED")
