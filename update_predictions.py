import os, json, requests, joblib, pandas as pd
from datetime import datetime, timedelta

FEATURES = [
    "PRECTOTCORR","T2M_MIN","T2M_MAX","RH2M",
    "WS2M","WD2M","PS","ALLSKY_SFC_SW_DWN"
]

NASA_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

LOCATIONS = {
    "parepare": (-4.0096, 119.6236),
    "makassar": (-5.1373, 119.4282),
    "pinrang": (-3.7850, 119.6520),
    "barru": (-4.4173, 119.6270)
}

scaler = joblib.load("scaler.pkl")
model = joblib.load("logreg_model.pkl")

os.makedirs("predictions", exist_ok=True)

def fetch_nasa(lat, lon):
    end = datetime.utcnow()
    start = end - timedelta(days=7)
    url = (
        f"{NASA_URL}?parameters={','.join(FEATURES)}"
        f"&community=AG&latitude={lat}&longitude={lon}"
        f"&start={start:%Y%m%d}&end={end:%Y%m%d}&format=JSON"
    )
    r = requests.get(url, timeout=30).json()
    params = r["properties"]["parameter"]
    for d in sorted(params[FEATURES[0]], reverse=True):
        values = {f: float(params[f][d]) for f in FEATURES}
        if all(v != -999 for v in values.values()):
            return d, values
    return None, None

def interpret(prob):
    if prob < 0.3:
        return {"status":"Aman","level":"rendah","warna":"green"}
    if prob < 0.6:
        return {"status":"Waspada","level":"sedang","warna":"yellow"}
    return {"status":"Berpotensi Banjir","level":"tinggi","warna":"red"}

for slug,(lat,lon) in LOCATIONS.items():
    date, data = fetch_nasa(lat, lon)
    if not data:
        continue

    df = pd.DataFrame([data], columns=FEATURES)
    prob = float(model.predict_proba(scaler.transform(df))[0][1])

    output = {
        "lokasi": slug.title(),
        "tanggal_data": date,
        "nasa_data": data,
        "prediction": {
            "probabilitas": prob
        },
        "interpretasi": interpret(prob),
        "updated_at": datetime.utcnow().isoformat()
    }

    with open(f"predictions/{slug}.json", "w") as f:
        json.dump(output, f, indent=2)

print("Prediksi berhasil diperbarui")
