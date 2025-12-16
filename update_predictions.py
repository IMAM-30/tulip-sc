import os, json, time, requests
import pandas as pd
import joblib
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "predictions")
os.makedirs(OUT_DIR, exist_ok=True)

scaler = joblib.load("scaler.pkl")
model = joblib.load("logreg_model.pkl")

FEATURES = [
    "PRECTOTCORR","T2M_MIN","T2M_MAX","RH2M",
    "WS2M","WD2M","PS","ALLSKY_SFC_SW_DWN"
]

NASA_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

from locations import LOCATION_INDEX

def fetch_valid(lat, lon, retry=3):
    """Fetch NASA data with retry mechanism"""
    end = datetime.utcnow()
    start = end - timedelta(days=7)
    
    for attempt in range(retry):
        try:
            url = (
                f"{NASA_URL}?parameters={','.join(FEATURES)}"
                f"&community=AG&latitude={lat}&longitude={lon}"
                f"&start={start:%Y%m%d}&end={end:%Y%m%d}&format=JSON"
            )
            
            logger.info(f"Fetching NASA data for ({lat},{lon}) - attempt {attempt+1}")
            r = requests.get(url, timeout=30).json()
            
            if "properties" not in r or "parameter" not in r["properties"]:
                logger.warning(f"No valid data in response for ({lat},{lon})")
                return None, None
                
            params = r["properties"]["parameter"]
            
            # Cari data terbaru yang valid
            for d in sorted(params[FEATURES[0]], reverse=True):
                vals = {f: float(params[f][d]) for f in FEATURES}
                if all(v != -999 for v in vals.values()):
                    logger.info(f"Found valid data for ({lat},{lon}) on date {d}")
                    return d, vals
            
            logger.warning(f"No valid data found for ({lat},{lon}) in date range")
            return None, None
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt+1} failed: {e}")
            if attempt < retry - 1:
                wait_time = 5 * (attempt + 1)
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
    
    logger.error(f"All {retry} attempts failed for ({lat},{lon})")
    return None, None

def interpret(prob):
    if prob < 0.3:
        return {"status":"Aman","warna":"green","level":"low"}
    if prob < 0.6:
        return {"status":"Waspada","warna":"yellow","level":"medium"}
    return {"status":"Berpotensi Banjir","warna":"red","level":"high"}

def main():
    """Main update function"""
    logger.info("=" * 50)
    logger.info("STARTING PREDICTION UPDATE")
    logger.info(f"Total locations: {len(LOCATION_INDEX)}")
    logger.info("=" * 50)
    
    updated_count = 0
    failed_count = 0
    start_time = time.time()
    
    for slug, loc in LOCATION_INDEX.items():
        try:
            logger.info(f"Processing {slug} ({loc['name']})...")
            
            date, data = fetch_valid(loc["lat"], loc["lon"])
            if not data:
                logger.warning(f"No data available for {slug}")
                failed_count += 1
                continue
            
            # Prepare features
            df = pd.DataFrame([data], columns=FEATURES)
            
            # Predict
            prob = float(model.predict_proba(scaler.transform(df))[0][1])
            
            result = {
                "slug": slug,
                "location": loc["name"],
                "group": loc["group"],
                "parent": loc["parent"],
                "date": date,
                "nasa": data,
                "prediction": {
                    "probabilitas": prob,
                    "percentage": round(prob * 100, 1)
                },
                "interpretasi": interpret(prob),
                "updated_at": datetime.utcnow().isoformat(),
                "coordinates": {
                    "lat": loc["lat"],
                    "lon": loc["lon"]
                }
            }
            
            # Save to file
            output_path = os.path.join(OUT_DIR, f"{slug}.json")
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)
            
            updated_count += 1
            logger.info(f"✅ Updated {slug}: {prob:.1%} ({interpret(prob)['status']})")
            
            # Rate limiting (NASA mungkin punya limit)
            time.sleep(1.5)
            
        except Exception as e:
            failed_count += 1
            logger.error(f"❌ Failed {slug}: {e}")
    
    elapsed_time = time.time() - start_time
    logger.info("=" * 50)
    logger.info("UPDATE COMPLETED")
    logger.info(f"Success: {updated_count}")
    logger.info(f"Failed:  {failed_count}")
    logger.info(f"Elapsed: {elapsed_time:.1f} seconds")
    logger.info("=" * 50)
    
    # Return summary for API
    return {
        "success": updated_count,
        "failed": failed_count,
        "total": len(LOCATION_INDEX),
        "elapsed_seconds": round(elapsed_time, 1),
        "completed_at": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2))