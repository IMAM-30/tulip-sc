import os, json, time, requests
import pandas as pd
import joblib
from datetime import datetime, timedelta
import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('update.log')
    ]
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "predictions")
os.makedirs(OUT_DIR, exist_ok=True)

# Load model and scaler
try:
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("logreg_model.pkl")
    logger.info("✅ Model and scaler loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load model/scaler: {e}")
    sys.exit(1)

FEATURES = [
    "PRECTOTCORR", "T2M_MIN", "T2M_MAX", "RH2M",
    "WS2M", "WD2M", "PS", "ALLSKY_SFC_SW_DWN"
]

NASA_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

from locations import LOCATION_INDEX

def slugify(name):
    """Simple slugify function"""
    return name.lower().replace(" ", "-")

def fetch_valid(lat, lon, retry=3):
    """Fetch NASA data with retry mechanism"""
    end = datetime.utcnow()
    start = end - timedelta(days=10)
    
    for attempt in range(retry):
        try:
            params = {
                "parameters": ",".join(FEATURES),
                "community": "AG",
                "latitude": lat,
                "longitude": lon,
                "start": start.strftime("%Y%m%d"),
                "end": end.strftime("%Y%m%d"),
                "format": "JSON"
            }
            
            logger.info(f"🌍 Fetching NASA data - attempt {attempt+1}")
            
            headers = {
                "User-Agent": "TULIP-SC/1.0",
                "Accept": "application/json"
            }
            
            response = requests.get(
                NASA_URL,
                params=params,
                headers=headers,
                timeout=60
            )
            
            if response.status_code != 200:
                logger.error(f"❌ NASA API error {response.status_code}")
                if response.status_code == 429:
                    logger.warning("⚠️ Rate limited, waiting...")
                    time.sleep(30)
                continue
                
            r = response.json()
            
            if "properties" not in r or "parameter" not in r["properties"]:
                logger.warning("⚠️ No parameters in NASA response")
                continue
                
            params_data = r["properties"]["parameter"]
            
            if not params_data or FEATURES[0] not in params_data:
                logger.warning(f"⚠️ No {FEATURES[0]} data in response")
                continue
            
            available_dates = list(params_data[FEATURES[0]].keys())
            logger.info(f"📅 Available dates: {len(available_dates)} total")
            
            if not available_dates:
                logger.warning("⚠️ No dates available")
                return None, None
            
            # Log 3 tanggal terbaru
            latest_dates = sorted(available_dates, reverse=True)[:3]
            logger.info(f"📆 Latest dates: {latest_dates}")
            
            # Cari data terbaru yang valid
            for d in sorted(available_dates, reverse=True):
                try:
                    vals = {}
                    valid = True
                    
                    for f in FEATURES:
                        value = params_data[f].get(d)
                        if value is None or value == -999:
                            valid = False
                            break
                        vals[f] = float(value)
                    
                    if valid:
                        data_date = datetime.strptime(d, "%Y%m%d")
                        days_ago = (datetime.utcnow() - data_date).days
                        
                        logger.info(f"📊 Data {d}: {days_ago} days ago")
                        
                        if days_ago <= 7:
                            logger.info(f"✅ Valid data for {d} ({days_ago} days ago)")
                            return d, vals
                        else:
                            logger.warning(f"⚠️ Data old: {d} ({days_ago} days ago)")
                            return d, vals
                            
                except (KeyError, ValueError, TypeError) as e:
                    logger.debug(f"Skip date {d}: {e}")
                    continue
            
            logger.warning(f"⚠️ No valid data in {len(available_dates)} dates")
            return None, None
            
        except requests.exceptions.RequestException as e:
            logger.error(f"🌐 Request failed (attempt {attempt+1}): {e}")
            if attempt < retry - 1:
                wait = 15 * (attempt + 1)
                logger.info(f"⏳ Waiting {wait}s before retry...")
                time.sleep(wait)
        except Exception as e:
            logger.error(f"🔥 Unexpected error: {e}")
            if attempt < retry - 1:
                time.sleep(10)
    
    logger.error(f"❌ All {retry} attempts failed")
    return None, None

def fetch_valid_with_fallback(lat, lon, location_name, slug):
    """Fetch NASA data with fallback to existing data"""
    date, data = fetch_valid(lat, lon)
    
    if not data:
        logger.warning(f"⚠️ No new NASA data for {location_name}")
        
        existing_file = os.path.join(OUT_DIR, f"{slug}.json")
        if os.path.exists(existing_file):
            try:
                with open(existing_file, "r") as f:
                    existing = json.load(f)
                    existing_date = existing.get("date")
                    existing_data = existing.get("nasa")
                    
                    if existing_date and existing_data:
                        try:
                            existing_dt = datetime.strptime(existing_date, "%Y%m%d")
                            days_old = (datetime.utcnow() - existing_dt).days
                            logger.info(f"↩️ Using existing data ({days_old} days old)")
                        except:
                            logger.info(f"↩️ Using existing data")
                        
                        return existing_date, existing_data
            except Exception as e:
                logger.error(f"Error loading existing data: {e}")
        
        return None, None
    
    return date, data

def interpret(prob):
    """Interpret probability to human readable format"""
    if prob < 0.3:
        return {"status": "Aman", "warna": "green", "level": "low"}
    elif prob < 0.6:
        return {"status": "Waspada", "warna": "yellow", "level": "medium"}
    else:
        return {"status": "Berpotensi Banjir", "warna": "red", "level": "high"}

def main():
    """Main update function"""
    logger.info("=" * 60)
    logger.info("🚀 STARTING PREDICTION UPDATE")
    logger.info(f"📊 Total locations: {len(LOCATION_INDEX)}")
    logger.info(f"📁 Output directory: {OUT_DIR}")
    logger.info(f"⏰ Current time: {datetime.utcnow().isoformat()}Z")
    logger.info("=" * 60)
    
    updated_count = 0
    failed_count = 0
    skipped_count = 0
    start_time = time.time()
    
    total_locations = len(LOCATION_INDEX)
    
    for idx, (slug, loc) in enumerate(LOCATION_INDEX.items(), 1):
        try:
            logger.info(f"📍 [{idx}/{total_locations}] {loc['name']} ({slug})...")
            
            # Fetch data
            date, data = fetch_valid_with_fallback(
                loc["lat"], 
                loc["lon"], 
                loc["name"],
                slug
            )
            
            if not data:
                logger.warning(f"⏭️ Skipping {slug}: no data")
                skipped_count += 1
                continue
            
            # Prepare features and predict
            df = pd.DataFrame([data], columns=FEATURES)
            
            try:
                scaled_features = scaler.transform(df)
                prob = float(model.predict_proba(scaled_features)[0][1])
            except Exception as e:
                logger.error(f"❌ Prediction error: {e}")
                failed_count += 1
                continue
            
            # Calculate percentage
            percentage = round(prob * 100, 1)
            
            # Create result
            result = {
                "slug": slug,
                "location": loc["name"],
                "group": loc["group"],
                "parent": loc["parent"],
                "date": date,
                "nasa": data,
                "prediction": {
                    "probabilitas": prob,
                    "percentage": percentage
                },
                "interpretasi": interpret(prob),
                "updated_at": datetime.utcnow().isoformat() + "Z",
                "coordinates": {
                    "lat": loc["lat"],
                    "lon": loc["lon"]
                },
                "metadata": {
                    "model_version": "v1.0",
                    "features_used": FEATURES,
                    "data_source": "NASA POWER"
                }
            }
            
            # Calculate data age
            try:
                data_date = datetime.strptime(date, "%Y%m%d")
                data_age = datetime.utcnow() - data_date
                result["data_age_days"] = data_age.days
                result["metadata"]["data_age_days"] = data_age.days
            except:
                pass
            
            # Save to file
            output_path = os.path.join(OUT_DIR, f"{slug}.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            updated_count += 1
            logger.info(f"✅ Updated {slug}: {percentage}% ({interpret(prob)['status']})")
            
            # Rate limiting
            if idx < total_locations and idx % 5 == 0:
                sleep_time = 3.0
                logger.debug(f"⏳ Sleeping {sleep_time}s...")
                time.sleep(sleep_time)
            
        except Exception as e:
            failed_count += 1
            logger.error(f"❌ Failed {slug}: {e}")
    
    elapsed_time = time.time() - start_time
    
    logger.info("=" * 60)
    logger.info("🎯 UPDATE COMPLETED")
    logger.info(f"✅ Success: {updated_count}")
    logger.info(f"⚠️ Skipped: {skipped_count}")
    logger.info(f"❌ Failed:  {failed_count}")
    logger.info(f"⏱️ Elapsed: {elapsed_time:.1f} seconds")
    logger.info("=" * 60)
    
    # Return summary
    summary = {
        "success": updated_count,
        "skipped": skipped_count,
        "failed": failed_count,
        "total": total_locations,
        "elapsed_seconds": round(elapsed_time, 1),
        "completed_at": datetime.utcnow().isoformat() + "Z",
        "status": "success" if updated_count > 0 else "partial" if skipped_count > 0 else "failed"
    }
    
    # Save summary
    summary_path = os.path.join(BASE_DIR, "update_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    return summary

if __name__ == "__main__":
    try:
        result = main()
        print(json.dumps(result, indent=2))
    except KeyboardInterrupt:
        logger.info("Update interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)