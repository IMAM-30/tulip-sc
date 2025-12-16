from flask import Flask, jsonify, request
from flask_cors import CORS
import os, json, threading, time, schedule
from datetime import datetime, timedelta
import subprocess
import traceback

app = Flask(__name__)
CORS(app)

PREDICTION_PATH = "predictions"

# ========== AUTO-UPDATE MECHANISM ==========
UPDATE_INTERVAL_HOURS = 6
LAST_UPDATE = None
UPDATE_IN_PROGRESS = False
UPDATE_LOCK = threading.Lock()

def update_predictions_background():
    global UPDATE_IN_PROGRESS, LAST_UPDATE
    
    if not UPDATE_LOCK.acquire(blocking=False):
        print("⚠️ Update already running, skipping...")
        return
    
    UPDATE_IN_PROGRESS = True
    try:
        print(f"🔄 [{datetime.now()}] Starting update...")
        
        script_path = os.path.join(os.path.dirname(__file__), "update_predictions.py")
        
        if not os.path.exists(script_path):
            print(f"❌ Script not found: {script_path}")
            return
        
        result = subprocess.run(
            ["python", script_path],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=os.path.dirname(__file__)
        )
        
        if result.returncode == 0:
            print(f"✅ [{datetime.now()}] Update successful!")
            print("Output:", result.stdout[:200])
            LAST_UPDATE = datetime.now().isoformat() + "Z"
            
            try:
                output_data = json.loads(result.stdout)
                success = output_data.get('success', 0)
                failed = output_data.get('failed', 0)
                print(f"📊 Summary: {success} success, {failed} failed")
            except:
                pass
        else:
            print(f"❌ [{datetime.now()}] Update failed!")
            print("Error:", result.stderr[:200])
            
    except subprocess.TimeoutExpired:
        print(f"⏰ [{datetime.now()}] Update timeout!")
    except Exception as e:
        print(f"🔥 [{datetime.now()}] Update error: {e}")
    finally:
        UPDATE_IN_PROGRESS = False
        UPDATE_LOCK.release()

def scheduler_worker():
    print(f"⏰ Scheduler started. Updates every {UPDATE_INTERVAL_HOURS} hours")
    
    schedule.every(UPDATE_INTERVAL_HOURS).hours.do(update_predictions_background)
    
    print("⏳ Waiting 30 seconds before initial update...")
    time.sleep(30)
    update_predictions_background()
    
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)
        except Exception as e:
            print(f"⚠️ Scheduler error: {e}")
            time.sleep(60)

# ========== API ENDPOINTS ==========
def load_prediction(slug):
    path = os.path.join(PREDICTION_PATH, f"{slug}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {slug}: {e}")
        return None

def list_predictions():
    try:
        return [
            f.replace(".json", "")
            for f in os.listdir(PREDICTION_PATH)
            if f.endswith(".json")
        ]
    except FileNotFoundError:
        return []

@app.route("/")
def health():
    from locations import LOCATION_INDEX
    
    status_info = {
        "status": "ok",
        "mode": "auto-update-enabled",
        "service": "TULIP Smart Climate API",
        "version": "2.3-laravel",
        "update_interval_hours": UPDATE_INTERVAL_HOURS,
        "last_update": LAST_UPDATE,
        "update_in_progress": UPDATE_IN_PROGRESS,
        "total_locations": len(LOCATION_INDEX),
        "locations_available": len(list_predictions()),
        "server_time": datetime.now().isoformat(),
        "compatible_with_laravel": True,
        "endpoints": {
            "/": "Health check",
            "/locations": "List all locations",
            "/predict/<slug>": "Get prediction for location",
            "/force-update": "Force update predictions",
            "/update-status": "Check update status",
            "/debug-update": "Debug update script",
            "/laravel-locations": "Get locations compatible with Laravel"
        }
    }
    return jsonify(status_info)

@app.route("/locations")
def locations():
    locations_list = list_predictions()
    return jsonify({
        "jumlah": len(locations_list),
        "locations": locations_list,
        "last_update": LAST_UPDATE,
        "server_time": datetime.now().isoformat()
    })

@app.route("/laravel-locations")
def laravel_locations():
    """Endpoint khusus untuk Laravel - return locations dalam format Laravel"""
    from locations import LOCATION_INDEX
    
    laravel_format = []
    for slug, loc in LOCATION_INDEX.items():
        laravel_format.append({
            "value": slug,
            "text": f"{loc['name']} ({loc['parent']})"
        })
    
    return jsonify({
        "locations": laravel_format,
        "count": len(laravel_format),
        "compatible": True
    })

@app.route("/predict/<slug>")
def predict_slug(slug):
    from datetime import datetime
    
    if not slug or slug == "undefined":
        return jsonify({"error": "Invalid location slug"}), 400
    
    data = load_prediction(slug)
    if not data:
        all_locations = list_predictions()
        possible_matches = [loc for loc in all_locations if slug in loc]
        
        if possible_matches:
            return jsonify({
                "error": "Location not found",
                "requested_slug": slug,
                "suggestions": possible_matches[:5]
            }), 404
        
        return jsonify({
            "error": "Prediction not available",
            "slug": slug,
            "available_locations": all_locations[:10]
        }), 404
    
    # Add metadata
    data["api_version"] = "2.3-laravel"
    data["retrieved_at"] = datetime.now().isoformat()
    
    # Calculate data age
    if "date" in data and data["date"] and "data_age_days" not in data:
        try:
            data_date = datetime.strptime(data["date"], "%Y%m%d")
            data_age = datetime.utcnow() - data_date
            data["data_age_days"] = data_age.days
        except:
            pass
    
    return jsonify(data)

@app.route("/force-update", methods=["POST"])
def force_update():
    if UPDATE_IN_PROGRESS:
        return jsonify({
            "status": "busy",
            "message": "Update already in progress",
            "last_update": LAST_UPDATE,
            "timestamp": datetime.now().isoformat()
        }), 409
    
    thread = threading.Thread(target=update_predictions_background, daemon=True)
    thread.start()
    
    return jsonify({
        "status": "started",
        "message": "Update started in background",
        "timestamp": datetime.now().isoformat()
    })

@app.route("/update-status")
def update_status():
    next_update = None
    next_update_in = None
    
    if LAST_UPDATE:
        try:
            last_update_dt = datetime.fromisoformat(LAST_UPDATE.replace('Z', '+00:00'))
            next_update_dt = last_update_dt + timedelta(hours=UPDATE_INTERVAL_HOURS)
            next_update = next_update_dt.isoformat()
            
            now = datetime.utcnow()
            if next_update_dt > now:
                next_update_in = round((next_update_dt - now).total_seconds() / 3600, 1)
        except:
            pass
    
    return jsonify({
        "last_update": LAST_UPDATE,
        "update_in_progress": UPDATE_IN_PROGRESS,
        "update_interval_hours": UPDATE_INTERVAL_HOURS,
        "next_update": next_update,
        "next_update_in_hours": next_update_in,
        "total_predictions": len(list_predictions()),
        "timestamp": datetime.now().isoformat()
    })

@app.route("/debug-update")
def debug_update():
    try:
        script_path = os.path.join(os.path.dirname(__file__), "update_predictions.py")
        
        predictions_files = []
        if os.path.exists(PREDICTION_PATH):
            predictions_files = os.listdir(PREDICTION_PATH)
        
        return jsonify({
            "script_exists": os.path.exists(script_path),
            "predictions_folder_exists": os.path.exists(PREDICTION_PATH),
            "predictions_count": len([f for f in predictions_files if f.endswith(".json")]),
            "sample_files": predictions_files[:5],
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ========== STARTUP ==========
def initialize_background_tasks():
    print("=" * 60)
    print("🚀 TULIP Smart Climate API v2.3 (Laravel Compatible)")
    print(f"📅 Auto-update every {UPDATE_INTERVAL_HOURS} hours")
    print(f"📁 Predictions path: {os.path.abspath(PREDICTION_PATH)}")
    
    from locations import LOCATION_INDEX
    print(f"📊 Total locations configured: {len(LOCATION_INDEX)}")
    
    print("\n📡 Available endpoints:")
    print("  - GET  /                   # Health check")
    print("  - GET  /locations          # List locations")
    print("  - GET  /laravel-locations  # Laravel format locations")
    print("  - GET  /predict/<slug>     # Get prediction")
    print("  - POST /force-update       # Manual update")
    print("  - GET  /update-status      # Check update status")
    print("=" * 60)
    
    scheduler_thread = threading.Thread(target=scheduler_worker, daemon=True)
    scheduler_thread.start()
    
    print("✅ Background scheduler started")

with app.app_context():
    initialize_background_tasks()

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    print(f"🌐 Starting server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)