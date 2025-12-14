from flask import Flask, jsonify
from flask_cors import CORS
import os, json

app = Flask(__name__)
CORS(app)

PREDICTION_PATH = "predictions"

def load_prediction(slug):
    path = os.path.join(PREDICTION_PATH, f"{slug}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)

def list_predictions():
    return [
        f.replace(".json", "")
        for f in os.listdir(PREDICTION_PATH)
        if f.endswith(".json")
    ]

@app.route("/")
def health():
    return jsonify({
        "status": "ok",
        "mode": "read-only",
        "service": "TULIP Smart Climate API"
    })

@app.route("/locations")
def locations():
    return jsonify({
        "jumlah": len(list_predictions()),
        "locations": list_predictions()
    })

@app.route("/predict/<slug>")
def predict_slug(slug):
    data = load_prediction(slug)
    if not data:
        return jsonify({
            "error": "Prediksi belum tersedia",
            "slug": slug
        }), 404
    return jsonify(data)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
