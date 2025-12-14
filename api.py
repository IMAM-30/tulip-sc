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

@app.route("/")
def health():
    return jsonify({
        "status": "ok",
        "service": "TULIP Smart Climate API",
        "mode": "read-only"
    })

@app.route("/predict/<slug>")
def predict(slug):
    data = load_prediction(slug)
    if not data:
        return jsonify({
            "error": "Prediksi belum tersedia",
            "slug": slug
        }), 404
    return jsonify(data)

@app.route("/predict/default")
def predict_default():
    results = {}
    for file in os.listdir(PREDICTION_PATH):
        if file.endswith(".json"):
            with open(os.path.join(PREDICTION_PATH, file)) as f:
                results[file.replace(".json", "")] = json.load(f)
    return jsonify(results)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
