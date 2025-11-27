import os
import json
import datetime
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

FRONTEND_URL = os.environ.get("FRONTEND_URL", "*")

if FRONTEND_URL == "*" or not FRONTEND_URL:
    CORS(app)
else:
    CORS(app, resources={r"/*": {"origins": [FRONTEND_URL]}})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/generate", methods=["POST"])
def generate():
    """Placeholder generate endpoint – returns 3 fake variations so frontend works.
    Later we can replace this with real OpenAI logic.
    """
    try:
        data = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"success": False, "error": "Request body must be JSON"}), 400

    product = (data.get("product") or "").strip()
    description = (data.get("description") or "").strip()
    size = (data.get("size") or "1024x1024").strip()

    if not product:
        return jsonify({"success": False, "error": 'Missing "product" in request body.'}), 400

    # create 3 simple fake variations
    variations = []
    for i in range(1, 4):
        headline = f"Sample headline {i}"
        marketing_copy = (
            "This is placeholder marketing copy for demo purposes only. "
            "Your real OpenAI-powered ad engine will generate photographic hybrid "
            "ads and exactly fifty-word copy once the final backend is connected."
        )
        # ensure exactly 50 words
        words = marketing_copy.split()
        if len(words) > 50:
            words = words[:50]
        elif len(words) < 50:
            words += ["demo"] * (50 - len(words))
        marketing_copy = " ".join(words)

        uid = uuid.uuid4().hex[:8]
        variations.append({
            "id": uid,
            "headline": headline,
            "marketing_copy": marketing_copy,
            "image_url": "",  # no real image yet
            "zip_url": ""
        })

    return jsonify({"success": True, "variations": variations})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
