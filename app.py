import os
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "mode": "debug-backend"})

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(force=True, silent=True) or {}
    product = (data.get("product") or "").strip()
    description = (data.get("description") or "").strip()
    size = (data.get("size") or "").strip()

    if not product:
        return jsonify({"error": "Missing 'product' in request body"}), 400

    ads = []
    for i in range(3):
        ads.append(
            {
                "headline": f"Sample headline {i+1} for {product}",
                "copy": (
                    "This is a DEBUG response from the backend. "
                    "The ACE Engine is not running here yet, but the "
                    "frontend, token and override flow are verified."
                ),
                # Empty placeholder base64 – image will not show, but JSON is valid.
                "image_base64": ""
            }
        )

    return jsonify({"mode": "debug", "ads": ads}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
