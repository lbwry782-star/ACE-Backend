from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid
import os

app = Flask(__name__)
CORS(app)

# In-memory token store: {token: {"used": bool}}
TOKENS = {}

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/create-token", methods=["POST"])
def create_token():
    # Simple endpoint to create a one-time token (for testing or server-side use)
    token = str(uuid.uuid4())
    TOKENS[token] = {"used": False}
    return jsonify({"token": token})

@app.route("/validate-token", methods=["GET"])
def validate_token():
    token = request.args.get("token", "").strip()
    if not token:
        return jsonify({"valid": False, "reason": "missing token"}), 400

    info = TOKENS.get(token)
    if not info:
        return jsonify({"valid": False, "reason": "unknown token"}), 404
    if info.get("used"):
        return jsonify({"valid": False, "reason": "token already used"}), 403

    return jsonify({"valid": True})

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(silent=True) or {}
    product = (data.get("product") or "").strip()
    description = (data.get("description") or "").strip()
    size = (data.get("size") or "").strip()
    token = (data.get("token") or "").strip()

    if not product:
        return jsonify({"error": "missing product"}), 400

    # Dev override: product '4242' bypasses token enforcement
    dev_mode = (product == "4242")

    if not dev_mode:
        if not token:
            return jsonify({"error": "missing token"}), 400
        info = TOKENS.get(token)
        if not info:
            return jsonify({"error": "unknown token"}), 404
        if info.get("used"):
            return jsonify({"error": "token already used"}), 403

    # Here you would call your real OpenAI logic to generate 3 ads.
    # For now we return placeholder structures compatible with the frontend.
    ads = []
    for i in range(3):
        ads.append({
            "headline": f"Sample Headline {i+1} for {product}",
            "copy": f"This is placeholder marketing copy for {product}. Replace this with real AI-generated text.",
            "image_url": "https://via.placeholder.com/1080x1080.png?text=ACE+Ad",
            "zip_base64": "",  # You can fill this with real ZIP bytes base64 if needed
            "filename": f"ad_{i+1}.zip"
        })

    # Mark token as used AFTER successful generation (non-dev)
    if not dev_mode and token in TOKENS:
        TOKENS[token]["used"] = True

    return jsonify({"ads": ads})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)