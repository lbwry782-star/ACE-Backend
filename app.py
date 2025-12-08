import os
from flask import Flask, request, jsonify

app = Flask(__name__)

# ---- Very permissive CORS on ALL responses (debug only) ----
@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@app.route("/health", methods=["GET", "OPTIONS"])
def health():
    if request.method == "OPTIONS":
        return ("", 204)
    return jsonify({"status": "ok", "mode": "debug-backend-wildcard-cors"})


@app.route("/generate", methods=["POST", "OPTIONS"])
def generate():
    # Handle CORS preflight explicitly
    if request.method == "OPTIONS":
        return ("", 204)

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
                    "This is a DEBUG response from the backend with WILDCARD CORS enabled. "
                    "The ACE Engine is not running here yet, but the "
                    "connection between frontend and backend is verified."
                ),
                "image_base64": ""  # placeholder
            }
        )

    return jsonify({"mode": "debug", "ads": ads}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
