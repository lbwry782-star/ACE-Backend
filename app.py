import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai

app = Flask(__name__)

# CORS
frontend_url = os.environ.get("FRONTEND_URL")
if frontend_url:
    CORS(app, resources={r"/*": {"origins": [frontend_url]}})
else:
    CORS(app)

# OpenAI legacy client
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    openai.api_key = None
else:
    openai.api_key = api_key
    openai.timeout = 120  # seconds

IMAGE_MODEL = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1")
ALLOWED_SIZES = {"1024x1024", "1024x1536", "1536x1024"}


def build_copy_50_words(product, description):
    base = (product + ". " + description).strip()
    filler = (
        "Discover the benefits today and boost your results with this "
        "automated creative advertising engine designed for powerful "
        "social media campaigns and engaging visual storytelling worldwide."
    )
    words = (base + " " + filler).split()
    if len(words) < 50:
        extra = (filler + " ") * 5
        words = (base + " " + extra).split()
    return " ".join(words[:50])


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/generate", methods=["POST"])
def generate():
    if openai.api_key is None:
        return jsonify({"error": "OPENAI_API_KEY missing on server"}), 500

    data = request.get_json(silent=True) or {}

    token = data.get("token", None)
    if not token:
        return jsonify({"error": "Token missing or false. Generation is not allowed."}), 403

    product = (data.get("product") or "").strip()
    description = (data.get("description") or "").strip()
    size = (data.get("size") or "").strip() or "1024x1024"

    if not product:
        return jsonify({"error": "Missing 'product' in request body"}), 400
    if not description:
        return jsonify({"error": "Missing 'description' in request body"}), 400
    if size not in ALLOWED_SIZES:
        return jsonify({"error": f"Unsupported size '{size}'"}), 400

    try:
        # Simple prompt
        prompt = (
            f"High-quality photographic advertising image for product '{product}'. "
            f"Description: {description}. "
            "Two real objects combined or placed side by side in a clever way, "
            "no logos, no written text inside the image, realistic lighting, "
            "suitable for a professional social media campaign."
        )

        # Exactly like the shell test: one image only
        img_resp = openai.Image.create(
            model=IMAGE_MODEL,
            prompt=prompt,
            size=size,
            n=1
        )

        images_data = img_resp.get("data", [])
        if not images_data:
            return jsonify({"error": "Image model returned no data"}), 500

        # Use the single image for all 3 ads (temporary stabilization)
        first_item = images_data[0]
        b64_data = first_item.get("b64_json")

        ads_out = []
        for idx in range(3):
            headline = f"ACE for {product}"[:60]
            copy_text = build_copy_50_words(product, description)
            ad_payload = {
                "headline": headline,
                "copy": copy_text,
            }
            if b64_data:
                ad_payload["image_base64"] = b64_data
            ads_out.append(ad_payload)

        return jsonify({"ads": ads_out, "size_used": size}), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal generation error", "details": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
