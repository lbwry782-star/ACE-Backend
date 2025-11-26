
import os
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)

FRONTEND_URL = os.environ.get("FRONTEND_URL")
if FRONTEND_URL:
    CORS(app, resources={r"/*": {"origins": [FRONTEND_URL]}})
else:
    CORS(app)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

ALLOWED_SIZES = {"1024x1024", "1024x1792", "1792x1024"}


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON body"}), 400

    product = (data.get("product") or "").strip()
    size = (data.get("size") or "1024x1024").strip()

    if not product:
        return jsonify({"error": "Missing 'product' in request body"}), 400

    if size not in ALLOWED_SIZES:
        size = "1024x1024"

    image_model = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1")
    text_model = os.environ.get("OPENAI_TEXT_MODEL", "gpt-4.1-mini")

    # Build prompts
    copy_prompt = (
        "You are an advertising copywriter. "
        "Write 3 distinct short marketing texts for a social media image ad. "
        "Each text must be exactly 50 English words. "
        "Do not number them. Separate the 3 texts using a single line with three hash symbols (###).\n\n"
        f"Product: {product}\n"
        "Audience: inferred from the product. Focus on benefits, clarity and a friendly, persuasive tone."
    )

    try:
        # Generate marketing copy
        chat_resp = client.chat.completions.create(
            model=text_model,
            messages=[
                {"role": "system", "content": "You write concise, high‑impact English marketing copy."},
                {"role": "user", "content": copy_prompt},
            ],
        )
        raw_text = chat_resp.choices[0].message.content.strip()
        parts = [p.strip() for p in raw_text.split("###") if p.strip()]
        # Ensure exactly 3 pieces
        if len(parts) < 3:
            # If fewer than 3, pad by repeating
            while len(parts) < 3:
                parts.append(parts[-1])
        copies = parts[:3]

        # Generate images
        img_prompt = (
            "Photorealistic social media image ad for the following product: "
            f"{product}. Modern, clean composition on a neutral or black background, "
            "optimized for readability of overlaid headline text (the text will be added separately). "
            "Do not include any text in the image."
        )

        img_resp = client.images.generate(
            model=image_model,
            prompt=img_prompt,
            n=3,
            size=size,
            response_format="b64_json",
        )

        results = []
        for i, item in enumerate(img_resp.data):
            b64_img = item.b64_json
            results.append(
                {
                    "index": i,
                    "image_b64": b64_img,
                    "copy": copies[i],
                    "size": size,
                }
            )

        return jsonify({"success": True, "results": results}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
