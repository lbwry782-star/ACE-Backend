import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)

frontend_url = os.environ.get("FRONTEND_URL")
if frontend_url:
    CORS(app, resources={r"/*": {"origins": [frontend_url]}})
else:
    CORS(app)

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    client = None
else:
    client = OpenAI(api_key=api_key)

TEXT_MODEL = os.environ.get("OPENAI_TEXT_MODEL", "gpt-4.1-mini")
IMAGE_MODEL = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/generate", methods=["POST"])
def generate():
    if not client:
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

    # Only accept true OpenAI sizes
    allowed_sizes = {"1024x1024", "1024x1536", "1536x1024"}
    if size not in allowed_sizes:
        return jsonify({"error": f"Unsupported size '{size}'"}), 400

    try:
        planning_prompt = f"""You are the ACE advertising engine.

Based on the following product and description:
- Infer the target audience (age, lifestyle, key needs).
- Define 3 distinct advertising objectives for 3 different ads.
- For each ad, create:
  * headline: short English headline, 3–7 words
  * copy: exactly 50 English words of persuasive marketing text
  * image_prompt: detailed English prompt for a realistic photographic advertising image
    following the ACE Engine concept (two real objects combined or placed together,
    no logos, no text in the image).

Product: {product}
Description: {description}

Return a JSON object with this structure:
{{
  "ads": [
    {{
      "headline": "...",
      "copy": "...",
      "image_prompt": "..."
    }},
    {{
      "headline": "...",
      "copy": "...",
      "image_prompt": "..."
    }},
    {{
      "headline": "...",
      "copy": "...",
      "image_prompt": "..."
    }}
  ]
}}

Rules:
- copy must be exactly 50 words for each ad.
- Headlines and copy must be in English only.
"""

        completion = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[{"role": "user", "content": planning_prompt}],
            response_format={"type": "json_object"},
        )

        content = completion.choices[0].message.content
        plan = json.loads(content)
        ads_plan = plan.get("ads") or []

        ads_out = []

        for ad in ads_plan[:3]:
            headline = (ad.get("headline") or "").strip()
            copy_text = (ad.get("copy") or "").strip()
            img_prompt = (ad.get("image_prompt") or "").strip()

            if not img_prompt:
                img_prompt = (
                    f"High-quality photographic advertising image for {product}. "
                    "No logos, no text in the image, realistic lighting."
                )

            img_resp = client.images.generate(
                model=IMAGE_MODEL,
                prompt=img_prompt,
                size=size,
                n=1
            )

            b64_data = img_resp.data[0].b64_json

            ads_out.append({
                "headline": headline,
                "copy": copy_text,
                "image_base64": b64_data,
            })

        if len(ads_out) == 0:
            return jsonify({"error": "No ads generated from text model"}), 500

        return jsonify({"ads": ads_out, "size_used": size}), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal generation error", "details": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
