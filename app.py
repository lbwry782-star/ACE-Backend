import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai

app = Flask(__name__)

# CORS — allow your domain; fallback * if not set
frontend_url = os.environ.get("FRONTEND_URL")
if frontend_url:
    CORS(app, resources={r"/*": {"origins": [frontend_url]}})
else:
    CORS(app)

# OpenAI legacy client with explicit timeout
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    openai.api_key = None
else:
    openai.api_key = api_key
    openai.timeout = 120  # seconds

TEXT_MODEL = os.environ.get("OPENAI_TEXT_MODEL", "gpt-4.1-mini")
IMAGE_MODEL = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1")

ALLOWED_SIZES = {"1024x1024", "1024x1536", "1536x1024"}


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/generate", methods=["POST"])
def generate():
    if openai.api_key is None:
        return jsonify({"error": "OPENAI_API_KEY missing on server"}), 500

    data = request.get_json(silent=True) or {}

    # Token enforcement
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
        # ---- STEP 1: text planning ----
        planning_prompt = (
            "You are the ACE advertising engine.\n\n"
            "Based on the following product and description:\n"
            "- Infer the target audience (age, lifestyle, key needs).\n"
            "- Define 3 distinct advertising objectives for 3 different ads.\n"
            "- For each ad, create:\n"
            "  * headline: short English headline, 3–7 words\n"
            "  * copy: exactly 50 English words of persuasive marketing text\n"
            "  * image_prompt: detailed English prompt for a realistic photographic advertising image\n"
            "    following the ACE Engine concept (two real objects combined or placed together,\n"
            "    no logos, no text inside the image).\n\n"
            f"Product: {product}\n"
            f"Description: {description}\n\n"
            "Return a JSON object with this structure:\n"
            "{\n"
            '  \"ads\": [\n'
            "    {\n"
            '      \"headline\": \"...\",\n'
            '      \"copy\": \"...\",\n'
            '      \"image_prompt\": \"...\"\n'
            "    },\n"
            "    {\n"
            '      \"headline\": \"...\",\n'
            '      \"copy\": \"...\",\n'
            '      \"image_prompt\": \"...\"\n'
            "    },\n"
            "    {\n"
            '      \"headline\": \"...\",\n'
            '      \"copy\": \"...\",\n'
            '      \"image_prompt\": \"...\"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Rules:\n"
            "- copy must be exactly 50 words for each ad.\n"
            "- Headlines and copy must be in English only.\n"
        )

        chat_resp = openai.ChatCompletion.create(
            model=TEXT_MODEL,
            messages=[{"role": "user", "content": planning_prompt}],
            temperature=0.7,
        )

        content = chat_resp["choices"][0]["message"]["content"]
        # Try to locate JSON in the response
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = content[start : end + 1]
        else:
            json_str = content

        try:
            plan = json.loads(json_str)
        except Exception:
            # Fallback: simple structure if parsing fails
            plan = {
                "ads": [
                    {"headline": product, "copy": description, "image_prompt": ""},
                    {"headline": product, "copy": description, "image_prompt": ""},
                    {"headline": product, "copy": description, "image_prompt": ""},
                ]
            }

        ads_plan = plan.get("ads") or []

        # Guarantee we have 3 slots
        while len(ads_plan) < 3:
            ads_plan.append(
                {
                    "headline": product,
                    "copy": description,
                    "image_prompt": "",
                }
            )

        # ---- STEP 2: image generation (n=3) ----
        first_prompt = (ads_plan[0].get("image_prompt") or "").strip()
        if not first_prompt:
            first_prompt = (
                f"High-quality photographic advertising image for {product}. "
                "Two real objects combined or placed side by side, no logos, "
                "no text inside the image, realistic lighting."
            )

        img_resp = openai.Image.create(
            model=IMAGE_MODEL,
            prompt=first_prompt,
            size=size,
            n=3,
            response_format="b64_json",
        )

        images_data = img_resp["data"]
        if len(images_data) < 3:
            images_data = (images_data * 3)[:3]

        ads_out = []
        for idx in range(3):
            ad_plan = ads_plan[idx]
            img_item = images_data[idx]

            headline = (ad_plan.get("headline") or "").strip() or product
            copy_text = (ad_plan.get("copy") or "").strip() or description

            b64_data = img_item.get("b64_json")

            ad_payload = {
                "headline": headline,
                "copy": copy_text,
            }
            if b64_data:
                ad_payload["image_base64"] = b64_data

            ads_out.append(ad_payload)

        if len(ads_out) == 0:
            return jsonify({"error": "No ads generated from image model"}), 500

        return jsonify({"ads": ads_out, "size_used": size}), 200

    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"error": "Internal generation error", "details": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
