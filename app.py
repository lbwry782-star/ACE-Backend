import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)

FRONTEND_URL = os.getenv("FRONTEND_URL", "*")
if FRONTEND_URL == "*":
    CORS(app)
else:
    CORS(app, resources={r"/*": {"origins": [FRONTEND_URL]}})

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini")

ALLOWED_SIZES = {"1024x1024", "1024x1536", "1536x1024"}


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/generate_ads", methods=["POST"])
def generate_ads():
    import json

    try:
        data = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"success": False, "error": "Invalid JSON"}), 400

    product_name = (data.get("product_name") or "").strip()
    product_description = (data.get("product_description") or "").strip()
    size = (data.get("size") or "1024x1024").strip()

    if not product_name:
        return jsonify({"success": False, "error": "Missing product_name"}), 400

    if size not in ALLOWED_SIZES:
        size = "1024x1024"

    prompt = f"""You are the text engine for the ACE Advertising Engine.

You receive a product name and an optional description.
Your task: generate EXACTLY 3 advertising variations.

For EACH variation you MUST output:
- "headline": a short English headline, 3–7 words, focused on the audience mindset and benefit. Do NOT describe any hybrid object literally.
- "copy": exactly 50 words of English marketing copy. It should focus on benefit, promise, or solution. No bullet lists.

Product name: "{product_name}"
Product description: "{product_description}"

Respond ONLY as JSON in the following format (no extra text):

[
  {{"headline": "...", "copy": "..."}},
  {{"headline": "...", "copy": "..."}},
  {{"headline": "...", "copy": "..."}}
]
"""  # noqa: E501

    try:
        chat_response = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise advertising copy generator that always follows instructions and JSON format strictly.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.7,
        )
        raw_text = chat_response.choices[0].message.content
    except Exception as e:
        return jsonify({"success": False, "error": f"Text generation failed: {e}"}), 500

    try:
        variations = json.loads(raw_text)
        if not isinstance(variations, list) or len(variations) != 3:
            raise ValueError("Model did not return 3 variations.")
    except Exception:
        variations = [
            {"headline": product_name[:60] or "ACE Ad Variation 1", "copy": raw_text or ""},
            {"headline": product_name[:60] or "ACE Ad Variation 2", "copy": raw_text or ""},
            {"headline": product_name[:60] or "ACE Ad Variation 3", "copy": raw_text or ""},
        ]

    # --- Generate ONE image only, reuse for all variations ---
    primary_headline = (variations[0].get("headline") or product_name or "ACE Ad").strip()

    image_prompt = f"""Photographic advertising visual for the product "{product_name}".
Black minimal background. One central hybrid object inspired by the product benefit and the inferred target audience.
Single clear composition, no clutter. No logos, no celebrities.
Embed the headline text in the image: "{primary_headline}".
"""  # noqa: E501

    image_url = None
    image_error = None

    try:
        img_resp = client.images.generate(
            model=IMAGE_MODEL,
            prompt=image_prompt,
            size=size,
            n=1,
        )
        image_url = img_resp.data[0].url
    except Exception as e:
        image_error = str(e)
        print("IMAGE_GENERATION_ERROR:", repr(e), flush=True)

    results = []

    for idx, item in enumerate(variations):
        headline = (item.get("headline") or "").strip()
        copy_text = (item.get("copy") or "").strip()

        if image_url is None and image_error:
            copy_text = (copy_text + f" [Image generation failed: {image_error}]").strip()

        results.append(
            {
                "headline": headline,
                "copy": copy_text,
                "image_url": image_url,
                "size": size,
            }
        )

    return jsonify({"success": True, "variations": results}), 200


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
