import os
import json
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai

app = Flask(__name__)

FRONTEND_URL = os.getenv("FRONTEND_URL", "https://ace-advertising.agency")
CORS(app, resources={r"/*": {"origins": [FRONTEND_URL]}})

openai.api_key = os.getenv("OPENAI_API_KEY")
TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini")
IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")

ALLOWED_SIZES = {
    "1024x1024": "1024x1024",
    "1024x1536": "1024x1536",
    "1536x1024": "1536x1024",
}


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


def is_english_only(text: str) -> bool:
    try:
        text.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def build_text_prompt(product: str, description: str, variation_index: int) -> str:
    return f"""You are ACE, an advertising copy expert.

Product: {product}
Description: {description or "(no description provided)"}
Variation: {variation_index}

Write an advertising concept for one single ad, and then output ONLY a JSON object
with exactly two fields:

"headline": a short headline in English, 3–7 words, sharp and smart, directly related to the visual scene.
"copy": a 50-word English marketing text (exactly 50 words, not 49 or 51).

Rules:
- Do not mention AI, engines, tokens, or technical details.
- Do not mention the number of variations.
- Focus on benefits and emotional impact.
- Output ONLY valid JSON, no commentary.
"""


def build_image_prompt(product: str, description: str, variation_index: int) -> str:
    return f"""Create a photographic advertising image for this product:

Product: {product}
Description: {description or "(no description provided)"}

Rules (internal ACE engine style):
- Use exactly two real-world physical objects (call them A and B) that are conceptually relevant to the product and its goal.
- Choose A and B so that they have strong visual and shape similarity, or at least a clear visual echo between them.
- Either fuse A and B into a single visually continuous form, or place them very close together or side by side so the similarity is obvious.
- Use ONLY A and B as main objects. Do not add any_extra props, icons, logos, text, UI elements, people, or decorations.
- Use a natural, classic background that belongs either to A or to B (for example: the natural environment where that object normally appears).
- Keep lighting realistic and consistent with the background.
- Respect safe margins around the frame (no cropping through the objects).

Composition:
- Center the composition cleanly in the frame.
- Leave some free space for a headline area, but do NOT render any text.
- The overall style is premium, sharp, and cinematic.

Do NOT include any written words in the image. The image must be suitable as an online ad visual without embedded typography.
"""


def generate_ad(product: str, description: str, size_key: str, index: int):
    text_prompt = build_text_prompt(product, description, index)

    text_resp = openai.ChatCompletion.create(
        model=TEXT_MODEL,
        messages=[
            {"role": "system", "content": "You write short, sharp ad copy and always respond with pure JSON."},
            {"role": "user", "content": text_prompt},
        ],
        temperature=0.8,
    )

    raw_text = text_resp["choices"][0]["message"]["content"].strip()
    try:
        data = json.loads(raw_text)
    except Exception:
        data = {"headline": "", "copy": raw_text}

    headline = data.get("headline", "").strip()
    copy = data.get("copy", "").strip()

    image_prompt = build_image_prompt(product, description, index)
    size_value = ALLOWED_SIZES[size_key]

    image_resp = openai.Image.create(
        model=IMAGE_MODEL,
        prompt=image_prompt,
        size=size_value,
        n=1,
        response_format="b64_json",
    )

    b64 = image_resp["data"][0]["b64_json"]

    return {
        "headline": headline,
        "marketing_text": copy,
        "image_base64": b64,
    }


@app.route("/generate", methods=["POST"])
def generate():
    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    product = (payload.get("product") or "").strip()
    description = (payload.get("description") or "").strip()
    size = (payload.get("size") or "").strip()

    if not product:
        return jsonify({"error": "Missing product"}), 400

    if not is_english_only(product) or not is_english_only(description):
        return jsonify({"error": "English only"}), 400

    if len(product.split()) > 15:
        return jsonify({"error": "Product too long"}), 400

    if size not in ALLOWED_SIZES:
        return jsonify({"error": "Invalid size"}), 400

    ads = []
    try:
        for i in range(1, 4):
            ads.append(generate_ad(product, description, size, i))
    except Exception as e:
        return jsonify({"error": "Generation failed"}), 500

    return jsonify({"ads": ads})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
