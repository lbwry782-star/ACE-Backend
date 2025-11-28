import os
import io
import base64
import zipfile
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)

frontend_url = os.getenv("FRONTEND_URL", "*")
if frontend_url == "*":
  CORS(app)
else:
  CORS(app, resources={r"/*": {"origins": [frontend_url]}})

client = OpenAI()

TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini")
IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


def build_text_prompt(product: str, description: str) -> str:
    return f"""You are ACE, an automated advertising copywriter.

Product: {product}
Extra context: {description or "N/A"}

Create EXACTLY three different ad concepts.
For each concept, follow these rules strictly:

1. Headline:
   - English only
   - 3–7 words
   - Do NOT include quotes.

2. Marketing copy:
   - Exactly 50 words in English.
   - Persuasive but clear and concrete.
   - No hashtags, no emojis.

3. Visual logic (do NOT describe in the output, only think with it):
   - Use ACE Shape-First Hybrid Engine (v3):
     * Choose two real objects A and B with strong 2D silhouette similarity.
     * Use the classic natural background of A.
     * Remove A as a visible object.
     * Place B fully, without cropping, as the main object in A's world.
     * Keep at least 10% safe margins from frame edges.
   - Do NOT mention A or B explicitly in the copy.
   - Do NOT mention the word "hybrid" or any technical description.

Return your answer as pure JSON in this structure:

{{
  "ads": [
    {{"headline": "...", "copy": "..."}},
    {{"headline": "...", "copy": "..."}},
    {{"headline": "...", "copy": "..."}}
  ]
}}"""


def ensure_ads_structure(obj):
    ads = obj.get("ads", [])
    cleaned = []
    for ad in ads:
        h = str(ad.get("headline", "")).strip()
        c = str(ad.get("copy", "")).strip()
        if h and c:
            cleaned.append({"headline": h, "copy": c})
    return cleaned[:3]


def generate_image_b64(prompt: str, size: str) -> str:
    img_resp = client.images.generate(
        model=IMAGE_MODEL,
        prompt=prompt,
        size=size,
        n=1
    )
    b64 = img_resp.data[0].b64_json
    return b64


def create_zip_base64(image_bytes: bytes, headline: str, copy: str, index: int) -> str:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"ad{index}_image.jpg", image_bytes)
        text_content = f"{headline}\n\n{copy}"
        zf.writestr(f"ad{index}_copy.txt", text_content)
    mem.seek(0)
    return base64.b64encode(mem.read()).decode("utf-8")


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(silent=True) or {}
    product = (data.get("product") or "").strip()
    description = (data.get("description") or "").strip()
    size = (data.get("size") or "1024x1024").strip()

    if not product:
        return jsonify({"error": "Missing 'product' in request body"}), 400

    # basic English + word-count validation
    words = product.split()
    if len(words) == 0 or len(words) > 15:
        return jsonify({"error": "Product must be 1–15 words"}), 400
    try:
        product.encode("ascii")
    except UnicodeEncodeError:
        return jsonify({"error": "Product must be English (ASCII only)."}), 400

    prompt = build_text_prompt(product, description)

    try:
        completion = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": "You are a precise JSON-only ad copy generator."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.8,
        )
        content = completion.choices[0].message.content
    except Exception as e:
        return jsonify({"error": f"Text generation failed: {str(e)}"}), 500

    import json
    try:
        obj = json.loads(content)
    except Exception:
        # Try to repair simple JSON formatting issues
        try:
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1:
                obj = json.loads(content[start:end+1])
            else:
                raise ValueError("No JSON object found in model output.")
        except Exception as e:
            return jsonify({"error": f"Failed to parse JSON from model: {str(e)}"}), 500

    ads_struct = ensure_ads_structure(obj)
    if len(ads_struct) == 0:
        return jsonify({"error": "Model did not return any valid ads."}), 500

    result_ads = []
    index = 1

    for ad in ads_struct:
        headline = ad["headline"]
        copy = ad["copy"]

        visual_prompt = f"""Ultra-realistic photographic advertisement.

Product: {product}
Headline: {headline}

Apply ACE Shape-First Hybrid Engine (v3) logic internally:
- Use classic natural background of Object A (not shown in text).
- Remove A as visible object.
- Insert Object B fully and clearly within the frame, no cropping.
- Keep at least 10% safe margins to all frame edges.
- Focus on one clear hybrid object + clean background.

Do NOT include any text in the image.
Do NOT show logos, brands, or celebrities.
"""

        try:
            img_b64 = generate_image_b64(visual_prompt, size)
        except Exception as e:
            return jsonify({"error": f"Image generation failed: {str(e)}"}), 500

        try:
            image_bytes = base64.b64decode(img_b64)
        except Exception:
            image_bytes = b""

        zip_b64 = create_zip_base64(image_bytes, headline, copy, index)

        result_ads.append(
            {
                "headline": headline,
                "copy": copy,
                "image_base64": img_b64,
                "zip_base64": zip_b64,
                "zip_filename": f"ad{index}.zip",
            }
        )
        index += 1

    return jsonify({"ads": result_ads}), 200


if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
