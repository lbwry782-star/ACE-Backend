import os
import json
import uuid
import datetime
import base64
import zipfile

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests

# Basic Flask app
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
ZIP_DIR = os.path.join(BASE_DIR, "zips")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ZIP_DIR, exist_ok=True)

FRONTEND_URL = os.environ.get("FRONTEND_URL", "*")

if FRONTEND_URL and FRONTEND_URL != "*":
    CORS(app, resources={r"/*": {"origins": [FRONTEND_URL]}})
else:
    CORS(app)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_IMAGE_MODEL = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1")
OPENAI_TEXT_MODEL = os.environ.get("OPENAI_TEXT_MODEL", "gpt-4.1-mini")

ALLOWED_SIZES = ["1024x1024", "1536x1024", "1024x1536"]


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


def chat_completion(product, description, size):
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable.")

    system_prompt = (
        "You are the ACE hybrid advertising engine. "
        "You receive an English product name and optional description, and you must create advertising ideas that follow these rules:\n"
        "1. ENGINE FLOW: Infer the target audience and advertising goal from the product. "
        "Then mentally choose two real-world objects whose shapes form a shared super-shape when overlapped, like a solar eclipse. "
        "From that, you imagine a single photographic hybrid object composition.\n"
        "2. HYBRID OBJECT RULES: The visual is always a single hybrid object that is made from exactly two real, everyday objects. "
        "No logos, no brands, no famous people, no text rendered inside the image. "
        "One object partially eclipses or overlaps the other with precise alignment of size, angle, and position. "
        "The background is minimal and clean, photography-based, with the hybrid object central.\n"
        "3. HEADLINE RULES: Each headline must be in English, between 3 and 7 words. "
        "It expresses the audience mindset or reaction, and never literally describes the hybrid object.\n"
        "4. MARKETING COPY RULES: The marketing copy must be exactly 50 words in English. "
        "It is written as if it is a short social media post for this one ad variation. "
        "Do not include a separate headline inside the copy.\n"
        "5. COMPOSITION: Assume the final image is a single photographic composition respecting margins and safe areas.\n"
        "Return output strictly as JSON."
    )

    user_prompt = (
        f"Product name: {product}\n"
        f"Product description: {description or 'N/A'}\n"
        f"Target image size: {size}\n\n"
        "Generate exactly 3 distinct ad variations. For each variation, output:\n"
        "1) headline: (3–7 words, English)\n"
        "2) marketing_copy: (exactly 50 words in English)\n"
        "3) image_prompt: a detailed English prompt describing a single photographic hybrid object "
        "built from exactly two real objects with overlapping shapes, plus composition, mood, lighting, and camera angle.\n\n"
        "Return a single JSON object with this structure:\n"
        "{\n"
        "  \"variations\": [\n"
        "    {\n"
        "      \"headline\": \"...\",\n"
        "      \"marketing_copy\": \"...\",\n"
        "      \"image_prompt\": \"...\"\n"
        "    },\n"
        "    { ... },\n"
        "    { ... }\n"
        "  ]\n"
        "}\n"
        "Make sure each marketing_copy field is exactly 50 words."
    )

    payload = {
        "model": OPENAI_TEXT_MODEL,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.9,
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=120,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI chat error: {resp.status_code} {resp.text}")

    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    try:
        parsed = json.loads(content)
    except Exception as exc:
        raise RuntimeError(f"Failed to parse JSON from model: {exc}; content={content[:400]}")

    variations = parsed.get("variations") or []
    if not isinstance(variations, list) or len(variations) != 3:
        raise RuntimeError("Model did not return exactly 3 variations.")

    return variations


def generate_image(prompt, size):
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable.")

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    body = {
        "model": OPENAI_IMAGE_MODEL,
        "prompt": prompt,
        "size": size,
        # do not pass response_format to avoid unknown parameter issues
    }

    resp = requests.post(
        "https://api.openai.com/v1/images/generations",
        headers=headers,
        json=body,
        timeout=180,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI image error: {resp.status_code} {resp.text}")

    data = resp.json()
    if not data.get("data"):
        raise RuntimeError("Image API returned no data.")

    item = data["data"][0]

    # Prefer b64_json if present, otherwise fallback to URL and download
    image_bytes = None
    if isinstance(item, dict) and "b64_json" in item:
        image_bytes = base64.b64decode(item["b64_json"])
    elif isinstance(item, dict) and "url" in item:
        img_resp = requests.get(item["url"], timeout=180)
        if img_resp.status_code != 200:
            raise RuntimeError(f"Failed to fetch image from url: {img_resp.status_code}")
        image_bytes = img_resp.content
    else:
        raise RuntimeError("Image item missing b64_json or url.")

    return image_bytes


def save_image_and_zip(image_bytes, headline, marketing_copy, prefix):
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:8]
    img_filename = f"{prefix}_{ts}_{uid}.jpg"
    img_path = os.path.join(OUTPUT_DIR, img_filename)
    with open(img_path, "wb") as f:
        f.write(image_bytes)

    copy_filename = f"{prefix}_{ts}_{uid}_copy.txt"
    copy_path = os.path.join(OUTPUT_DIR, copy_filename)
    with open(copy_path, "w", encoding="utf-8") as f:
        f.write(headline.strip() + "\n\n" + marketing_copy.strip())

    zip_filename = f"{prefix}_{ts}_{uid}.zip"
    zip_path = os.path.join(ZIP_DIR, zip_filename)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(img_path, arcname=img_filename)
        z.write(copy_path, arcname="copy.txt")

    return img_filename, zip_filename


@app.route("/outputs/<path:filename>", methods=["GET"])
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=False)


@app.route("/zips/<path:filename>", methods=["GET"])
def serve_zip(filename):
    return send_from_directory(ZIP_DIR, filename, as_attachment=True)


@app.route("/generate", methods=["POST"])
def generate():
    # basic JSON body parsing
    try:
        data = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"success": False, "error": "Request body must be JSON."}), 400

    product = (data.get("product") or "").strip()
    description = (data.get("description") or "").strip()
    size = (data.get("size") or "1024x1024").strip()

    if not product:
        return jsonify({"success": False, "error": 'Missing "product" in request body.'}), 400

    if size not in ALLOWED_SIZES:
        size = "1024x1024"

    try:
        spec_variations = chat_completion(product, description, size)
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500

    response_variations = []

    for idx, spec in enumerate(spec_variations, start=1):
        headline = (spec.get("headline") or f"Variation {idx}").strip()
        marketing_copy = (spec.get("marketing_copy") or "").strip()
        image_prompt = (spec.get("image_prompt") or "").strip()

        if not image_prompt:
            image_prompt = (
                f"Hyper-realistic studio photograph of a single hybrid object made from two real items, "
                f"symbolizing {product}. Dark background, dramatic lighting."
            )

        try:
            img_bytes = generate_image(image_prompt, size)
            img_filename, zip_filename = save_image_and_zip(
                img_bytes, headline, marketing_copy, prefix=f"var{idx}"
            )
            image_url = f"/outputs/{img_filename}"
            zip_url = f"/zips/{zip_filename}"
        except Exception as exc_img:
            # If image fails, still return text-only variation.
            image_url = ""
            zip_url = ""
            marketing_copy = marketing_copy + " (Image generation failed – text only.)"

        response_variations.append(
            {
                "headline": headline,
                "marketing_copy": marketing_copy,
                "image_url": image_url,
                "zip_url": zip_url,
            }
        )

    return jsonify({"success": True, "variations": response_variations})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
