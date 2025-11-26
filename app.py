import os
import json
import base64
import uuid
import datetime
import zipfile

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests

# Basic Flask app
app = Flask(__name__)

# Output directories for images and ZIPs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
ZIP_DIR = os.path.join(BASE_DIR, "zips")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ZIP_DIR, exist_ok=True)

FRONTEND_URL = os.environ.get("FRONTEND_URL", "*")

if FRONTEND_URL == "*" or not FRONTEND_URL:
    CORS(app)
else:
    CORS(app, resources={r"/*": {"origins": [FRONTEND_URL]}})


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_IMAGE_MODEL = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1")
OPENAI_TEXT_MODEL = os.environ.get("OPENAI_TEXT_MODEL", "gpt-4.1-mini")

ALLOWED_SIZES = ["1024x1024", "1536x1024", "1024x1536"]


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


def build_engine_system_prompt():
    """System prompt encoding ACE + 'Mishrad Pirsum' engine rules."""
    return (
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
        "It expresses the audience mindset or reaction, and never literally describes the hybrid object. "
        "It should feel like a strong ad headline, not a caption.\n"
        "4. MARKETING COPY RULES: The marketing copy must be exactly 50 words in English. "
        "It is written as if it is a short social media post for this one ad variation. "
        "Do not include a separate headline inside the copy. "
        "Focus on benefits, promise, or solution, and keep a clear, human, conversational tone.\n"
        "5. COMPOSITION: Assume the final image is a single photographic composition respecting margins and safe areas. "
        "No clutter. The hybrid object is clearly visible and large.\n"
        "You must NOT include any illegal content, adult content, self-harm, hate, or anything that violates OpenAI policies.\n"
        "Return output strictly as JSON."
    )


def build_chat_payload(product, description, size):
    system_prompt = build_engine_system_prompt()

    user_content = (
        f"Product name: {product}\n"
        f"Product description: {description or 'N/A'}\n"
        f"Target image size: {size}\n\n"
        "Generate exactly 3 distinct ad variations. For each variation, you must output:\n"
        "1) headline: (3–7 words, English, not literally describing the hybrid object)\n"
        "2) marketing_copy: (exactly 50 words in English)\n"
        "3) image_prompt: a detailed English prompt describing a single photographic hybrid object "
        "built from exactly two real objects with overlapping shapes, plus composition, mood, lighting, and camera angle, "
        "following all engine rules. Do NOT embed the 50-word copy or any headline into the image.\n\n"
        "Return a single JSON object with this structure:\n"
        "{\n"
        "  "variations": [\n"
        "    {\n"
        "      "headline": "...",\n"
        "      "marketing_copy": "...",\n"
        "      "image_prompt": "..."\n"
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
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.9,
    }
    return payload


def call_openai_chat(payload):
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable.")

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
        raise RuntimeError(f"OpenAI chat API error: {resp.status_code} {resp.text}")
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    return content


def call_openai_image(prompt, size):
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
        "n": 1,
        "response_format": "b64_json",
    }
    resp = requests.post(
        "https://api.openai.com/v1/images/generations",
        headers=headers,
        json=body,
        timeout=180,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI image API error: {resp.status_code} {resp.text}")
    data = resp.json()
    if not data.get("data"):
        raise RuntimeError("No image data returned from OpenAI.")
    b64 = data["data"][0].get("b64_json")
    if not b64:
        raise RuntimeError("Image field 'b64_json' missing in OpenAI response.")
    return base64.b64decode(b64)


def save_variation_files(image_bytes, headline, marketing_copy, index):
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    uid = uuid.uuid4().hex[:8]
    base_id = f"{timestamp}_{uid}_{index}"

    image_filename = f"{base_id}.jpg"
    text_filename = f"{base_id}.txt"
    zip_filename = f"{base_id}.zip"

    image_path = os.path.join(OUTPUT_DIR, image_filename)
    text_path = os.path.join(OUTPUT_DIR, text_filename)
    zip_path = os.path.join(ZIP_DIR, zip_filename)

    with open(image_path, "wb") as img_f:
        img_f.write(image_bytes)

    with open(text_path, "w", encoding="utf-8") as txt_f:
        txt_f.write(headline.strip() + "\n\n" + marketing_copy.strip())

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(image_path, arcname="image.jpg")
        zf.write(text_path, arcname="copy.txt")

    return {
        "id": base_id,
        "headline": headline.strip(),
        "marketing_copy": marketing_copy.strip(),
        "image_url": f"/image/{image_filename}",
        "zip_url": f"/download/{zip_filename}",
    }


@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"success": False, "error": "Request body must be JSON."}), 400

    product = (data.get("product") or "").strip()
    description = (data.get("description") or "").strip()
    size = (data.get("size") or "1024x1024").strip()

    if not product:
        return (
            jsonify({"success": False, "error": 'Missing "product" in request body.'}),
            400,
        )

    if size not in ALLOWED_SIZES:
        size = "1024x1024"

    try:
        chat_payload = build_chat_payload(product, description, size)
        content = call_openai_chat(chat_payload)
        parsed = json.loads(content)
        variations = parsed.get("variations") or []
        if len(variations) != 3:
            raise RuntimeError("Expected exactly 3 variations in JSON output.")

        output_variations = []
        for idx, v in enumerate(variations, start=1):
            headline = (v.get("headline") or "").strip()
            marketing_copy = (v.get("marketing_copy") or "").strip()
            image_prompt = (v.get("image_prompt") or "").strip()

            if not headline or not marketing_copy or not image_prompt:
                raise RuntimeError("Variation is missing headline, marketing_copy, or image_prompt.")

            image_bytes = call_openai_image(image_prompt, size)
            variation_record = save_variation_files(image_bytes, headline, marketing_copy, idx)
            output_variations.append(variation_record)

        return jsonify({"success": True, "variations": output_variations})

    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/image/<path:filename>", methods=["GET"])
def serve_image(filename):
    return send_from_directory(OUTPUT_DIR, filename)


@app.route("/download/<path:filename>", methods=["GET"])
def download_zip(filename):
    return send_from_directory(ZIP_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
