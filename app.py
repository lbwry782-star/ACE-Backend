
import os
import io
import base64
import textwrap as _textwrap
import uuid

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
from openai import OpenAI

app = Flask(__name__)

# Allow CORS only from the frontend origin (set in environment)
frontend_origin = os.getenv("FRONTEND_ORIGIN", "*")
CORS(app, resources={r"/*": {"origins": frontend_origin}})

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


def generate_marketing_copy(product_name: str, product_description: str) -> str:
    """Use OpenAI text model to generate ~50-word marketing copy."""
    prompt = (
        "You are an advertising copywriter.\n"
        f"Product name: {product_name}\n"
        f"Product description: {product_description}\n\n"
        "Write a short marketing text in English, exactly 50 words, "
        "focused on the benefit, promise, or solution. Do not use a headline, "
        "only one compact paragraph."
    )

    resp = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {"role": "system", "content": "You write marketing copy in concise English."},
            {"role": "user", "content": prompt},
        ],
    )

    text = resp.choices[0].message.content.strip()
    # As a safeguard, roughly trim to 50 words if needed
    words = text.split()
    if len(words) > 55:
        words = words[:55]
        text = " ".join(words)
    return text


def size_to_pixels(size_str: str):
    """Map logical ad size to pixel dimensions acceptable for gpt-image-1."""
    mapping = {
        "1200x630": "1200x630",
        "1080x1350": "1080x1350",
        "1080x1080": "1080x1080",
        "1080x1920": "1080x1920",
    }
    if size_str not in mapping:
        return None
    return mapping[size_str]


def generate_ad_image(product_name: str, product_description: str, headline: str, size_str: str) -> bytes:
    """Generate a photographic ad image using OpenAI gpt-image-1."""
    size_pixels = size_to_pixels(size_str)
    if not size_pixels:
        size_pixels = "1080x1080"

    prompt = (
        "Photographic advertising image. Minimal clean background. "
        "Central hybrid object that visually represents the product's benefit. "
        "No logos, no brands, no celebrities. "
        f"Product name: {product_name}. "
        f"Description: {product_description}. "
        f"Embed this headline text in English clearly in the image: '{headline}'. "
        "Do not include any other text besides this headline."
    )

    img_resp = client.images.generate(
        model=IMAGE_MODEL,
        prompt=prompt,
        size=size_pixels,
        n=1,
        response_format="b64_json",
    )

    b64_data = img_resp.data[0].b64_json
    img_bytes = base64.b64decode(b64_data)
    return img_bytes


def image_bytes_to_data_url(img_bytes: bytes) -> str:
    # Ensure it's a valid JPEG/PNG data URL. We'll convert to JPEG for consistency.
    try:
        img = Image.open(io.BytesIO(img_bytes))
        out = io.BytesIO()
        img.convert("RGB").save(out, format="JPEG", quality=90)
        encoded = base64.b64encode(out.getvalue()).decode("ascii")
    except Exception:
        encoded = base64.b64encode(img_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def create_zip_for_variation(image_bytes: bytes, copy_text: str) -> str:
    # Save zip into a temp folder in the container
    tmp_dir = "/tmp/ace_ads"
    os.makedirs(tmp_dir, exist_ok=True)

    filename = f"ad_{uuid.uuid4().hex}.zip"
    zip_path = os.path.join(tmp_dir, filename)

    import zipfile

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Image file
        zf.writestr("ad.jpg", image_bytes)
        # Copy text as txt file
        zf.writestr("copy.txt", copy_text)

    return filename


@app.route("/download/<path:zip_name>", methods=["GET"])
def download_zip(zip_name):
    tmp_dir = "/tmp/ace_ads"
    return send_from_directory(tmp_dir, zip_name, as_attachment=True)


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(silent=True) or {}

    product_name = (data.get("product_name") or "").strip()
    product_description = (data.get("product_description") or "").strip()
    size_str = (data.get("size") or "").strip()

    if not product_name or not product_description or not size_str:
        return jsonify({"error": "Missing product_name, product_description or size"}), 400

    # First generate a single base marketing copy (same for all 3 variations)
    try:
        marketing_copy = generate_marketing_copy(product_name, product_description)
    except Exception as e:
        # Fallback simple placeholder if OpenAI text fails
        marketing_copy = (
            f"{product_name} stands out with a clear benefit for your audience. "
            f"This placeholder copy is here so you can visually test the builder page even "
            f"if the text model failed. "
            f"Replace this engine later with the full ACE hybrid-object logic."
        )
        words = marketing_copy.split()
        if len(words) > 50:
            words = words[:50]
            marketing_copy = " ".join(words)

    variations = []
    for idx in range(3):
        headline = f"{product_name} – Variation {idx + 1}"

        try:
            img_bytes = generate_ad_image(product_name, product_description, headline, size_str)
        except Exception as e:
            # If image generation fails, return an error for now
            return jsonify({"error": f"Image generation failed: {e}"}), 500

        data_url = image_bytes_to_data_url(img_bytes)
        zip_filename = create_zip_for_variation(img_bytes, marketing_copy)
        host = request.host_url.rstrip("/")
        zip_url = f"{host}/download/{zip_filename}"

        variations.append(
            {
                "image_data_url": data_url,
                "headline": headline,
                "copy": marketing_copy,
                "zip_url": zip_url,
            }
        )

    return jsonify({"variations": variations})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
