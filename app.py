
import os
import io
import base64
import uuid
import textwrap

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont

# OpenAI client (new SDK)
try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception:
    client = None

IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini")

app = Flask(__name__)
frontend_origin = os.getenv("FRONTEND_ORIGIN", "*")
CORS(app, resources={r"/*": {"origins": frontend_origin}})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# --- Placeholder helpers ---
def create_placeholder_image(size_tuple, headline, idx):
    width, height = size_tuple
    base_colors = [(28, 40, 72), (20, 60, 80), (60, 35, 90)]
    bg = base_colors[idx % len(base_colors)]
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)

    margin = int(min(width, height) * 0.04)
    draw.rectangle(
        [margin, margin, width - margin, height - margin],
        outline=(250, 210, 120),
        width=max(2, int(min(width, height) * 0.01)),
    )

    text = headline or "ACE Ad"
    wrapped = textwrap.fill(text, width=18)

    try:
        font = ImageFont.truetype("arial.ttf", size=int(height * 0.06))
    except Exception:
        font = ImageFont.load_default()

    try:
        bbox = draw.multiline_textbbox((0, 0), wrapped, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except Exception:
        text_w, text_h = draw.multiline_textsize(wrapped, font=font)

    x = (width - text_w) / 2
    y = (height - text_h) / 2

    draw.multiline_text(
        (x, y),
        wrapped,
        fill=(255, 245, 225),
        font=font,
        align="center",
    )

    return img


def image_to_data_url_from_bytes(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes))
        out = io.BytesIO()
        img.convert("RGB").save(out, format="JPEG", quality=90)
        encoded = base64.b64encode(out.getvalue()).decode("ascii")
    except Exception:
        encoded = base64.b64encode(img_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def image_to_data_url(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def create_zip_for_variation(image_bytes, copy_text):
    tmp_dir = "/tmp/ace_ads"
    os.makedirs(tmp_dir, exist_ok=True)
    filename = f"ad_{uuid.uuid4().hex}.zip"
    zip_path = os.path.join(tmp_dir, filename)

    import zipfile
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("ad.jpg", image_bytes)
        zf.writestr("copy.txt", copy_text)

    return filename


# --- OpenAI helpers ---
def generate_marketing_copy(product_name, product_description):
    base_fallback = (
        f"{product_name} gives your audience a clear, memorable benefit. "
        f"This placeholder text is here so you can fully test the builder page, "
        f"including images, headlines, and downloads, before connecting the real ACE "
        f"hybrid-object engine and OpenAI text logic in production."
    )
    words = base_fallback.split()
    if len(words) > 50:
        words = words[:50]
    fallback_copy = " ".join(words)

    if client is None:
        return fallback_copy

    try:
        prompt = (
            "You are an advertising copywriter.\n"
            f"Product name: {product_name}\n"
            f"Product description: {product_description}\n\n"
            "Write a short marketing text in English, exactly 50 words, "
            "focused on the benefit, promise, or solution. "
            "Do not use a headline, only one compact paragraph."
        )
        resp = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": "You write concise marketing copy in English."},
                {"role": "user", "content": prompt},
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            return fallback_copy
        w = text.split()
        if len(w) > 55:
            w = w[:55]
            text = " ".join(w)
        return text
    except Exception:
        return fallback_copy


def size_is_valid_for_openai(size_str):
    return size_str in {"1024x1024", "1024x1536", "1536x1024"}


def generate_openai_image(product_name, product_description, headline, size_str):
    if client is None or not size_is_valid_for_openai(size_str):
        return None

    try:
        prompt = (
            "Photographic advertising image. Minimal, clean background with depth. "
            "Central hybrid object that visually represents the product benefit. "
            "No logos, no brands, no celebrities, no recognizable IP. "
            f"Product name: {product_name}. "
            f"Product description: {product_description}. "
            f"Embed this headline clearly in the image: '{headline}'. "
            "Don't include any other text besides this headline."
        )
        resp = client.images.generate(
            model=IMAGE_MODEL,
            prompt=prompt,
            size=size_str,
            n=1,
            response_format="b64_json",
        )
        b64_data = resp.data[0].b64_json
        img_bytes = base64.b64decode(b64_data)
        return img_bytes
    except Exception:
        return None


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

    allowed_sizes = {
        "1024x1024": (1024, 1024),
        "1024x1536": (1024, 1536),
        "1536x1024": (1536, 1024),
    }
    if size_str not in allowed_sizes:
        return jsonify({"error": "Invalid size"}), 400

    width, height = allowed_sizes[size_str]

    marketing_copy = generate_marketing_copy(product_name, product_description)

    variations = []
    for idx in range(3):
        headline = f"{product_name} – Variation {idx + 1}"

        img_bytes = generate_openai_image(product_name, product_description, headline, size_str)

        if img_bytes is None:
            img = create_placeholder_image((width, height), headline, idx)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            img_bytes = buf.getvalue()
            data_url = image_to_data_url(img)
        else:
            data_url = image_to_data_url_from_bytes(img_bytes)

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
