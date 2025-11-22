
import os
import io
import base64
import textwrap
import uuid
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)

# Allow CORS only from the frontend origin (set in environment)
frontend_origin = os.getenv("FRONTEND_ORIGIN", "*")
CORS(app, resources={r"/*": {"origins": frontend_origin}})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


def create_placeholder_image(size_tuple, headline, idx):
    """Create a simple placeholder image with the headline text."""
    width, height = size_tuple

    # Background color depends on variation index
    base_colors = [(28, 40, 72), (20, 60, 80), (60, 35, 90)]
    bg = base_colors[idx % len(base_colors)]

    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)

    # Draw a simple frame
    margin = int(min(width, height) * 0.04)
    draw.rectangle(
        [margin, margin, width - margin, height - margin],
        outline=(250, 210, 120),
        width=max(2, int(min(width, height) * 0.01)),
    )

    # Text (headline)
    text = headline or "ACE Ad"
    wrapped = textwrap.fill(text, width=18)

    # Try to load a basic font; fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", size=int(height * 0.06))
    except Exception:
        font = ImageFont.load_default()

    # Center text
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


def image_to_data_url(img):
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def create_zip_for_variation(image_bytes, copy_text):
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

    # Allowed sizes
    allowed_sizes = {
        "1200x630": (1200, 630),
        "1080x1350": (1080, 1350),
        "1080x1080": (1080, 1080),
        "1080x1920": (1080, 1920),
    }
    if size_str not in allowed_sizes:
        return jsonify({"error": "Invalid size"}), 400

    width, height = allowed_sizes[size_str]

    # Simple placeholder marketing copy ~50 words
    base_copy = (
        f"{product_name} gives your audience a clear, memorable benefit. "
        f"This placeholder text is here so you can fully test the builder page, "
        f"including images, headlines, and downloads, before connecting the real ACE "
        f"hybrid-object engine and OpenAI text logic in production."
    )
    words = base_copy.split()
    if len(words) > 50:
        words = words[:50]
    placeholder_copy = " ".join(words)

    variations = []
    for idx in range(3):
        headline = f"{product_name} – Variation {idx + 1}"

        img = create_placeholder_image((width, height), headline, idx)
        img_buffer = io.BytesIO()
        img.save(img_buffer, format="JPEG", quality=90)
        img_bytes = img_buffer.getvalue()

        data_url = image_to_data_url(img)

        zip_filename = create_zip_for_variation(img_bytes, placeholder_copy)
        host = request.host_url.rstrip("/")
        zip_url = f"{host}/download/{zip_filename}"

        variations.append(
            {
                "image_data_url": data_url,
                "headline": headline,
                "copy": placeholder_copy,
                "zip_url": zip_url,
            }
        )

    return jsonify({"variations": variations})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
