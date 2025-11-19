import os
import io
import base64
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont
import zipfile

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": os.getenv("FRONTEND_URL", "*")}})

def parse_size(size_str: str):
    try:
        parts = size_str.lower().replace(" ", "").split("x")
        if len(parts) != 2:
            return 1080, 1080
        return int(parts[0]), int(parts[1])
    except Exception:
        return 1080, 1080

def build_headline(product_name: str, variation_index: int) -> str:
    base = (product_name or "Your product").strip()
    templates = [
        f"{base} in focus",
        f"{base} that stands out",
        f"{base} for bold minds",
    ]
    headline = templates[(variation_index - 1) % len(templates)]
    # Ensure 3–7 words
    words = headline.split()
    if len(words) < 3:
        words.append("now")
    if len(words) > 7:
        words = words[:7]
    return " ".join(words)

def build_marketing_copy(product_name: str, product_description: str, variation_index: int) -> str:
    name = (product_name or "your product").strip()
    desc = (product_description or "a powerful, modern solution for everyday challenges").strip()
    base_sentence = (
        f"{name} helps your audience solve real problems with clarity and confidence. "
        f"This variation highlights everyday usability, trust, and emotional impact, "
        f"showing how {name} naturally fits into their routine and turns ordinary moments "
        f"into opportunities to feel more connected, focused, and inspired each day."
    )
    words = base_sentence.split()
    if len(words) > 50:
        words = words[:50]
    elif len(words) < 50:
        last = words[-1]
        while len(words) < 50:
            words.append(last)
    return " ".join(words)

def draw_hybrid_placeholder(width: int, height: int, headline: str) -> Image.Image:
    img = Image.new("RGB", (width, height), (10, 12, 18))
    draw = ImageDraw.Draw(img)

    r = int(min(width, height) * 0.25)
    cx = width // 2
    cy = height // 2
    offset = int(r * 0.6)

    draw.ellipse((cx - r - offset, cy - r, cx + r - offset, cy + r), fill=(34, 139, 230))
    draw.ellipse((cx - r + offset, cy - r, cx + r + offset, cy + r), fill=(208, 148, 42))

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", int(height * 0.05))
    except Exception:
        font = ImageFont.load_default()

    text = headline[:60]
    tw, th = draw.textsize(text, font=font)
    tx = (width - tw) // 2
    ty = int(height * 0.08)
    draw.text((tx, ty), text, fill=(255, 255, 255), font=font)

    return img

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(silent=True) or {}
    product_name = data.get("product_name", "").strip()
    product_description = data.get("product_description", "").strip()
    size_str = data.get("size", "1080x1080")

    width, height = parse_size(size_str)

    variations = []
    for i in range(1, 4):
        headline = build_headline(product_name, i)
        copy_text = build_marketing_copy(product_name, product_description, i)
        img = draw_hybrid_placeholder(width, height, headline)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG", quality=90)
        img_bytes.seek(0)
        b64 = base64.b64encode(img_bytes.getvalue()).decode("ascii")
        variations.append({
            "index": i,
            "headline": headline,
            "copy": copy_text,
            "image_data": "data:image/jpeg;base64," + b64
        })

    return jsonify({
        "product_name": product_name,
        "size": size_str,
        "variations": variations
    })

@app.route("/download_zip", methods=["POST"])
def download_zip():
    data = request.get_json(silent=True) or {}
    product_name = data.get("product_name", "").strip()
    product_description = data.get("product_description", "").strip()
    size_str = data.get("size", "1080x1080")
    index = int(data.get("index", 1))

    width, height = parse_size(size_str)
    headline = build_headline(product_name, index)
    copy_text = build_marketing_copy(product_name, product_description, index)
    img = draw_hybrid_placeholder(width, height, headline)

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG", quality=90)
        img_bytes.seek(0)
        img_name = f"ad_{index}.jpg"
        txt_name = f"ad_{index}_copy.txt"
        zf.writestr(img_name, img_bytes.getvalue())
        zf.writestr(txt_name, copy_text)

    zip_buffer.seek(0)
    return send_file(
        zip_buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"ace_ad_{index}.zip",
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
