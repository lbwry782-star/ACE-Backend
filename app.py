import os
import io
import zipfile
from datetime import datetime
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

# Try to import Pillow; if not available we still return a valid ZIP (text only)
try:
    from PIL import Image, ImageDraw, ImageFont  # type: ignore
    PIL_AVAILABLE = True
except Exception:  # ImportError or anything else
    PIL_AVAILABLE = False

app = Flask(__name__)

# CORS – allow all origins so frontend on GitHub Pages can always call us
CORS(app, resources={r"/*": {"origins": "*"}})


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "time": datetime.utcnow().isoformat() + "Z",
            "pil_available": PIL_AVAILABLE,
        }
    )


def create_placeholder_image(text: str, size=(1080, 1350)):
    if not PIL_AVAILABLE:
        return None

    img = Image.new("RGB", size, (17, 24, 39))  # dark slate background
    draw = ImageDraw.Draw(img)
    w, h = img.size

    margin = int(min(w, h) * 0.08)
    draw.rectangle(
        [margin, margin, w - margin, h - margin],
        outline=(250, 204, 21),
        width=4,
    )

    msg = text[:60]
    font = ImageFont.load_default()
    tw, th = draw.textsize(msg, font=font)
    draw.text(((w - tw) / 2, (h - th) / 2), msg, fill=(250, 250, 250), font=font)

    return img


@app.post("/generate")
def generate():
    data = request.get_json(silent=True) or {}
    product = (data.get("product") or "Your product").strip()
    description = (data.get("description") or "").strip()
    size_str = (data.get("size") or "1080x1350").strip()

    try:
        width, height = map(int, size_str.lower().split("x"))
    except Exception:
        width, height = 1080, 1350

    mem_file = io.BytesIO()
    with zipfile.ZipFile(mem_file, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # If Pillow is available, include 3 JPG placeholders
        if PIL_AVAILABLE:
            for i in range(1, 4):
                img = create_placeholder_image(f"{product} – Ad {i}", size=(width, height))
                if img is not None:
                    img_bytes = io.BytesIO()
                    img.save(img_bytes, format="JPEG", quality=90)
                    img_bytes.seek(0)
                    zf.writestr(f"ad_{i}.jpg", img_bytes.read())
        else:
            # If no Pillow, include tiny text placeholders instead of images
            for i in range(1, 4):
                zf.writestr(
                    f"ad_{i}.txt",
                    f"Placeholder for Ad {i} – in the full engine this would be a JPG image.",
                )

        copy_text = (
            f"ACE demo package for product: {product}\n"
            f"Short description: {description}\n\n"
            "This demo ZIP contains three placeholder ads for the ACE demo.\n"
            "If the environment supports Pillow, you will see three JPG images.\n"
            "Otherwise you will see three small text files instead of images.\n"
            "In the full ACE engine each ad would be a photographic hybrid-object visual, "
            "with three separate 50-word marketing texts in English, "
            "according to the official engine rules and Terms & Policies.\n"
        )
        zf.writestr("copy.txt", copy_text)

    mem_file.seek(0)
    return send_file(
        mem_file,
        mimetype="application/zip",
        as_attachment=True,
        download_name="ace_ads_package.zip",
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
