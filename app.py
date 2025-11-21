import os
import io
import zipfile
import random
from datetime import datetime
from typing import List, Dict

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "time": datetime.utcnow().isoformat() + "Z",
        "engine": "ENGINE_V0.5",
    })


def parse_size(raw: str):
    if not isinstance(raw, str):
        return 1080, 1350
    cleaned = raw.lower().replace(" ", "").replace("×", "x")
    if "x" not in cleaned:
        return 1080, 1350
    w_str, h_str = cleaned.split("x", 1)
    try:
        w = int(w_str)
        h = int(h_str)
        if w <= 0 or h <= 0:
            raise ValueError
        return w, h
    except Exception:
        return 1080, 1350


def pick_persona_states(product: str, description: str) -> List[Dict]:
    base_product = product.strip() or "this service"
    base_desc = (description or "").strip()

    personas = [
        {
            "name": "Overwhelmed mind",
            "goal": "clarity",
            "copy": "Less Noise, More Focus",
            "opening": "When everything feels crowded in your head, history can turn into one long blur.",
        },
        {
            "name": "Quiet achiever",
            "goal": "confidence",
            "copy": "Quiet Steps To Success",
            "opening": "Some students know they can succeed, but feel unsure how to turn pages into real answers.",
        },
        {
            "name": "Time-pressed student",
            "goal": "calm before the exam",
            "copy": "Calm Before The Bell",
            "opening": "The exam date keeps moving closer while the material still feels unfinished and scattered.",
        },
    ]

    for p in personas:
        p["product"] = base_product
        p["description"] = base_desc

    return personas


def build_marketing_text(persona: Dict) -> str:
    product = persona.get("product", "this service")
    goal = persona.get("goal", "clarity")
    opening = persona.get("opening", "Sometimes studying feels heavier than it should be.")

    raw = (
        f"{opening} "
        f"{product} stays patient, organised and focused on what really matters for the exam. "
        f"Together you sort topics, build simple structures and practise real questions. "
        f"The result is quiet {goal}, steady progress and a feeling that you finally know where to begin and how to finish."
    )

    words = raw.split()
    target = 50

    if len(words) > target:
        words = words[:target]
    elif len(words) < target:
        fillers = ["calmly", "gently", "clearly", "slowly", "steadily"]
        i = 0
        while len(words) < target:
            words.append(fillers[i % len(fillers)])
            i += 1

    return " ".join(words)


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont):
    """Pillow 10 removed ImageDraw.textsize, emulate it via textbbox."""
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return width, height


def make_placeholder_image(width: int, height: int, copy_text: str) -> bytes:
    """
    Create a minimal poster-style placeholder:
    - soft gradient background (no hard geometric "objects")
    - COPY headline large and centered
    """
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)

    base_colors = [
        (240, 244, 255),
        (254, 247, 240),
        (242, 248, 245),
        (248, 243, 252),
        (238, 246, 250),
    ]
    top_color = random.choice(base_colors)
    bottom_color = random.choice(base_colors)

    for y in range(height):
        ratio = y / max(1, height - 1)
        r = int(top_color[0] * (1 - ratio) + bottom_color[0] * ratio)
        g = int(top_color[1] * (1 - ratio) + bottom_color[1] * ratio)
        b = int(top_color[2] * (1 - ratio) + bottom_color[2] * ratio)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # Larger font for stronger visual presence
    try:
        font = ImageFont.truetype("arial.ttf", size=int(min(width, height) * 0.09))
    except Exception:
        font = ImageFont.load_default()

    max_text_width = int(width * 0.9)
    wrapped = []
    current = ""
    for word in copy_text.split():
        test = (current + " " + word).strip()
        w, _ = _text_size(draw, test, font)
        if w <= max_text_width:
            current = test
        else:
            if current:
                wrapped.append(current)
            current = word
    if current:
        wrapped.append(current)

    # Compute total text height
    line_heights = []
    for line in wrapped:
        _, h = _text_size(draw, line, font)
        line_heights.append(h)
    total_text_height = sum(line_heights) + (len(line_heights) - 1) * 8

    # Center vertically
    start_y = (height - total_text_height) // 2

    for i, line in enumerate(wrapped):
        w, h = _text_size(draw, line, font)
        x = (width - w) // 2
        draw.text((x, start_y), line, font=font, fill=(20, 20, 30))
        start_y += h + 8

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return buf.getvalue()


@app.post("/generate")
def generate():
    data = request.get_json(silent=True) or {}

    product = (
        data.get("product")
        or data.get("productName")
        or data.get("name")
        or "Your Product"
    )
    description = (
        data.get("description")
        or data.get("productDescription")
        or data.get("shortDescription")
        or ""
    )
    size_raw = (
        data.get("size")
        or data.get("dimensions")
        or data.get("adSize")
        or "1080x1350"
    )
    width, height = parse_size(size_raw)

    personas = pick_persona_states(product, description)

    mem_file = io.BytesIO()
    with zipfile.ZipFile(mem_file, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for idx, persona in enumerate(personas, start=1):
            copy_text = persona["copy"]
            body_text = build_marketing_text(persona)

            img_bytes = make_placeholder_image(width, height, copy_text)
            zf.writestr(f"ad_{idx}.jpg", img_bytes)
            zf.writestr(f"ad_{idx}.txt", body_text)

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
