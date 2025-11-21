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
# Open CORS so GitHub Pages frontend can call this backend
CORS(app, resources={r"/*": {"origins": "*"}})


@app.get("/health")
def health() -> "flask.Response":
    """Simple health check endpoint."""
    return jsonify({
        "status": "ok",
        "time": datetime.utcnow().isoformat() + "Z",
        "engine": "ENGINE_V0.5",
    })


def parse_size(raw: str):
    """
    Parse size strings like '1080x1350', '1080×1350', '1080 x 1350'.
    Defaults to (1080, 1350) if parsing fails.
    """
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
    """
    Very simple, deterministic persona state generator.
    In a future version this could be replaced with a model-based persona extractor.
    For now we always return three states tailored by wording only.
    """
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

    # Personalize a bit using the product name
    for p in personas:
        p["product"] = base_product
        p["description"] = base_desc

    return personas


def build_marketing_text(persona: Dict) -> str:
    """
    Build a 50-word English marketing paragraph that talks to the persona.
    Does NOT describe visual shapes or the hybrid object.
    Ensures exactly 50 words by trimming or padding with soft fillers.
    """
    product = persona.get("product", "this service")
    goal = persona.get("goal", "clarity")
    opening = persona.get("opening", "Sometimes studying feels heavier than it should be.")
    name = persona.get("name", "student")

    # Raw text, slightly longer than 50 words
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
        # pad with soft neutral words that do not change meaning
        fillers = ["calmly", "gently", "clearly", "slowly", "steadily"]
        i = 0
        while len(words) < target:
            words.append(fillers[i % len(fillers)])
            i += 1

    return " ".join(words)


def make_placeholder_image(width: int, height: int, copy_text: str) -> bytes:
    """
    Create a very minimal photographic-style placeholder:
    - soft gradient-style background
    - no geometric primitives meant as objects
    - only the COPY headline in the frame

    This is a TEMPORARY visual implementation until real hybrid-object photos are plugged in.
    """
    # Create base image
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)

    # Simple vertical color blend using two soft random colors
    # (not literal shapes, just background atmosphere)
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

    # Add the COPY text
    try:
        font = ImageFont.truetype("arial.ttf", size=int(height * 0.045))
    except Exception:
        font = ImageFont.load_default()

    max_text_width = int(width * 0.8)
    wrapped = []
    current = ""
    for word in copy_text.split():
        test = (current + " " + word).strip()
        w, _ = draw.textsize(test, font=font)
        if w <= max_text_width:
            current = test
        else:
            if current:
                wrapped.append(current)
            current = word
    if current:
        wrapped.append(current)

    total_text_height = len(wrapped) * (font.size + 4)
    start_y = int(height * 0.1)

    for line in wrapped:
        w, h = draw.textsize(line, font=font)
        x = (width - w) // 2
        draw.text((x, start_y), line, font=font, fill=(20, 20, 30))
        start_y += h + 4

    # Save to bytes
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return buf.getvalue()


@app.post("/generate")
def generate():
    """
    Main generation endpoint.

    Input JSON (any of these keys):
        {
          "product": "...",
          "description": "...",
          "size": "1080x1350"
        }

    Output:
        ZIP file with:
          ad_1.jpg, ad_1.txt, ad_2.jpg, ad_2.txt, ad_3.jpg, ad_3.txt

    Current implementation:
        - Uses ENGINE V0.5 logic for personas + goals + copy + 50-word texts.
        - Uses minimal photographic-style placeholders for visuals
          (background + COPY only, no geometric objects).
    """
    data = request.get_json(silent=True) or {}

    # Try multiple possible key names to avoid breaking existing frontends
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

            # Create JPG
            img_bytes = make_placeholder_image(width, height, copy_text)
            zf.writestr(f"ad_{idx}.jpg", img_bytes)

            # Create TXT (50-word marketing text)
            zf.writestr(f"ad_{idx}.txt", body_text)

    mem_file.seek(0)
    response = send_file(
        mem_file,
        mimetype="application/zip",
        as_attachment=True,
        download_name="ace_ads_package.zip",
    )
    return response


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
