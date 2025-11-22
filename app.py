import os
import io
import zipfile
import base64
from datetime import datetime
from typing import List, Dict

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont

from openai import OpenAI

# OpenAI client (new SDK)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


@app.get("/health")
def health():
    """Health check with engine version and OpenAI status."""
    return jsonify({
        "status": "ok",
        "time": datetime.utcnow().isoformat() + "Z",
        "engine": "ENGINE_V0.7",
        "openai_configured": bool(os.environ.get("OPENAI_API_KEY")),
        "image_model": "gpt-image-latest",
    })


def parse_size(raw: str):
    """Parse size strings like '1080x1350', '1080×1350', '1080 x 1350'."""
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


def pick_personas(product: str, description: str) -> List[Dict]:
    """Simple persona generator for three emotional states."""
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
    """
    Build a 50-word English marketing paragraph that:
    - talks to the persona by psychology, not demographics
    - never describes shapes or the hybrid object
    """
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
    """Pillow 10 compatible text size helper (using textbbox)."""
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return width, height


def choose_object_pair(persona: Dict):
    """
    Minimal stand-in for the 100-object library.
    We select pairs that are:
    - real-world photographic objects
    - compatible by basic silhouette metaphor
    """
    goal = (persona.get("goal") or "").lower()

    if "clarity" in goal:
        return "an open history textbook on a desk", "a clean glass window with blue sky outside"

    if "confidence" in goal:
        return "an exam paper graded A plus on a table", "a solid brushed metal shield standing upright"

    if "calm" in goal or "exam" in goal:
        return "a glass hourglass standing on a wooden table", "a calm ocean wave at golden hour"

    return "a neat stack of organised study notes", "a quiet warm desk lamp"


def gradient_fallback(width: int, height: int, top_color=(245, 245, 245), bottom_color=(230, 235, 240)) -> Image.Image:
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    for y in range(height):
        ratio = y / max(1, height - 1)
        r = int(top_color[0] * (1 - ratio) + bottom_color[0] * ratio)
        g = int(top_color[1] * (1 - ratio) + bottom_color[1] * ratio)
        b = int(top_color[2] * (1 - ratio) + bottom_color[2] * ratio)
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    return img


def generate_photo_with_openai(persona: Dict, width: int, height: int) -> Image.Image:
    """
    Use OpenAI Images API (new SDK) to generate a balanced commercial photograph
    of a hybrid object according to ENGINE V0.7 rules.

    If OPENAI_API_KEY is missing, or any error occurs, fall back to a soft gradient placeholder.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        app.logger.warning("OPENAI_API_KEY not set – using gradient fallback.")
        return gradient_fallback(width, height, top_color=(240, 244, 255), bottom_color=(238, 246, 250))

    obj_a, obj_b = choose_object_pair(persona)

    prompt = (
        "Balanced commercial photograph with a single clear hybrid object created by shape-based substitution between "
        f"{obj_a} and {obj_b}. "
        "Use natural or studio background that does not compete with the object. "
        "The hybrid is dominant in the frame (~70%), centered, with clean edges and realistic materials. "
        "Soft diffused lighting, gentle shadows, balanced color temperature. "
        "If strong shape similarity allows, replace a matching region of one object with the other, keeping both natural backgrounds subtly visible. "
        "If similarity is moderate, show the two objects side-by-side in matching perspective, still as one calm composition. "
        "No people, no animals, no logos, no brands, no text, no CGI, no abstract geometry. "
        "Mood should be clean but atmospheric, between minimalist tech and elegant advertising."
    )

    try:
        response = client.images.generate(
            model="gpt-image-latest",
            prompt=prompt,
            size="1024x1024"
        )
        image_base64 = response.data[0].b64_json
        img_bytes = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize((width, height), Image.LANCZOS)
        return img
    except Exception as e:
        # Log the exact OpenAI error so we can debug from Render logs
        app.logger.error(f"OpenAI image generation error: {e}")
        return gradient_fallback(width, height, top_color=(245, 245, 245), bottom_color=(230, 235, 240))


def overlay_copy(img: Image.Image, copy_text: str) -> Image.Image:
    """Add COPY (headline) at the bottom on a dark translucent strip."""
    img = img.convert("RGBA")
    draw = ImageDraw.Draw(img)
    width, height = img.size

    try:
        font = ImageFont.truetype("arial.ttf", size=int(min(width, height) * 0.08))
    except Exception:
        font = ImageFont.load_default()

    max_text_width = int(width * 0.9)
    words = copy_text.split()
    wrapped = []
    current = ""
    for word in words:
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

    if not wrapped:
        return img.convert("RGB")

    line_heights = []
    for line in wrapped:
        _, h = _text_size(draw, line, font)
        line_heights.append(h)
    total_text_height = sum(line_heights) + (len(line_heights) - 1) * 8

    start_y = height - int(height * 0.12) - total_text_height
    safe_top = int(height * 0.6)
    if start_y < safe_top:
        start_y = safe_top

    padding_x = int(width * 0.06)
    padding_y = 6

    max_line_width = 0
    for line in wrapped:
        w, _ = _text_size(draw, line, font)
        max_line_width = max(max_line_width, w)

    box_width = max_line_width + 2 * padding_x
    box_height = total_text_height + 2 * padding_y
    box_x0 = (width - box_width) // 2
    box_y0 = start_y - padding_y
    box_x1 = box_x0 + box_width
    box_y1 = box_y0 + box_height

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle([box_x0, box_y0, box_x1, box_y1], fill=(10, 10, 25, 150))
    img = Image.alpha_composite(img, overlay)
    draw = ImageDraw.Draw(img)

    y = start_y
    for line in wrapped:
        w, h = _text_size(draw, line, font)
        x = (width - w) // 2
        draw.text((x, y), line, font=font, fill=(245, 246, 252, 255))
        y += h + 8

    return img.convert("RGB")


@app.post("/generate")
def generate():
    """
    ENGINE V0.7 main endpoint.

    Input JSON:
        {
          "product": "...",
          "description": "...",
          "size": "1080x1350"
        }
    """
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

    personas = pick_personas(product, description)

    mem_file = io.BytesIO()
    with zipfile.ZipFile(mem_file, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for idx, persona in enumerate(personas, start=1):
            copy_text = persona["copy"]
            body_text = build_marketing_text(persona)

            base_img = generate_photo_with_openai(persona, width, height)
            final_img = overlay_copy(base_img, copy_text)

            img_bytes = io.BytesIO()
            final_img.save(img_bytes, format="JPEG", quality=90)
            img_bytes.seek(0)

            zf.writestr(f"ad_{idx}.jpg", img_bytes.read())
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
