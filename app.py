import os
import io
import re
import uuid
import time
import json
import base64
import zipfile
import logging
from typing import Any, Dict, Optional, Tuple

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from PIL import Image, ImageDraw, ImageFont

from openai import OpenAI
from openai import APIError, RateLimitError, APIConnectionError

# -----------------------
# Config
# -----------------------
ALLOWED_SIZES = {"1024x1024", "1024x1536", "1536x1024"}

OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
OPENAI_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini")
FRONTEND_URL = (os.getenv("FRONTEND_URL") or "").strip()
PORT = int(os.getenv("PORT", "10000"))

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ace-backend")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# In-memory jobs (Render free tier resets on restart — OK for now)
JOBS: Dict[str, Dict[str, Any]] = {}


# -----------------------
# Helpers
# -----------------------
def _error(status: int, message: str, code: str = "error"):
    return jsonify({"error": code, "message": message}), status


def _validate_payload(payload: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], str, int, Optional[str]]:
    product = (payload.get("product") or "").strip()
    description = (payload.get("description") or "").strip()
    size = (payload.get("size") or "").strip() or "1024x1024"
    ad_number = int(payload.get("ad_number") or 1)
    job_id = (payload.get("job_id") or "").strip() or None

    if not product:
        return None, None, "", 0, None
    if not description:
        return product, None, "", 0, job_id
    if size not in ALLOWED_SIZES:
        return product, description, size, 0, job_id
    if ad_number not in (1, 2, 3):
        ad_number = 1
    return product, description, size, ad_number, job_id


def _safe_headline(text: str) -> str:
    # Keep bold + short + printable
    t = (text or "").strip()
    t = re.sub(r"\s+", " ", t)
    # remove quotes
    t = t.strip('"\'')

    # hard cap
    if len(t) > 48:
        t = t[:48].rsplit(" ", 1)[0]
    # 3-7 words preferred; if too long, shorten
    words = t.split()
    if len(words) > 7:
        t = " ".join(words[:7])
    return t


def _word_count(text: str) -> int:
    return len([w for w in re.split(r"\s+", (text or "").strip()) if w])


def _enforce_50_words(text: str) -> str:
    # If model returns not exactly 50 words, we truncate or pad minimally.
    words = [w for w in re.split(r"\s+", (text or "").strip()) if w]
    if len(words) == 50:
        return " ".join(words)
    if len(words) > 50:
        return " ".join(words[:50])
    # pad with simple neutral words (keeps 50)
    filler = ["today"] * (50 - len(words))
    return " ".join(words + filler)


def _pick_font(size_px: int) -> ImageFont.FreeTypeFont:
    """
    Render environment usually has DejaVu fonts installed.
    We try common paths, else fallback to default PIL bitmap font.
    """
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size_px)
            except Exception:
                continue
    return ImageFont.load_default()


def _draw_headline_on_image(img_bytes: bytes, headline: str) -> bytes:
    """
    Headline is part of the image, placed on the background of object A.
    No special banner/box. Just text with stroke for readability.
    """
    im = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    W, H = im.size

    # Strong, prominent headline:
    # Scale based on width. 1536x1024 => larger than 1024.
    font_size = int(W * 0.06)  # ~92px for 1536 width, ~61px for 1024
    font_size = max(46, min(font_size, 110))
    font = _pick_font(font_size)

    draw = ImageDraw.Draw(im)

    text = _safe_headline(headline)
    if not text:
        return img_bytes

    # Compute text box
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=max(2, font_size // 18))
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    pad = int(W * 0.04)

    # Place top-left-ish (commonly "background area") without a banner
    x = pad
    y = pad

    # If too wide, reduce font until fits
    stroke = max(2, font_size // 18)
    while tw > (W - 2 * pad) and font_size > 34:
        font_size -= 4
        font = _pick_font(font_size)
        stroke = max(2, font_size // 18)
        bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

    # Draw
    draw.text(
        (x, y),
        text,
        font=font,
        fill=(255, 255, 255, 255),
        stroke_width=stroke,
        stroke_fill=(0, 0, 0, 200),
    )

    out = io.BytesIO()
    im.convert("RGB").save(out, format="JPEG", quality=92)
    return out.getvalue()


def _to_data_url(jpg_bytes: bytes) -> str:
    b64 = base64.b64encode(jpg_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _make_zip(image_bytes: bytes, text: str, ad_number: int) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"ad{ad_number}.jpg", image_bytes)
        z.writestr(f"ad{ad_number}.txt", text)
    return buf.getvalue()


def _engine_prompt(product: str, description: str, ad_number: int) -> str:
    """
    Minimal, stable implementation consistent with the engine rules:
    - Background is the classical environment of object A.
    - Visual is either HYBRID (preferred) or SIDE-BY-SIDE if needed.
    - No text in the generated image (we add headline later).
    - Leave clean negative space on background for headline placement.
    """
    return f"""
Create ONE advertising photo (not illustration). NO text, NO letters, NO watermark.

PRODUCT: {product}
CONTEXT: {description}
AD NUMBER: {ad_number}/3

COMPOSITION RULES:
- Choose two physical objects A and B that match the ad goal (you decide).
- Prefer a HYBRID object A+B if their silhouettes can overlap naturally; otherwise place A and B side-by-side.
- The background MUST be the classical background of object A (H06).
- Make A+B the main subject. Camera angle must be realistic and consistent.
- IMPORTANT: leave clear empty background space (negative space) in the upper-left area to allow a headline later.
- Lighting: clean studio-quality or natural cinematic, high contrast, sharp focus, premium ad look.
""".strip()


def _generate_headline_and_text(product: str, description: str, ad_number: int) -> Tuple[str, str]:
    sys = "You are an advertising copywriter. Output JSON only."
    user = {
        "product": product,
        "description": description,
        "ad_number": ad_number,
        "rules": {
            "headline": "3-7 words, bold, not a quote, no punctuation at the end.",
            "text": "Exactly 50 words in English. No emojis. No bullet points.",
        },
    }
    resp = client.chat.completions.create(
        model=OPENAI_TEXT_MODEL,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": json.dumps(user)},
        ],
        temperature=0.8,
    )
    content = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(content)
    except Exception:
        # fallback parse
        data = {"headline": "", "text": content}
    headline = _safe_headline(data.get("headline") or "")
    text = _enforce_50_words(data.get("text") or "")
    return headline, text


def _generate_image(product: str, description: str, size: str, ad_number: int) -> bytes:
    prompt = _engine_prompt(product, description, ad_number)
    img = client.images.generate(
        model=OPENAI_IMAGE_MODEL,
        prompt=prompt,
        size=size,
    )
    # openai python returns b64 json sometimes; gpt-image-1 returns base64 in data[0].b64_json
    b64 = img.data[0].b64_json
    return base64.b64decode(b64)


# -----------------------
# App
# -----------------------
app = Flask(__name__)

# CORS: allow your known frontend if provided, otherwise allow all origins (dev-friendly)
if FRONTEND_URL:
    CORS(app, resources={r"/*": {"origins": [FRONTEND_URL, "https://ace-advertising.agency", "https://*.github.io", "https://*.githubusercontent.com"]}})
else:
    CORS(app)

@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.post("/generate")
def generate():
    payload = request.get_json(silent=True) or {}
    product, description, size, ad_number, job_id = _validate_payload(payload)

    if product is None:
        return _error(400, "Missing 'product' in request body.", "bad_request")
    if description is None:
        return _error(400, "Missing 'description' in request body.", "bad_request")
    if ad_number == 0:
        return _error(400, f"Invalid 'size'. Allowed: {sorted(ALLOWED_SIZES)}", "bad_request")

    if not job_id:
        job_id = uuid.uuid4().hex

    JOBS.setdefault(job_id, {"created": time.time(), "ads": {}})

    try:
        headline, text = _generate_headline_and_text(product, description, ad_number)
        raw_img = _generate_image(product, description, size, ad_number)
        final_img = _draw_headline_on_image(raw_img, headline)

        zip_bytes = _make_zip(final_img, text, ad_number)

        JOBS[job_id]["ads"][str(ad_number)] = {
            "headline": headline,
            "text": text,
            "image_bytes": final_img,
            "zip_bytes": zip_bytes,
            "size": size,
        }

        return jsonify({
            "job_id": job_id,
            "ad_number": ad_number,
            "ad": {
                "headline": headline,
                "text": text,
                "image_data_url": _to_data_url(final_img),
                "zip_url": f"/zip/{job_id}/{ad_number}",
                "size": size,
            }
        }), 200

    except RateLimitError:
        return _error(429, "Rate limit. Please try again in a minute.", "rate_limit")
    except APIConnectionError:
        return _error(502, "Network error contacting OpenAI. Please try again.", "upstream")
    except APIError as e:
        # Map OpenAI 400 to 400
        msg = str(e)
        status = 400 if "invalid" in msg.lower() or "bad request" in msg.lower() else 502
        return _error(status, msg, "openai_error")
    except Exception as e:
        log.exception("Unhandled error in /generate")
        return _error(500, f"Server error: {e}", "server_error")


@app.get("/zip/<job_id>/<int:ad_number>")
def get_zip(job_id: str, ad_number: int):
    job = JOBS.get(job_id)
    if not job:
        return _error(404, "Job not found.", "not_found")
    ad = job["ads"].get(str(ad_number))
    if not ad:
        return _error(404, "Ad not found.", "not_found")

    mem = io.BytesIO(ad["zip_bytes"])
    mem.seek(0)
    return send_file(
        mem,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"ad{ad_number}.zip",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
