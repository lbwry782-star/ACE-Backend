import os
import io
import time
import json
import uuid
import zipfile
import logging
import re
from typing import Dict, Any, List, Optional, Tuple

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

from openai import OpenAI
from openai import APIError, RateLimitError, APIConnectionError

# -----------------------
# Config
# -----------------------
ALLOWED_SIZES = {"1024x1024", "1024x1536", "1536x1024"}

OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
OPENAI_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini")
FRONTEND_URL = os.getenv("FRONTEND_URL", "").strip()
PORT = int(os.getenv("PORT", "10000"))

# Local storage (Render/container friendly)
DATA_DIR = os.path.join(os.getcwd(), "generated")
os.makedirs(DATA_DIR, exist_ok=True)

# In-memory jobs
JOBS: Dict[str, Dict[str, Any]] = {}

# Logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ace-backend")

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------
# App
# -----------------------
app = Flask(__name__)

# CORS: allow any origin (frontend may be served from GitHub Pages and/or custom domain)
CORS(app, resources={r"/*": {"origins": "*"}})
@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200

def _error(status: int, message: str, code: str = "error"):
    return jsonify({"error": code, "message": message}), status

def _validate_payload(payload: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], str, Optional[Any]]:
    product = (payload.get("product") or "").strip()
    description = (payload.get("description") or "").strip()
    size = (payload.get("size") or "").strip() or "1024x1024"

    if not product:
        return None, None, "", _error(400, "Missing or empty 'product'.", "bad_request")
    if not description:
        return None, None, "", _error(400, "Missing or empty 'description'.", "bad_request")
    if size not in ALLOWED_SIZES:
        return None, None, "", _error(400, f"Invalid 'size'. Must be one of: {sorted(ALLOWED_SIZES)}", "bad_request")

    return product, description, size, None

def _retry_backoff(fn, *, max_retries: int = 3, delays: List[float] = [2.0, 5.0, 10.0]):
    """Retry wrapper for OpenAI rate limits (429) and transient network errors."""
    last_exc = None
    for attempt in range(max_retries):
        try:
            return fn()
        except RateLimitError as e:
            last_exc = e
            wait = delays[min(attempt, len(delays) - 1)]
            log.warning("OpenAI 429 RateLimitError. Retry %s/%s after %ss", attempt + 1, max_retries, wait)
            time.sleep(wait)
        except APIConnectionError as e:
            last_exc = e
            wait = delays[min(attempt, len(delays) - 1)]
            log.warning("OpenAI connection error. Retry %s/%s after %ss", attempt + 1, max_retries, wait)
            time.sleep(wait)
    raise last_exc

def _system_prompt_engine() -> str:
    return (
        "You are ACE ENGINE. Follow rules H00-H10 strictly with zero interpretation. "
        "No conceptual similarity; only strict visual/projection shape similarity. "
        "Output must be ONLY valid JSON, no commentary."
    )

def _text_prompt_for_one_ad(product: str, description: str, size: str, ad_index: int) -> str:
    # IMPORTANT: escape braces for JSON example so the model sees literal JSON
    return f"""
Create ONE advertising ad spec for the product below, following ACE ENGINE H00-H10 strictly.

PRODUCT:
- name: {product}
- description: {description}
- size: {size}
- ad_number: {ad_index}

STRICT RULES (must follow):
- Derive audience ONLY from product name+description (H00).
- Each ad has a DIFFERENT advertising goal (H01).
- Produce 80 physical, real, everyday objects from the goal (H02) (objects only, not abstract ideas).
- Select A (central meaning object) first, then B (emphasis object) (H03). A and B are not part of the same natural object.
- Choose a viewing angle/projection that maximizes shape similarity BEFORE finalizing the pair (H04).
- Shape similarity law (H05): if similarity is clearly high -> HYBRID; if medium -> SIDE_BY_SIDE; if not immediately obvious -> reject and choose another pair.
- No forced perspective tricks. Similarity must be immediately obvious to an average human eye.
- No objects with text/logos/letters/numbers/printed graphics unless it is physically engraved as part of the object (H03 rules).
- Background must be the classic background of object A (H06), even for side-by-side.
- Visual must be photorealistic (H09). No vector/illustration/3D/AI look.
- Ad includes ONLY VISUAL + HEADLINE (H07). Headline includes product name, 3-7 words, original, not a quote nor a variation of the product description.
- Headline IS PART OF THE IMAGE COMPOSITION: it appears above/below/next-to the visual, on the BACKGROUND of object A (negative space), never on top of A/B.
- Marketing text is exactly 50 words (H08) and is NOT on the image.

OUTPUT FORMAT (JSON ONLY):
{{
  "ad_number": {ad_index},
  "audience": {{
    "age_range": "...",
    "lifestyle": "...",
    "needs": "...",
    "knowledge_level": "...",
    "pains": "..."
  }},
  "ad_goal": "...",
  "objects_80": ["...","..."],
  "A": "...",
  "B": "...",
  "mode": "HYBRID" | "SIDE_BY_SIDE",
  "projection_angle": "...",
  "headline": "...",
  "marketing_text_50_words": "...",
  "image_prompt": "A single prompt for gpt-image-1 that produces ONLY the photorealistic VISUAL (no text). Must reflect HYBRID or SIDE_BY_SIDE and use classic background of A. Must maximize projection shape visibility."
}}

The marketing_text_50_words must be EXACTLY 50 words. Return valid JSON only.
""".strip()

def _generate_text_spec(product: str, description: str, size: str, ad_index: int) -> Dict[str, Any]:
    prompt = _text_prompt_for_one_ad(product, description, size, ad_index)

    def call():
        resp = client.chat.completions.create(
            model=OPENAI_TEXT_MODEL,
            messages=[
                {"role": "system", "content": _system_prompt_engine()},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        return resp.choices[0].message.content

    raw = _retry_backoff(call)
    try:
        return json.loads(raw)
    except Exception:
        s = (raw or "").strip()
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(s[start:end + 1])
        raise ValueError("Text model did not return valid JSON.")

def _save_image_b64(b64: str, filename: str) -> str:
    import base64
    path = os.path.join(DATA_DIR, filename)
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64))
    return path

def _add_headline_to_image(image_path: str, headline: str) -> None:
    """Draw headline directly onto the image (no special bar/background), within safe margins.
    Headline must sit on the BACKGROUND area (negative space) of object A, not on the A/B objects.
    We can't 'detect' A/B, so we enforce a top-area placement and require the prompt to leave space.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as e:
        raise RuntimeError("Pillow is required to render headline. Add 'pillow' to requirements.txt") from e

    h = (headline or "").strip()
    if not h:
        return

    img = Image.open(image_path).convert("RGB")
    W, H = img.size

    pad_x = int(W * 0.05)
    pad_y = int(H * 0.04)
    region_h = int(H * 0.18)

    draw = ImageDraw.Draw(img)

    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    font = None
    base_size = max(32, int(W * 0.075))
    for fp in font_paths:
        if os.path.exists(fp):
            font = ImageFont.truetype(fp, size=base_size)
            break
    if font is None:
        font = ImageFont.load_default()

    max_w = W - 2 * pad_x

    def measure(text: str, fnt):
        bbox = draw.textbbox((0, 0), text, font=fnt, stroke_width=5)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    def wrap_lines(fnt):
        words = h.split()
        lines = []
        line = ""
        for w in words:
            test = (line + " " + w).strip()
            tw, _ = measure(test, fnt)
            if tw <= max_w:
                line = test
            else:
                if line:
                    lines.append(line)
                line = w
        if line:
            lines.append(line)
        return lines

    def total_height(lines, fnt):
        hh = 0
        for ln in lines:
            _, th = measure(ln, fnt)
            hh += th
        hh += int(len(lines) * (H * 0.01))
        return hh

    lines = wrap_lines(font)

    # shrink font until lines fit in region height (prefer <=2 lines)
    for _ in range(14):
        th = total_height(lines, font)
        if th <= region_h and len(lines) <= 2:
            break
        # shrink
        new_size = max(12, int(getattr(font, "size", base_size) * 0.9))
        try:
            if os.path.exists(font_paths[0]):
                font = ImageFont.truetype(font_paths[0], size=new_size)
            elif os.path.exists(font_paths[1]):
                font = ImageFont.truetype(font_paths[1], size=new_size)
            else:
                font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()
        lines = wrap_lines(font)

    y = pad_y
    for ln in lines:
        tw, th = measure(ln, font)
        x = pad_x + max(0, (max_w - tw) // 2)
        draw.text(
            (x, y),
            ln,
            font=font,
            fill=(255, 255, 255),
            stroke_width=5,
            stroke_fill=(0, 0, 0),
        )
        y += th + int(H * 0.01)

    img.save(image_path, format="JPEG", quality=92, optimize=True)


def _build_image_prompt_visual_only(base_prompt: str) -> str:
    """Return an image prompt that produces ONLY the photorealistic visual (no text).
    We will render the headline ourselves in backend to guarantee correctness.
    """
    rules = """FINISHED ADVERTISEMENT PHOTO (photorealistic).
NO TEXT IN IMAGE: Do not render any letters, numbers, logos, labels, signs, watermarks, UI, or captions.
COMPOSITION: Leave clean negative space on the BACKGROUND of object A (top/side/bottom) so a headline can be placed there later, without touching or covering the visual objects A/B.
BACKGROUND: Must be the classic background of object A (H06), even in SIDE BY SIDE.
STRICT: A/B must be the only visual focal objects.

BASE VISUAL PROMPT:
"""
    return rules + (base_prompt or "").strip() + "\n"

def _generate_image(image_prompt: str, size: str, filename: str) -> str:
    def call():
        img = client.images.generate(
            model=OPENAI_IMAGE_MODEL,
            prompt=image_prompt,
            size=size,
        )
        return img.data[0].b64_json

    b64 = _retry_backoff(call)
    return _save_image_b64(b64, filename)

def _word_count(text: str) -> int:
    return len([w for w in (text or "").strip().split() if w])

def _normalize_headline(headline: str, product: str) -> str:
    h = (headline or "").strip()
    h = h.replace('"', '').replace("'", "")
    if product and product.lower() not in h.lower():
        h = f"{product} {h}".strip() if h else f"{product} Made For You"
    words = [w for w in h.split() if w]
    if len(words) < 3:
        h = f"{product} Made For You".strip()
        words = h.split()
    if len(words) > 7:
        h = " ".join(words[:7])
    return h

def _headline_too_similar(headline: str, description: str) -> bool:
    h_words = [w.lower() for w in re.findall(r"[a-zA-Z0-9]+", headline or "") if w]
    d_words = set(w.lower() for w in re.findall(r"[a-zA-Z0-9]+", description or "") if w)
    if not h_words:
        return True
    overlap = sum(1 for w in h_words if w in d_words)
    return (overlap / max(1, len(h_words))) >= 0.6

def _regen_headline(product: str, description: str, intent: str) -> str:
    prompt = f"""Create ONE advertising headline.

Rules:
- 3 to 7 words ONLY
- Must include the product name: {product}
- Must be original (not a quote, not a rephrase of the product description)
- No quotes
- No heavy punctuation

Product description: {description}
Ad intent: {intent}

Return ONLY the headline text."""

    def call():
        resp = client.chat.completions.create(
            model=OPENAI_TEXT_MODEL,
            messages=[
                {"role": "system", "content": "Return only the headline text."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        return resp.choices[0].message.content

    raw = _retry_backoff(call)
    return _normalize_headline(raw, product)

def _ensure_50_words(text: str) -> str:
    words = [w for w in (text or "").strip().split() if w]
    if len(words) == 50:
        return " ".join(words)
    if len(words) > 50:
        return " ".join(words[:50])
    filler = ["today", "with", "ease", "and", "confidence"]
    i = 0
    while len(words) < 50:
        words.append(filler[i % len(filler)])
        i += 1
    return " ".join(words[:50])

@app.post("/generate")
def generate():
    payload = request.get_json(silent=True) or {}
    product, description, size, err = _validate_payload(payload)
    if err:
        return err

    # sequential attempts: frontend sends attempt 1..3
    ad_number = int(payload.get("ad_number") or payload.get("attempt") or 1)
    if ad_number < 1: ad_number = 1
    if ad_number > 3: ad_number = 3

    job_id = (payload.get("job_id") or "").strip() or str(uuid.uuid4())
    log.info("Generate request received job_id=%s ad_number=%s size=%s", job_id, ad_number, size)

    try:
        spec = _generate_text_spec(product, description, size, ad_number)

        headline = _normalize_headline(spec.get("headline", ""), product)
        if _headline_too_similar(headline, description):
            headline = _regen_headline(product, description, spec.get("intent", ""))

        marketing = _ensure_50_words(spec.get("marketing_text_50_words", ""))

        image_prompt = (spec.get("image_prompt") or "").strip()
        if not image_prompt:
            raise ValueError("Missing image_prompt from text model.")

        img_filename = f"{job_id}_ad{ad_number}.jpg"
        _generate_image(_build_image_prompt_visual_only(image_prompt), size, img_filename)
        _add_headline_to_image(os.path.join(DATA_DIR, img_filename), headline)

        txt_filename = f"{job_id}_ad{ad_number}.txt"
        txt_path = os.path.join(DATA_DIR, txt_filename)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(marketing)

        # build data URL so frontend can show immediately (no storage dependency)
        import base64
        img_path = os.path.join(DATA_DIR, img_filename)
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        image_data_url = "data:image/jpeg;base64," + b64

        root = request.url_root.rstrip("/")
        zip_url = f"{root}/zip/{job_id}/{ad_number}"

        ad_obj = {
            "ad_number": ad_number,
            "headline": headline,
            "text": marketing,
            "image_data_url": image_data_url,
            "zip_url": zip_url
        }

        # store/merge in JOBS
        job = JOBS.get(job_id) or {"ads": [], "created_at": time.time()}
        # replace if exists
        ads = [a for a in job.get("ads", []) if int(a.get("ad_number",0)) != ad_number]
        ads.append({k:v for k,v in ad_obj.items() if k != "image_data_url"})  # store lightweight (no big b64)
        ads.sort(key=lambda x: int(x.get("ad_number", 0)))
        job["ads"] = ads
        JOBS[job_id] = job

        return jsonify({"job_id": job_id, "ad": ad_obj}), 200

    except RateLimitError:
        return _error(503, "OpenAI rate limit (429). Please try again in a moment.", "rate_limited")
    except (APIConnectionError, APIError):
        log.exception("OpenAI/API error job_id=%s ad_number=%s", job_id, ad_number)
        return _error(500, "Network / OpenAI error. Please try again.", "upstream_error")
    except Exception:
        log.exception("Unexpected error job_id=%s ad_number=%s", job_id, ad_number)
        return _error(500, "Generation failed. Please try again.", "generation_error")


@app.get("/file/<path:filename>")
def file_get(filename: str):
    safe = secure_filename(filename)
    path = os.path.join(DATA_DIR, safe)
    if not os.path.exists(path):
        return _error(404, "File not found.", "not_found")
    return send_file(path)

@app.get("/zip/<job_id>/<int:ad_number>")
def zip_get(job_id: str, ad_number: int):
    if ad_number not in (1, 2, 3):
        return _error(400, "Invalid ad number.", "bad_request")

    img_name = f"{job_id}_ad{ad_number}.jpg"
    txt_name = f"{job_id}_ad{ad_number}.txt"
    img_path = os.path.join(DATA_DIR, img_name)
    txt_path = os.path.join(DATA_DIR, txt_name)

    if not os.path.exists(img_path) or not os.path.exists(txt_path):
        return _error(404, "ZIP content not found. Generate first.", "not_found")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(img_path, arcname=f"ad_{ad_number}.jpg")
        z.write(txt_path, arcname=f"ad_{ad_number}.txt")
    buf.seek(0)

    return send_file(
        buf,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"ACE_ad_{ad_number}.zip"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)