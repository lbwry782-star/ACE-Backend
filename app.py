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

# CORS: prefer allow only FRONTEND_URL; else allow all (temporary)
if FRONTEND_URL:
    CORS(app, resources={r"/*": {"origins": [FRONTEND_URL]}})
else:
    CORS(app)

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
- Ad includes ONLY VISUAL + HEADLINE (H07). Headline includes product name, 3-7 words, original, not a quote nor a variation of the product description. Headline is NOT on the image.
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
        resp = client.responses.create(
            model=OPENAI_TEXT_MODEL,
            input=[
                {"role": "system", "content": _system_prompt_engine()},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        return resp.output_text

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

    job_id = str(uuid.uuid4())
    log.info("Generate request received job_id=%s size=%s", job_id, size)

    ads_out = []
    for ad_index in (1, 2, 3):
        try:
            log.info("Generating ad %s/3 job_id=%s", ad_index, job_id)
            spec = _generate_text_spec(product, description, size, ad_index)

            headline = _normalize_headline(spec.get("headline", ""), product)
            marketing = _ensure_50_words(spec.get("marketing_text_50_words", ""))

            image_prompt = (spec.get("image_prompt") or "").strip()
            if not image_prompt:
                raise ValueError("Missing image_prompt from text model.")

            img_filename = f"{job_id}_ad{ad_index}.jpg"
            _generate_image(image_prompt, size, img_filename)

            txt_filename = f"{job_id}_ad{ad_index}.txt"
            txt_path = os.path.join(DATA_DIR, txt_filename)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(marketing)

            root = request.url_root.rstrip("/")
            image_url = f"{root}/file/{img_filename}"
            zip_url = f"{root}/zip/{job_id}/{ad_index}"

            ads_out.append({
                "ad_number": ad_index,
                "headline": headline,
                "text": marketing,
                "image_url": image_url,
                "zip_url": zip_url
            })

        except RateLimitError:
            return _error(503, "OpenAI rate limit (429). Please try again in a moment.", "rate_limited")
        except (APIConnectionError, APIError):
            log.exception("OpenAI/API error on ad %s job_id=%s", ad_index, job_id)
            return _error(500, "Network / OpenAI error. Please try again.", "upstream_error")
        except Exception:
            log.exception("Unexpected error on ad %s job_id=%s", ad_index, job_id)
            return _error(500, "Generation failed. Please try again.", "generation_error")

    JOBS[job_id] = {"ads": ads_out, "created_at": time.time()}
    return jsonify({"job_id": job_id, "ads": ads_out}), 200

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
