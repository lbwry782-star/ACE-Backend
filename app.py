import os
import io
import time
import json
import uuid
import zipfile
import logging
from typing import Dict, Any, List, Optional, Tuple

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

from openai import OpenAI
from openai import APIError, RateLimitError, APIConnectionError

ALLOWED_SIZES = {"1024x1024", "1024x1792", "1792x1024"}
SIZE_MAP = {"1024x1536": "1024x1792", "1536x1024": "1792x1024"}

OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
OPENAI_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini")
FRONTEND_URL = (os.getenv("FRONTEND_URL") or "").strip()
PORT = int(os.getenv("PORT", "10000"))

DATA_DIR = os.path.join(os.getcwd(), "generated")
os.makedirs(DATA_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ace-backend")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)

# Robust CORS: restrict if FRONTEND_URL set, else allow all (temporary safety net)
if FRONTEND_URL:
    CORS(app, resources={r"/*": {"origins": [FRONTEND_URL]}})
    log.info("CORS restricted to FRONTEND_URL=%s", FRONTEND_URL)
else:
    CORS(app)
    log.warning("FRONTEND_URL not set; CORS allows all origins (temporary)")

@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200

def _error(status: int, message: str, code: str = "error"):
    return jsonify({"error": code, "message": message}), status

def _normalize_size(size_raw: str) -> str:
    size = (size_raw or "").strip()
    if not size:
        return "1024x1024"
    return SIZE_MAP.get(size, size)

def _validate_payload(payload: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], str, Optional[Any]]:
    product = (payload.get("product") or "").strip()
    description = (payload.get("description") or "").strip()
    size = _normalize_size(payload.get("size") or "")

    if not product:
        return None, None, "", _error(400, "Missing or empty 'product'.", "bad_request")
    if not description:
        return None, None, "", _error(400, "Missing or empty 'description'.", "bad_request")
    if size not in ALLOWED_SIZES:
        return None, None, "", _error(400, f"Invalid 'size'. Must be one of: {sorted(ALLOWED_SIZES)}", "bad_request")
    return product, description, size, None

def _retry_backoff(fn, *, max_retries: int = 3, delays: List[float] = [2.0, 5.0, 10.0]):
    last_exc = None
    for attempt in range(max_retries):
        try:
            return fn()
        except RateLimitError as e:
            last_exc = e
            wait = delays[min(attempt, len(delays)-1)]
            log.warning("OpenAI 429 RateLimitError. Retry %s/%s after %ss", attempt+1, max_retries, wait)
            time.sleep(wait)
        except APIConnectionError as e:
            last_exc = e
            wait = delays[min(attempt, len(delays)-1)]
            log.warning("OpenAI connection error. Retry %s/%s after %ss", attempt+1, max_retries, wait)
            time.sleep(wait)
    raise last_exc

def _system_prompt_engine() -> str:
    return "You are ACE ENGINE. Output ONLY valid JSON. No extra text."

def _text_prompt_for_one_ad(product: str, description: str, size: str, ad_index: int) -> str:
    return f"""
Create ONE advertising ad spec for the product below.

PRODUCT:
- name: {product}
- description: {description}
- size: {size}
- ad_number: {ad_index}

Return JSON ONLY:
{{
  "ad_number": {ad_index},
  "headline": "...",
  "marketing_text_50_words": "...",
  "image_prompt": "Photorealistic prompt for gpt-image-1, NO text on image."
}}

Rules:
- Headline: 3-7 words, includes product name, not a quote, not copied from description, not on image.
- Marketing text: EXACTLY 50 words, not on image.
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
            return json.loads(s[start:end+1])
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

def _ensure_50_words(text: str) -> str:
    words = [w for w in (text or "").strip().split() if w]
    if len(words) == 50:
        return " ".join(words)
    if len(words) > 50:
        return " ".join(words[:50])
    filler = ["today","with","ease","and","confidence"]
    i = 0
    while len(words) < 50:
        words.append(filler[i % len(filler)]); i += 1
    return " ".join(words[:50])

def _normalize_headline(headline: str, product: str) -> str:
    h = (headline or "").strip()
    if product and product.lower() not in h.lower():
        h = f"{product} {h}".strip()
    if len(h.split()) < 3:
        h = f"{product} Made For You"
    if len(h.split()) > 7:
        h = " ".join(h.split()[:7])
    return h

@app.post("/generate")
def generate():
    payload = request.get_json(silent=True) or {}
    product, description, size, err = _validate_payload(payload)
    if err:
        return err

    job_id = str(uuid.uuid4())
    log.info("Generate request received job_id=%s size=%s origin=%s", job_id, size, request.headers.get("Origin"))

    ads_out = []
    for ad_index in (1,2,3):
        try:
            spec = _generate_text_spec(product, description, size, ad_index)
            headline = _normalize_headline(spec.get("headline",""), product)
            marketing = _ensure_50_words(spec.get("marketing_text_50_words",""))
            image_prompt = (spec.get("image_prompt") or "").strip()
            if not image_prompt:
                raise ValueError("Missing image_prompt")

            img_filename = f"{job_id}_ad{ad_index}.jpg"
            _generate_image(image_prompt, size, img_filename)

            txt_filename = f"{job_id}_ad{ad_index}.txt"
            with open(os.path.join(DATA_DIR, txt_filename), "w", encoding="utf-8") as f:
                f.write(marketing)

            root = request.url_root.rstrip("/")
            ads_out.append({
                "ad_number": ad_index,
                "headline": headline,
                "text": marketing,
                "image_url": f"{root}/file/{img_filename}",
                "zip_url": f"{root}/zip/{job_id}/{ad_index}",
            })

        except RateLimitError:
            return _error(503, "OpenAI rate limit (429). Please try again in a moment.", "rate_limited")
        except (APIConnectionError, APIError):
            log.exception("OpenAI/API error")
            return _error(500, "Network / OpenAI error. Please try again.", "upstream_error")
        except Exception:
            log.exception("Generation error")
            return _error(500, "Generation failed. Please try again.", "generation_error")

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
    if ad_number not in (1,2,3):
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

    return send_file(buf, mimetype="application/zip", as_attachment=True, download_name=f"ACE_ad_{ad_number}.zip")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
