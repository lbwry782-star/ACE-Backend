import os
import io
import time
import json
import uuid
import zipfile
import base64
import logging
from typing import Dict, Any, List, Optional, Tuple

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from openai import OpenAI
from openai import APIError, RateLimitError, APIConnectionError

ALLOWED_SIZES = {"1024x1024", "1024x1536", "1536x1024"}

OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
OPENAI_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini")
FRONTEND_URL = (os.getenv("FRONTEND_URL") or "").strip()
PORT = int(os.getenv("PORT", "10000"))

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ace-backend")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)

if FRONTEND_URL:
    CORS(app, resources={r"/*": {"origins": [FRONTEND_URL]}})
    log.info("CORS restricted to FRONTEND_URL=%s", FRONTEND_URL)
else:
    CORS(app)
    log.warning("FRONTEND_URL not set; CORS allows all origins")

# In-memory cache: job_id -> {"ads": { "1": {"img": bytes, "txt": str}, ... }, "ts": float}
CACHE: Dict[str, Dict[str, Any]] = {}
CACHE_TTL_SECONDS = 30 * 60  # 30 minutes

@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200

def _error(status: int, message: str, code: str = "error"):
    return jsonify({"error": code, "message": message}), status

def _cleanup_cache():
    now = time.time()
    dead = [job_id for job_id, data in CACHE.items() if now - float(data.get("ts", 0)) > CACHE_TTL_SECONDS]
    for job_id in dead:
        CACHE.pop(job_id, None)

def _validate_payload(payload: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], str, int, str, Optional[Any]]:
    product = (payload.get("product") or "").strip()
    description = (payload.get("description") or "").strip()
    size = (payload.get("size") or "").strip() or "1024x1024"
    ad_number = int(payload.get("ad_number") or 1)
    job_id = (payload.get("job_id") or "").strip()

    if not product:
        return None, None, "", 0, "", _error(400, "Missing or empty 'product'.", "bad_request")
    if not description:
        return None, None, "", 0, "", _error(400, "Missing or empty 'description'.", "bad_request")
    if size not in ALLOWED_SIZES:
        return None, None, "", 0, "", _error(400, f"Invalid 'size'. Must be one of: {sorted(ALLOWED_SIZES)}", "bad_request")
    if ad_number not in (1,2,3):
        return None, None, "", 0, "", _error(400, "Invalid 'ad_number'. Must be 1, 2, or 3.", "bad_request")
    if not job_id:
        job_id = str(uuid.uuid4())
    return product, description, size, ad_number, job_id, None

def _retry_backoff(fn, *, max_retries: int = 5, delays: List[float] = [2.0, 5.0, 10.0, 20.0, 30.0]):
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

def _engine_prompt(product: str, description: str, size: str, ad_number: int) -> str:
    return f"""
Follow exactly the ACE ENGINE rules (H00–H10). No interpretation.

INPUT:
- product_name: {product}
- product_description: {description}
- size: {size}
- ad_number: {ad_number} (attempt {ad_number}/3; each attempt MUST use a different advertising intent)

APPLY THESE RULES (strict):
H00 Audience: infer only from product+description.
H01: define 1 unique intent for this ad_number (different from other attempts).
H02: generate EXACTLY 80 associative PHYSICAL objects derived from the intent (no abstract).
H03: pick A first (central meaning), then B (highlight) from the 80. No objects with printed text/logos/labels.
H04: choose the camera angle/projection that maximizes silhouette similarity between A and B.
H05: if high similarity -> HYBRID; if medium -> SIDE_BY_SIDE; if not immediate -> reject and re-pick A/B until valid. No perspective tricks.
H06: background MUST be the classic background of A, even in SIDE_BY_SIDE.
H07: headline 3–7 words, includes product name, original (not a quote; not a variation of description). Not on image.
H08: marketing text EXACTLY 50 words. Not on image.
H09: photorealistic photo. No vector/illustration/3D/AI-art.
NO TEXT ON IMAGE: no letters, no words, no logos, no labels, no signage, no UI, no readable screens.

Return JSON ONLY in this schema:
{{
  "ad_number": {ad_number},
  "audience": {{
    "age_range": "...",
    "lifestyle": "...",
    "needs": "...",
    "knowledge_level": "...",
    "pain_points": "..."
  }},
  "intent": "...",
  "objects_80": ["obj1", "...", "obj80"],
  "A": "...",
  "B": "...",
  "projection_angle": "...",
  "composition_mode": "HYBRID" | "SIDE_BY_SIDE",
  "classic_background_of_A": "...",
  "headline": "...",
  "marketing_text_50_words": "...",
  "image_prompt": "Photorealistic prompt for gpt-image-1. Describe HYBRID or SIDE_BY_SIDE; emphasize projection; classic background of A. No text."
}}
""".strip()

def _generate_engine_spec(product: str, description: str, size: str, ad_number: int) -> Dict[str, Any]:
    prompt = _engine_prompt(product, description, size, ad_number)

    def call():
        resp = client.chat.completions.create(
            model=OPENAI_TEXT_MODEL,
            messages=[
                {"role": "system", "content": _system_prompt_engine()},
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,
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

def _generate_image_bytes(image_prompt: str, size: str) -> bytes:
    anti_text = " No text, no letters, no words, no logos, no labels, no signage, no UI, no readable screens. "
    prompt = (image_prompt or "").strip() + anti_text

    def call():
        img = client.images.generate(
            model=OPENAI_IMAGE_MODEL,
            prompt=prompt,
            size=size,
        )
        return img.data[0].b64_json

    b64 = _retry_backoff(call)
    return base64.b64decode(b64)

@app.post("/generate")
def generate():
    _cleanup_cache()

    payload = request.get_json(silent=True) or {}
    product, description, size, ad_number, job_id, err = _validate_payload(payload)
    if err:
        return err

    log.info("Generate request job_id=%s ad_number=%s size=%s origin=%s", job_id, ad_number, size, request.headers.get("Origin"))

    try:
        spec = _generate_engine_spec(product, description, size, ad_number)
        headline = _normalize_headline(spec.get("headline",""), product)
        marketing = _ensure_50_words(spec.get("marketing_text_50_words",""))
        image_prompt = (spec.get("image_prompt") or "").strip()
        if not image_prompt:
            raise ValueError("Missing image_prompt")

        image_bytes = _generate_image_bytes(image_prompt, size)

        job = CACHE.get(job_id) or {"ads": {}, "ts": time.time()}
        job["ads"][str(ad_number)] = {"img": image_bytes, "txt": marketing}
        job["ts"] = time.time()
        CACHE[job_id] = job

        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        root = request.url_root.rstrip("/")

        return jsonify({
            "job_id": job_id,
            "ad_number": ad_number,
            "ad": {
                "headline": headline,
                "text": marketing,
                "image_data_url": f"data:image/jpeg;base64,{image_b64}",
                "zip_url": f"{root}/zip/{job_id}/{ad_number}"
            }
        }), 200

    except RateLimitError:
        return _error(503, "OpenAI rate limit (429). Please try again in a moment.", "rate_limited")
    except (APIConnectionError, APIError):
        log.exception("OpenAI/API error")
        return _error(500, "Network / OpenAI error. Please try again.", "upstream_error")
    except Exception:
        log.exception("Generation error")
        return _error(500, "Generation failed. Please try again.", "generation_error")

@app.get("/zip/<job_id>/<int:ad_number>")
def zip_get(job_id: str, ad_number: int):
    _cleanup_cache()
    job = CACHE.get(job_id)
    if not job:
        return _error(404, "ZIP content not found (expired). Please generate again.", "not_found")

    item = (job.get("ads") or {}).get(str(ad_number))
    if not item:
        return _error(404, "ZIP content not found for this ad. Please generate that ad again.", "not_found")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"ad_{ad_number}.jpg", item["img"])
        z.writestr(f"ad_{ad_number}.txt", item["txt"])
    buf.seek(0)

    return send_file(
        buf,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"ACE_ad_{ad_number}.zip"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
