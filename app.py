import os
import io
import time
import json
import uuid
import zipfile
import base64
import logging
from typing import Dict, Any, List, Optional, Tuple

from flask import Flask, request, jsonify
from flask_cors import CORS

from openai import OpenAI
from openai import APIError, RateLimitError, APIConnectionError

# gpt-image-1 supported sizes (as confirmed by API error message)
ALLOWED_SIZES = {"1024x1024", "1024x1536", "1536x1024"}

OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
OPENAI_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini")
FRONTEND_URL = (os.getenv("FRONTEND_URL") or "").strip()
PORT = int(os.getenv("PORT", "10000"))

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
    def call():
        img = client.images.generate(
            model=OPENAI_IMAGE_MODEL,
            prompt=image_prompt,
            size=size,
        )
        return img.data[0].b64_json

    b64 = _retry_backoff(call)
    return base64.b64decode(b64)

def _zip_bytes(ad_number: int, image_bytes: bytes, marketing_text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"ad_{ad_number}.jpg", image_bytes)
        z.writestr(f"ad_{ad_number}.txt", marketing_text)
    return buf.getvalue()

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

            image_bytes = _generate_image_bytes(image_prompt, size)

            image_b64 = base64.b64encode(image_bytes).decode("ascii")
            zip_b64 = base64.b64encode(_zip_bytes(ad_index, image_bytes, marketing)).decode("ascii")

            ads_out.append({
                "ad_number": ad_index,
                "headline": headline,
                "text": marketing,
                "image_data_url": f"data:image/jpeg;base64,{image_b64}",
                "zip_data_url": f"data:application/zip;base64,{zip_b64}"
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

# Legacy endpoints intentionally disabled (filesystem-less V4)
@app.get("/file/<path:filename>")
def file_get(filename: str):
    return _error(410, "File endpoint disabled. Use embedded image_data_url.", "gone")

@app.get("/zip/<job_id>/<int:ad_number>")
def zip_get(job_id: str, ad_number: int):
    return _error(410, "ZIP endpoint disabled. Use embedded zip_data_url.", "gone")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
