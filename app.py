import os
import uuid
import time
import threading
import json
from io import BytesIO
from zipfile import ZipFile
from pathlib import Path

import requests
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
OPENAI_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini")

ALLOWED_SIZES = {
    "1024x1024": (1024, 1024),
    "1024x1536": (1024, 1536),
    "1536x1024": (1536, 1024),
}

# Optional draft mode to reduce load:
# - If DRAFT_MODE=1, backend will generate images in DRAFT_SIZE regardless of user's selection.
DRAFT_MODE = os.getenv("DRAFT_MODE", "0").strip() == "1"
DRAFT_SIZE = os.getenv("DRAFT_SIZE", "1024x1024").strip().lower()
if DRAFT_SIZE not in ALLOWED_SIZES:
    DRAFT_SIZE = "1024x1024"

JOBS_ROOT = Path(os.getenv("JOBS_ROOT", "/tmp/ace_jobs"))
JOBS_ROOT.mkdir(parents=True, exist_ok=True)

# Soft retry policy with countdown messaging
MAX_RETRY_SECONDS = int(os.getenv("OPENAI_MAX_RETRY_SECONDS", "330"))
BASE_BACKOFF = float(os.getenv("OPENAI_RETRY_BASE_BACKOFF", "2.0"))
MAX_BACKOFF = float(os.getenv("OPENAI_RETRY_MAX_BACKOFF", "30.0"))

def job_dir(job_id: str) -> Path:
    p = JOBS_ROOT / job_id
    p.mkdir(parents=True, exist_ok=True)
    return p

def job_json_path(job_id: str) -> Path:
    return job_dir(job_id) / "job.json"

def write_job(job_id: str, data: dict):
    d = job_dir(job_id)
    tmp = d / "job.json.tmp"
    final = d / "job.json"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, final)

def read_job(job_id: str):
    p = job_json_path(job_id)
    if not p.exists():
        return None
    for _ in range(5):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            time.sleep(0.05)
    return None

def _placeholder_jpg(width: int, height: int) -> bytes:
    img = Image.new("RGB", (width, height), (0, 0, 0))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=92, optimize=True)
    return buf.getvalue()

def _openai_headers():
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing")
    return {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

def _should_retry(status_code: int) -> bool:
    return status_code in (429, 500, 502, 503, 504)

def _request_with_retry(method: str, url: str, *, headers: dict, json_body: dict, timeout: int, on_retry=None):
    start = time.time()
    backoff = BASE_BACKOFF
    last_status = None
    last_text = None

    while True:
        try:
            r = requests.request(method, url, headers=headers, json=json_body, timeout=timeout)
            last_status = r.status_code
            if r.status_code == 200:
                return r

            try:
                last_text = r.text
            except Exception:
                last_text = None

            elapsed = time.time() - start
            if _should_retry(r.status_code) and elapsed < MAX_RETRY_SECONDS:
                ra = r.headers.get("retry-after")
                if ra:
                    try:
                        sleep_s = float(ra)
                    except Exception:
                        sleep_s = backoff
                else:
                    sleep_s = backoff
                sleep_s = min(MAX_BACKOFF, max(1.0, sleep_s))
                if on_retry:
                    on_retry(int(sleep_s), r.status_code)
                time.sleep(sleep_s)
                backoff = min(MAX_BACKOFF, backoff * 2)
                continue

            r.raise_for_status()
            return r

        except requests.RequestException as e:
            elapsed = time.time() - start
            if elapsed < MAX_RETRY_SECONDS:
                sleep_s = min(MAX_BACKOFF, backoff)
                if on_retry:
                    on_retry(int(sleep_s), last_status or 0)
                time.sleep(sleep_s)
                backoff = min(MAX_BACKOFF, backoff * 2)
                continue
            raise RuntimeError(str(e) if str(e) else f"Request failed (status={last_status}) body={last_text}")

def openai_chat_json(system_prompt: str, user_prompt: str, on_retry=None) -> dict:
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": OPENAI_TEXT_MODEL,
        "temperature": 0.9,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    r = _request_with_retry("POST", url, headers=_openai_headers(), json_body=payload, timeout=90, on_retry=on_retry)
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    return json.loads(content)

def openai_image_bytes(prompt: str, size: str, on_retry=None) -> bytes:
    url = "https://api.openai.com/v1/images/generations"
    payload = {"model": OPENAI_IMAGE_MODEL, "prompt": prompt, "size": size, "n": 1}
    r = _request_with_retry("POST", url, headers=_openai_headers(), json_body=payload, timeout=120, on_retry=on_retry)
    data = r.json()
    item = (data.get("data") or [None])[0] or {}
    if "b64_json" in item and item["b64_json"]:
        import base64
        return base64.b64decode(item["b64_json"])
    if "url" in item and item["url"]:
        img_r = requests.get(item["url"], timeout=120)
        img_r.raise_for_status()
        return img_r.content
    raise RuntimeError("Unexpected image response from OpenAI")

def _set_phase(job_id: str, phase: str, message: str = "", retrying: bool = False, retry_in: int = 0):
    job = read_job(job_id) or {}
    job["phase"] = phase
    job["message"] = message
    job["retrying"] = bool(retrying)
    job["retry_in_seconds"] = int(retry_in) if retry_in else 0
    write_job(job_id, job)

def _image_prompt(product_name: str, product_description: str, objective: str) -> str:
    return (
        f"Photorealistic advertising image for: {product_name}. "
        f"Objective: {objective}. "
        f"Based on this product: {product_description}. "
        "High-end studio photo aesthetic, realistic lighting, realistic materials, sharp focus. "
        "NO text, NO logos, NO letters, NO numbers, NO watermark."
    )

def _pick_size(user_size: str) -> str:
    return DRAFT_SIZE if DRAFT_MODE else user_size

@app.get("/health")
def health():
    return "ok", 200

def generate_image_for_index(job_id: str, index: int, product_name: str, product_description: str, size: str):
    w, h = ALLOWED_SIZES[size]
    job = read_job(job_id) or {}
    ads = job.get("ads") or []
    ad = next((a for a in ads if int(a.get("index", 0)) == index), None)
    if not ad:
        raise RuntimeError("Invalid ad index")

    objective = (ad.get("objective") or "").strip() or f"Objective {index}"

    def on_retry_img(retry_in, status):
        _set_phase(job_id, f"retrying_image_{index}", f"עומס זמני על OpenAI (תמונה {index}). מנסה שוב…", True, retry_in)

    _set_phase(job_id, f"image_{index}", f"מייצר תמונה {index}…", False, 0)

    try:
        img_bytes = openai_image_bytes(_image_prompt(product_name, product_description, objective), size=size, on_retry=on_retry_img)
    except Exception:
        img_bytes = _placeholder_jpg(w, h)

    img_path = job_dir(job_id) / f"ad_{index}.jpg"
    with open(img_path, "wb") as f:
        f.write(img_bytes)

    # Update job
    job = read_job(job_id) or {}
    for a in (job.get("ads") or []):
        if int(a.get("index", 0)) == index:
            a["image_ready"] = True
    job["status"] = "partial"
    job["ready"] = True
    job["phase"] = f"done_{index}"
    job["retrying"] = False
    job["retry_in_seconds"] = 0
    job["message"] = f"תמונה {index} מוכנה."
    write_job(job_id, job)

def run_generate_text_and_first_image(job_id: str, product_name: str, product_description: str, requested_size: str):
    size = _pick_size(requested_size)
    try:
        _set_phase(job_id, "text", "מכין טקסטים שיווקיים…", False, 0)

        def on_retry_text(retry_in, status):
            _set_phase(job_id, "retrying_text", "עומס זמני על OpenAI (טקסט). מנסה שוב…", True, retry_in)

        system = (
            "You are ACE ENGINE. Follow these rules strictly:\n"
            "- Create 3 VERY DIFFERENT ad objectives for the same product.\n"
            "- For each ad: create an ORIGINAL headline that includes the product name, 3-7 words.\n"
            "- Create a 50-word marketing copy (not for on-image).\n"
            "- The 3 copies must be VERY DIFFERENT in angle, benefits, and message.\n"
            "- Return valid JSON only."
        )
        user = (
            f"Product name: {product_name}\n"
            f"Product description: {product_description}\n\n"
            "Return JSON with this schema:\n"
            "{\n"
            "  \"ads\": [\n"
            "    { \"index\": 1, \"objective\": \"...\", \"headline\": \"...\", \"copy_50\": \"...\" },\n"
            "    { \"index\": 2, \"objective\": \"...\", \"headline\": \"...\", \"copy_50\": \"...\" },\n"
            "    { \"index\": 3, \"objective\": \"...\", \"headline\": \"...\", \"copy_50\": \"...\" }\n"
            "  ]\n"
            "}"
        )

        plan = openai_chat_json(system, user, on_retry=on_retry_text)
        ads_plan = plan.get("ads") or []
        ads_out = []
        for i in range(1, 4):
            ad = next((a for a in ads_plan if int(a.get("index", 0)) == i), {})
            ads_out.append({
                "index": i,
                "objective": (ad.get("objective") or "").strip() or f"Objective {i}",
                "headline": (ad.get("headline") or "").strip(),
                "text": (ad.get("copy_50") or "").strip(),
                "image_ready": False,
            })

        write_job(job_id, {
            "status": "running",
            "ready": False,
            "size": size,
            "requested_size": requested_size,
            "phase": "image_1",
            "retrying": False,
            "retry_in_seconds": 0,
            "message": "מייצר תמונה ראשונה…",
            "ads": ads_out,
            "error": None
        })

        # Only image #1
        generate_image_for_index(job_id, 1, product_name, product_description, size)

        job = read_job(job_id) or {}
        job["status"] = "partial"
        job["ready"] = True
        job["phase"] = "done_1"
        job["message"] = "תמונה 1 מוכנה. אפשר לבקש עוד תמונה אם אהבת."
        write_job(job_id, job)

    except Exception as e:
        write_job(job_id, {
            "status": "error",
            "ready": False,
            "size": size,
            "requested_size": requested_size,
            "phase": "error",
            "retrying": False,
            "retry_in_seconds": 0,
            "message": "נכשל. נסה שוב מאוחר יותר.",
            "ads": [],
            "error": str(e),
        })

def run_generate_image(job_id: str, index: int, product_name: str, product_description: str):
    job = read_job(job_id)
    if not job:
        return
    size = job.get("size") or "1024x1024"
    try:
        generate_image_for_index(job_id, index, product_name, product_description, size)
        job = read_job(job_id) or {}
        ads = job.get("ads") or []
        if ads and all(bool(a.get("image_ready")) for a in ads):
            job["status"] = "done"
            job["phase"] = "done_3"
            job["message"] = "כל התמונות מוכנות."
            write_job(job_id, job)
    except Exception as e:
        job = read_job(job_id) or {}
        job["status"] = "error"
        job["ready"] = False
        job["phase"] = "error"
        job["retrying"] = False
        job["retry_in_seconds"] = 0
        job["message"] = "נכשל ביצירת תמונה. נסה שוב."
        job["error"] = str(e)
        write_job(job_id, job)

@app.post("/api/generate")
def generate():
    data = request.get_json(force=True) or {}
    product_name = (data.get("product_name") or "").strip()
    product_description = (data.get("product_description") or "").strip()
    size = (data.get("size") or "").strip().lower()

    if not product_name or not product_description or size not in ALLOWED_SIZES:
        return jsonify({"error": "invalid_input"}), 400

    job_id = str(uuid.uuid4())
    write_job(job_id, {
        "status": "running",
        "ready": False,
        "size": _pick_size(size),
        "requested_size": size,
        "phase": "queued",
        "retrying": False,
        "retry_in_seconds": 0,
        "message": "בתור…",
        "ads": [],
        "error": None
    })

    t = threading.Thread(target=run_generate_text_and_first_image, args=(job_id, product_name, product_description, size), daemon=True)
    t.start()

    return jsonify({"job_id": job_id}), 200

@app.post("/api/jobs/<job_id>/generate-image")
def generate_image_endpoint(job_id):
    job = read_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    data = request.get_json(force=True) or {}
    try:
        index = int(data.get("index"))
    except Exception:
        return jsonify({"error": "invalid_index"}), 400
    if index not in (1, 2, 3):
        return jsonify({"error": "invalid_index"}), 400

    # Busy check
    phase = (job.get("phase") or "")
    if phase.startswith("image_") or phase.startswith("retrying_image_") or phase == "text" or phase.startswith("retrying_text"):
        return jsonify({"error": "busy", "message": "Job is busy. Please wait."}), 409

    product_name = (data.get("product_name") or "").strip()
    product_description = (data.get("product_description") or "").strip()
    if not product_name or not product_description:
        return jsonify({"error": "missing_product_fields"}), 400

    _set_phase(job_id, f"image_{index}", f"מייצר תמונה {index}…", False, 0)
    t = threading.Thread(target=run_generate_image, args=(job_id, index, product_name, product_description), daemon=True)
    t.start()
    return jsonify({"ok": True}), 200

@app.get("/api/jobs/<job_id>")
def job_status(job_id):
    job = read_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job), 200

@app.get("/api/jobs/<job_id>/ads/<int:index>/image")
def get_image(job_id, index):
    job = read_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    img_path = job_dir(job_id) / f"ad_{index}.jpg"
    if not img_path.exists():
        return jsonify({"error": "Not found"}), 404
    return send_file(str(img_path), mimetype="image/jpeg", download_name=f"ad_{index}.jpg")

@app.get("/api/jobs/<job_id>/ads/<int:index>/zip")
def download_zip(job_id, index):
    job = read_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    img_path = job_dir(job_id) / f"ad_{index}.jpg"
    if not img_path.exists():
        return jsonify({"error": "Not found"}), 404

    ad = next((a for a in (job.get("ads") or []) if int(a.get("index", 0)) == index), {})
    headline = ad.get("headline", "")
    text = ad.get("text", "")
    objective = ad.get("objective", "")

    zip_buffer = BytesIO()
    with ZipFile(zip_buffer, "w") as z:
        with open(img_path, "rb") as f:
            z.writestr(f"ad_{index}.jpg", f.read())
        z.writestr(f"ad_{index}.txt", text)
        z.writestr(f"ad_{index}_headline.txt", headline)
        z.writestr(f"ad_{index}_objective.txt", objective)

    zip_buffer.seek(0)
    return send_file(zip_buffer, as_attachment=True, download_name=f"ad_{index}.zip", mimetype="application/zip")

if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
