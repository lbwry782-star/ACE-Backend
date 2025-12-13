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

JOBS_ROOT = Path(os.getenv("JOBS_ROOT", "/tmp/ace_jobs"))
JOBS_ROOT.mkdir(parents=True, exist_ok=True)

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
    for _ in range(3):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            time.sleep(0.05)
        except Exception:
            break
    return None

def _placeholder_jpg(width: int, height: int) -> bytes:
    img = Image.new("RGB", (width, height), (0, 0, 0))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=92, optimize=True)
    return buf.getvalue()

def _openai_headers():
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing")
    return {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

def openai_chat_json(system_prompt: str, user_prompt: str) -> dict:
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
    r = requests.post(url, headers=_openai_headers(), json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    return json.loads(content)

def openai_image_bytes(prompt: str, size: str) -> bytes:
    url = "https://api.openai.com/v1/images/generations"
    payload = {"model": OPENAI_IMAGE_MODEL, "prompt": prompt, "size": size, "n": 1}
    r = requests.post(url, headers=_openai_headers(), json=payload, timeout=240)
    r.raise_for_status()
    data = r.json()
    item = (data.get("data") or [None])[0] or {}
    if "b64_json" in item and item["b64_json"]:
        import base64
        return base64.b64decode(item["b64_json"])
    if "url" in item and item["url"]:
        img_r = requests.get(item["url"], timeout=240)
        img_r.raise_for_status()
        return img_r.content
    raise RuntimeError("Unexpected image response from OpenAI")

@app.get("/health")
def health():
    return "ok", 200

def run_generation(job_id: str, product_name: str, product_description: str, size: str):
    w, h = ALLOWED_SIZES[size]
    try:
        system = (
            "You are ACE ENGINE. Follow these rules strictly:\n"
            "- Create 3 DIFFERENT ad objectives for the same product.\n"
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

        plan = openai_chat_json(system, user)
        ads_plan = plan.get("ads") or []

        ads_out = []
        for i in range(1, 4):
            ad = next((a for a in ads_plan if int(a.get("index", 0)) == i), {})
            objective = (ad.get("objective") or "").strip() or f"Objective {i}"
            headline = (ad.get("headline") or "").strip()
            copy_50 = (ad.get("copy_50") or "").strip()

            img_prompt = (
                f"Photorealistic advertising image for: {product_name}. "
                f"Objective: {objective}. "
                f"Based on this product: {product_description}. "
                "High-end studio photo aesthetic, realistic lighting, realistic materials, sharp focus. "
                "NO text, NO logos, NO letters, NO numbers, NO watermark."
            )

            try:
                img_bytes = openai_image_bytes(img_prompt, size=size)
            except Exception:
                img_bytes = _placeholder_jpg(w, h)

            img_path = job_dir(job_id) / f"ad_{i}.jpg"
            with open(img_path, "wb") as f:
                f.write(img_bytes)

            ads_out.append({"index": i, "objective": objective, "headline": headline, "text": copy_50})

        write_job(job_id, {"status": "done", "ready": True, "size": size, "ads": ads_out, "error": None})

    except Exception as e:
        write_job(job_id, {"status": "error", "ready": False, "size": size, "ads": [], "error": str(e)})

@app.post("/api/generate")
def generate():
    data = request.get_json(force=True) or {}
    product_name = (data.get("product_name") or "").strip()
    product_description = (data.get("product_description") or "").strip()
    size = (data.get("size") or "").strip().lower()

    if not product_name or not product_description or size not in ALLOWED_SIZES:
        return jsonify({"error": "invalid_input"}), 400

    job_id = str(uuid.uuid4())
    write_job(job_id, {"status": "running", "ready": False, "size": size, "ads": [], "error": None})

    t = threading.Thread(target=run_generation, args=(job_id, product_name, product_description, size), daemon=True)
    t.start()

    return jsonify({"job_id": job_id}), 200

@app.get("/api/jobs/<job_id>")
def job_status(job_id):
    job = read_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job), 200

@app.get("/api/jobs/<job_id>/ads/<int:index>/image")
def get_image(job_id, index):
    job = read_job(job_id)
    if not job or job.get("status") != "done":
        return jsonify({"error": "Not ready"}), 400
    img_path = job_dir(job_id) / f"ad_{index}.jpg"
    if not img_path.exists():
        return jsonify({"error": "Not found"}), 404
    return send_file(str(img_path), mimetype="image/jpeg", download_name=f"ad_{index}.jpg")

@app.get("/api/jobs/<job_id>/ads/<int:index>/zip")
def download_zip(job_id, index):
    job = read_job(job_id)
    if not job or job.get("status") != "done":
        return jsonify({"error": "Not ready"}), 400

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
