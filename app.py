import os
import uuid
import time
import threading
from io import BytesIO
from zipfile import ZipFile
import base64

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
OPENAI_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini")

# In-memory job store (simple MVP)
JOBS = {}

# Valid tiny JPEG so browsers can render image endpoints
TINY_JPG_BYTES = base64.b64decode("/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAALCAABAAEBAREA/8QAFQABAQAAAAAAAAAAAAAAAAAAAAb/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAb/xAAUEQEAAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCkA//Z")

@app.get("/health")
def health():
    return "ok", 200


def run_generation(job_id: str, product_name: str, product_description: str, size: str):
    try:
        # Simulate engine runtime (real engine would call OpenAI here)
        time.sleep(8)

        ads = []
        for i in range(1, 4):
            # Store image bytes (valid JPG)
            JOBS[job_id]["images"][i] = TINY_JPG_BYTES

            # 50-word-ish marketing text placeholder
            text = (
                f"{product_name} helps you reach your goal with a clear, practical approach. "
                f"Built for people who need results fast, it reduces friction, saves time, "
                f"and delivers a confident experience. This ad highlights a distinct benefit "
                f"that fits the audience and matches the chosen format for conversion."
            )

            ads.append({
                "index": i,
                "text": text
            })

        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["ready"] = True
        JOBS[job_id]["ads"] = ads

    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["ready"] = False
        JOBS[job_id]["error"] = str(e)


@app.post("/api/generate")
def generate():
    data = request.get_json(force=True) or {}

    product_name = (data.get("product_name") or "").strip()
    product_description = (data.get("product_description") or "").strip()
    size = (data.get("size") or "").strip()

    if not product_name or not product_description or size not in ["1024x1024", "1024x1536", "1536x1024"]:
        return jsonify({"error": "invalid_input"}), 400

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "status": "running",
        "ready": False,
        "ads": [],
        "images": {}
    }

    t = threading.Thread(target=run_generation, args=(job_id, product_name, product_description, size), daemon=True)
    t.start()

    return jsonify({"job_id": job_id}), 200


@app.get("/api/jobs/<job_id>")
def job_status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    return jsonify({
        "status": job.get("status"),
        "ready": bool(job.get("ready")),
        "ads": job.get("ads", [])
    }), 200


@app.get("/api/jobs/<job_id>/ads/<int:index>/image")
def get_image(job_id, index):
    job = JOBS.get(job_id)
    if not job or job.get("status") != "done":
        return jsonify({"error": "Not ready"}), 400

    img_bytes = job.get("images", {}).get(index)
    if not img_bytes:
        return jsonify({"error": "Not found"}), 404

    return send_file(
        BytesIO(img_bytes),
        mimetype="image/jpeg",
        download_name=f"ad_{index}.jpg"
    )


@app.get("/api/jobs/<job_id>/ads/<int:index>/zip")
def download_zip(job_id, index):
    job = JOBS.get(job_id)
    if not job or job.get("status") != "done":
        return jsonify({"error": "Not ready"}), 400

    img_bytes = job.get("images", {}).get(index)
    if not img_bytes:
        return jsonify({"error": "Not found"}), 404

    text = ""
    for ad in job.get("ads", []):
        if int(ad.get("index", -1)) == index:
            text = ad.get("text", "")
            break

    zip_buffer = BytesIO()
    with ZipFile(zip_buffer, "w") as z:
        z.writestr(f"ad_{index}.jpg", img_bytes)
        z.writestr(f"ad_{index}.txt", text)

    zip_buffer.seek(0)
    return send_file(
        zip_buffer,
        as_attachment=True,
        download_name=f"ad_{index}.zip",
        mimetype="application/zip"
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
