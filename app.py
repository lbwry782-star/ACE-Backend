import os
import uuid
import time
import threading
from io import BytesIO
from zipfile import ZipFile

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from PIL import Image

app = Flask(__name__)
CORS(app)

OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
OPENAI_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini")

# In-memory job store (simple MVP)
JOBS = {}

ALLOWED_SIZES = {
    "1024x1024": (1024, 1024),
    "1024x1536": (1024, 1536),
    "1536x1024": (1536, 1024),
}

def _make_placeholder_jpg(width: int, height: int) -> bytes:
    # Black background ONLY (no shapes/text on image) – aligns with "no text on image"
    img = Image.new("RGB", (width, height), (0, 0, 0))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=92, optimize=True)
    return buf.getvalue()

@app.get("/health")
def health():
    return "ok", 200


def run_generation(job_id: str, product_name: str, product_description: str, size: str):
    try:
        # Simulate engine runtime (real engine would call OpenAI here)
        time.sleep(4)

        w, h = ALLOWED_SIZES[size]

        # Three distinct ad "purposes" (placeholder)
        purposes = [
            ("Speed", "Get results faster", "A faster path with fewer steps and less friction."),
            ("Confidence", "Feel in control", "Clear guidance that builds trust and reduces uncertainty."),
            ("Clarity", "Know what to do next", "Simple structure that keeps focus and avoids overwhelm."),
        ]

        ads = []
        for i in range(1, 4):
            purpose_name, hook, angle = purposes[i-1]

            # Correct-size image
            JOBS[job_id]["images"][i] = _make_placeholder_jpg(w, h)

            # Distinct 50-word-ish copy per ad
            text = (
                f"{product_name} is made for {purpose_name.lower()}. {hook}. "
                f"{angle} Designed for the right audience, it saves time, reduces effort, "
                f"and supports consistent outcomes. Use this message alongside the visual to "
                f"drive clicks and conversions with a single clear next step."
            )

            ads.append({
                "index": i,
                "purpose": purpose_name,
                "text": text,
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
    size = (data.get("size") or "").strip().lower()

    if not product_name or not product_description or size not in ALLOWED_SIZES:
        return jsonify({"error": "invalid_input"}), 400

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "status": "running",
        "ready": False,
        "ads": [],
        "images": {},
        "size": size,
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
        "size": job.get("size"),
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
