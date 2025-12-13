import os
import uuid
import time
import threading
from io import BytesIO
from zipfile import ZipFile

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from PIL import Image, ImageDraw, ImageFont

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

def _make_placeholder_jpg(width: int, height: int, headline: str) -> bytes:
    # Black background, simple photo-like vignette, no text on image per engine rules.
    img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    # subtle frame
    draw.rectangle([20, 20, width-20, height-20], outline=(60, 60, 60), width=3)
    # subtle light gradient block (visual placeholder)
    draw.ellipse([width*0.15, height*0.10, width*0.85, height*0.70], outline=(90, 90, 90), width=4)
    # Export
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

        ads = []
        # Distinct purpose/copy per ad (placeholder logic; real engine determines purposes)
        purposes = [
            "clarity",
            "confidence",
            "speed",
        ]

        for i in range(1, 4):
            headline = f"{product_name} for {purposes[i-1]}"
            # Placeholder image in correct dimensions
            JOBS[job_id]["images"][i] = _make_placeholder_jpg(w, h, headline)

            # 50-word marketing text (approx, no extra UI text)
            text = (
                f"{product_name} is built for people who need results without friction. "
                f"This message focuses on {purposes[i-1]}: a clearer path, fewer steps, and more control. "
                f"It supports the audience’s needs and pain points, stays practical, and drives action. "
                f"Use it on landing pages and social posts alongside the visual."
            )

            ads.append({
                "index": i,
                "headline": headline,
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

    headline = ""
    text = ""
    for ad in job.get("ads", []):
        if int(ad.get("index", -1)) == index:
            headline = ad.get("headline", "")
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
