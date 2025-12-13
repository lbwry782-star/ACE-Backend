import os
import uuid
import time
import threading
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from zipfile import ZipFile
from io import BytesIO

app = Flask(__name__)
CORS(app)

OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
OPENAI_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini")

JOBS = {}

@app.get("/health")
def health():
    return "ok", 200

def run_generation(job_id, product_name, product_description, size):
    try:
        time.sleep(8)
        ads = []
        for i in range(1, 4):
            image_bytes = BytesIO()
            image_bytes.write(b"FAKE_JPG_IMAGE_DATA")
            image_bytes.seek(0)

            text_content = (
                f"{product_name} is designed to solve real problems with clarity, "
                f"precision and confidence. This ad focuses on a unique benefit, "
                f"tailored to the audience and crafted visually according to the ACE Engine rules."
            )

            ads.append({
                "image": image_bytes,
                "text": text_content
            })

        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["ads"] = ads

    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(e)

@app.post("/api/generate")
def generate():
    data = request.json
    job_id = str(uuid.uuid4())

    JOBS[job_id] = {
        "status": "running",
        "ads": []
    }

    thread = threading.Thread(
        target=run_generation,
        args=(
            job_id,
            data.get("product_name"),
            data.get("product_description"),
            data.get("size"),
        )
    )
    thread.start()

    return jsonify({"job_id": job_id})

@app.get("/api/jobs/<job_id>")
def job_status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    return jsonify({
        "status": job["status"],
        "ready": job["status"] == "done"
    })

@app.get("/api/jobs/<job_id>/ads/<int:index>/image")
def get_image(job_id, index):
    job = JOBS.get(job_id)
    if not job or job["status"] != "done":
        return jsonify({"error": "Not ready"}), 400

    ad = job["ads"][index - 1]
    return send_file(
        ad["image"],
        mimetype="image/jpeg",
        download_name=f"ad_{index}.jpg"
    )

@app.get("/api/jobs/<job_id>/ads/<int:index>/zip")
def download_zip(job_id, index):
    job = JOBS.get(job_id)
    if not job or job["status"] != "done":
        return jsonify({"error": "Not ready"}), 400

    ad = job["ads"][index - 1]

    zip_buffer = BytesIO()
    with ZipFile(zip_buffer, "w") as zip_file:
        zip_file.writestr(f"ad_{index}.jpg", ad["image"].getvalue())
        zip_file.writestr(f"ad_{index}.txt", ad["text"])

    zip_buffer.seek(0)

    return send_file(
        zip_buffer,
        as_attachment=True,
        download_name=f"ad_{index}.zip",
        mimetype="application/zip"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
