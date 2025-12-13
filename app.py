import os
import uuid
import time
import threading
from io import BytesIO
from zipfile import ZipFile

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
OPENAI_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini")

# In-memory job store (simple MVP)
JOBS = {}

# Valid tiny JPEG bytes embedded as HEX (no base64 -> no padding issues)
_TINY_JPG_HEX = """
ffd8ffe000104a46494600010100000100010000ffdb00430006040506050406060506070706080a100a0a09090a140e0f0c1017141818171416161a
1d251f1a1b231c1616202c20232627292a29191f2d302d283025282928ffdb0043010707070a080a130a0a13281a161a282828282828282828282828
2828282828282828282828282828282828282828282828282828282828282828282828282828ffc00011080001000103012200021101031101ffc400
1f0000010501010101010100000000000000000102030405060708090a0bffc400b5100002010303020403050504040000017d010203000411051221
31410613516107227114328191a1082342b1c11552d1f02433627282090a161718191a25262728292a3435363738393a434445464748494a53545556
5758595a636465666768696a737475767778797a838485868788898a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6
c7c8c9cad2d3d4d5d6d7d8d9dae1e2e3e4e5e6e7e8e9eaf1f2f3f4f5f6f7f8f9faffc4001f0100030101010101010101010000000000000102030405
060708090a0bffc400b51100020102040403040705040400010277000102031104052131061241510761711322328108144291a1b1c109233352f015
6272d10a162434e125f11718191a262728292a35363738393a434445464748494a535455565758595a636465666768696a737475767778797a828384
85868788898a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae2e3e4e5e6e7e8e9ea
f2f3f4f5f6f7f8f9faffda000c03010002110311003f00faa68a28a00fffd9
"""

def _hex_to_bytes(s: str) -> bytes:
    return bytes.fromhex("".join(s.split()))

TINY_JPG_BYTES = _hex_to_bytes(_TINY_JPG_HEX)

@app.get("/health")
def health():
    return "ok", 200


def run_generation(job_id: str, product_name: str, product_description: str, size: str):
    try:
        # Simulate engine runtime (real engine would call OpenAI here)
        time.sleep(8)

        ads = []
        for i in range(1, 4):
            JOBS[job_id]["images"][i] = TINY_JPG_BYTES

            # Placeholder copy for MVP; real engine will generate 50-word copy per ad
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
