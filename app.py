
import os
import uuid
from flask import Flask, jsonify, request

app = Flask(__name__)
jobs = {}

@app.get("/health")
def health():
    return "ok", 200

@app.post("/api/generate")
def generate():
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "job_id": job_id,
        "ads_created": 1,
        "max_ads": 3,
        "ad": {"headline": "מודעה 1", "text": "טקסט שיווקי 1"}
    }
    return jsonify({"job_id": job_id, **jobs[job_id]["ad"], "ads_created": 1, "max_ads": 3})

@app.post("/api/next-ad")
def next_ad():
    data = request.get_json(silent=True) or {}
    job_id = data.get("job_id")
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    if job["ads_created"] >= job["max_ads"]:
        return jsonify({"error": "Limit reached", "ads_created": job["ads_created"], "max_ads": job["max_ads"]}), 409

    job["ads_created"] += 1
    n = job["ads_created"]
    job["ad"] = {"headline": f"מודעה {n}", "text": f"טקסט שיווקי {n}"}
    return jsonify({"job_id": job_id, **job["ad"], "ads_created": job["ads_created"], "max_ads": job["max_ads"]})

@app.get("/api/job/<job_id>")
def job(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({"job_id": job_id, **job["ad"], "ads_created": job["ads_created"], "max_ads": job["max_ads"]})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
