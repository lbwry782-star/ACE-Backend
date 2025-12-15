
from flask import Flask, jsonify, request
import uuid

app = Flask(__name__)
jobs = {}

@app.post("/api/generate")
def generate():
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"ads_created": 1, "max_ads": 3, "headline": "מודעה 1", "text": "טקסט שיווקי 1"}
    return jsonify({"job_id": job_id, "headline": jobs[job_id]["headline"], "text": jobs[job_id]["text"]})

@app.post("/api/next-ad")
def next_ad():
    job_id = request.json.get("job_id")
    job = jobs.get(job_id)
    if not job or job["ads_created"] >= job["max_ads"]:
        return jsonify({"error": "limit reached"}), 400
    job["ads_created"] += 1
    job["headline"] = f"מודעה {job['ads_created']}"
    job["text"] = f"טקסט שיווקי {job['ads_created']}"
    return jsonify(job)

@app.get("/api/job/<job_id>")
def job(job_id):
    return jsonify(jobs.get(job_id, {}))

if __name__ == "__main__":
    app.run()
