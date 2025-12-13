import os
import time
import uuid
import json
import base64
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from threading import Thread, Lock

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from openai import OpenAI

OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
OPENAI_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PORT = int(os.getenv("PORT", "10000"))

DATA_DIR = Path(os.getenv("ACE_DATA_DIR", "/tmp/ace_engine_jobs"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_SIZES = {"1024x1024", "1024x1536", "1536x1024"}

app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else OpenAI()

_jobs = {}
_jobs_lock = Lock()

@dataclass
class AdResult:
    index: int
    headline: str
    copy_50_words: str
    image_path: str
    text_path: str
    zip_path: str
    layout: str
    objective: str
    A: str
    B: str

def _job_dir(job_id: str) -> Path:
    d = DATA_DIR / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d

def _write_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)

def _write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def _make_zip(zip_path: Path, jpg_path: Path, txt_path: Path):
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(jpg_path, arcname=jpg_path.name)
        z.write(txt_path, arcname=txt_path.name)

ENGINE_SYSTEM = """ACE ENGINE 🚀
Follow exactly the steps 00H-10H with NO interpretation and NO shortcuts.
Do NOT use external knowledge; derive everything only from product name and product description.
Do NOT use conceptual similarity when selecting objects; obey strict structural/shape rules.
Ad must include only: VISUAL + HEADLINE (headline not on the visual).
Visual must be realistic photo (no illustration, no vector, no 3D/AI look).
Background: beautiful black design, no extra text beyond headline.
No logos/text/letters/numbers on objects unless intrinsically part of the object (e.g., dice dots, playing cards rank/queen/king/ace, engraved compass letters).
"""

AD_SCHEMA = {
  "name": "ace_engine_ad_plan",
  "schema": {
    "type": "object",
    "additionalProperties": False,
    "properties": {
      "target_audience": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
          "age": {"type": "string"},
          "lifestyle": {"type": "string"},
          "needs": {"type": "string"},
          "knowledge_level": {"type": "string"},
          "pains": {"type": "string"}
        },
        "required": ["age","lifestyle","needs","knowledge_level","pains"]
      },
      "objective": {"type": "string"},
      "objects_80": {"type": "array", "items": {"type":"string"}, "minItems": 80, "maxItems": 80},
      "A": {"type":"string"},
      "B": {"type":"string"},
      "projection_angle": {"type":"string"},
      "shape_similarity_assessment": {
        "type":"object",
        "additionalProperties": False,
        "properties": {
          "similarity_level": {"type":"string","enum":["high","medium","none"]},
          "eye_test_pass": {"type":"boolean"},
          "decision": {"type":"string","enum":["HYBRID","SIDE_BY_SIDE","REJECT"]}
        },
        "required": ["similarity_level","eye_test_pass","decision"]
      },
      "headline": {"type":"string"},
      "copy_50_words": {"type":"string"},
      "final_image_prompt": {"type":"string"}
    },
    "required": ["target_audience","objective","objects_80","A","B","projection_angle",
                 "shape_similarity_assessment","headline","copy_50_words","final_image_prompt"]
  }
}

def _enforce_headline_rules(product_name: str, headline: str) -> str:
    words = headline.strip().split()
    if product_name not in headline:
        headline = f"{product_name} {headline}".strip()
        words = headline.split()
    if len(words) < 3:
        headline = (headline + " עכשיו בשבילך").strip()
        words = headline.split()
    if len(words) > 7:
        headline = " ".join(words[:7])
    return headline.strip()

def _trim_to_50_words(text: str) -> str:
    words = [w for w in text.strip().split() if w]
    if len(words) > 50:
        return " ".join(words[:50])
    if len(words) < 50:
        while len(words) < 50:
            words.append("באופן")
            if len(words) < 50:
                words.append("פשוט")
    return " ".join(words[:50])

def _create_ad_plan(product_name: str, product_description: str, size: str, previous_plans: list):
    user_prompt = f"""INPUTS (only source of truth):
- Product name: {product_name}
- Product description: {product_description}

Constraints:
- Produce exactly ONE ad plan.
- Choose an objective DIFFERENT from previous objectives: {[p.get('objective') for p in previous_plans]}.
- Produce exactly 80 REAL physical objects, simple everyday items with clear geometry, associative to the objective.
- Choose A first (central meaning for the objective), then B (ideational emphasis based on the 80 list). A and B are NOT the same natural object.
- No text/logo/letters/numbers on objects unless intrinsically part of the object (allowed: playing cards rank/queen/king/ace, dice dots, engraved compass letters).
- Choose projection angle that maximizes SHAPE similarity between A and B (true projection/silhouette logic).
- Decide HYBRID if shape similarity is high and eye test passes, SIDE_BY_SIDE if medium, REJECT if none.
- If REJECT, still output all fields but set decision=REJECT, similarity_level=none, eye_test_pass=false.
- Visual prompt MUST specify: realistic photo, black background design, no text on image, headline separate, no vector/illustration/3D/AI look, output size {size}.
- Headline: 3-7 words, original, MUST include product name, not on the visual.
- Copy: exactly 50 words, not on image.

Return valid JSON only.
"""

    resp = client.responses.create(
        model=OPENAI_TEXT_MODEL,
        input=[{"role":"system","content":ENGINE_SYSTEM},{"role":"user","content":user_prompt}],
        text={"format":{"type":"json_schema","name":AD_SCHEMA["name"],"schema":AD_SCHEMA["schema"]}},
    )
    plan = json.loads(resp.output_text)
    plan["headline"] = _enforce_headline_rules(product_name, plan["headline"])
    plan["copy_50_words"] = _trim_to_50_words(plan["copy_50_words"])
    return plan

def _generate_image_jpg(prompt: str, size: str) -> bytes:
    img = client.images.generate(
        model=OPENAI_IMAGE_MODEL,
        prompt=prompt,
        n=1,
        size=size,
        output_format="jpeg",
        quality="high",
    )
    return base64.b64decode(img.data[0].b64_json)

def _run_job(job_id: str, product_name: str, product_description: str, size: str):
    try:
        with _jobs_lock:
            _jobs[job_id]["status"] = "running"
            _jobs[job_id]["started_at"] = time.time()

        d = _job_dir(job_id)
        results = []
        plans = []

        for i in range(1, 4):
            plan = _create_ad_plan(product_name, product_description, size, plans)
            decision = plan["shape_similarity_assessment"]["decision"]

            # Failure policy: if rejected repeatedly, reuse a non-rejected pair => duplicates allowed
            if decision == "REJECT":
                retries = 0
                while retries < 2 and decision == "REJECT":
                    retries += 1
                    plan = _create_ad_plan(product_name, product_description, size, plans + [plan])
                    decision = plan["shape_similarity_assessment"]["decision"]
                if decision == "REJECT":
                    for p in reversed(plans):
                        if p["shape_similarity_assessment"]["decision"] != "REJECT":
                            plan = p
                            decision = plan["shape_similarity_assessment"]["decision"]
                            break

            plans.append(plan)

            jpg_bytes = _generate_image_jpg(plan["final_image_prompt"], size)

            jpg_path = d / f"ad_{i}.jpg"
            txt_path = d / f"ad_{i}.txt"
            zip_path = d / f"ad_{i}.zip"

            _write_bytes(jpg_path, jpg_bytes)
            _write_text(txt_path, plan["copy_50_words"])
            _make_zip(zip_path, jpg_path, txt_path)

            results.append(AdResult(
                index=i,
                headline=plan["headline"],
                copy_50_words=plan["copy_50_words"],
                image_path=str(jpg_path),
                text_path=str(txt_path),
                zip_path=str(zip_path),
                layout="HYBRID" if decision == "HYBRID" else "SIDE_BY_SIDE",
                objective=plan["objective"],
                A=plan["A"],
                B=plan["B"],
            ))

        with _jobs_lock:
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["finished_at"] = time.time()
            _jobs[job_id]["ads"] = [asdict(r) for r in results]

    except Exception as e:
        with _jobs_lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = str(e)
            _jobs[job_id]["finished_at"] = time.time()

@app.get("/api/health")
def health():
    return jsonify({"ok": True})

@app.post("/api/generate")
def generate():
    data = request.get_json(force=True, silent=True) or {}
    product_name = (data.get("product_name") or "").strip()
    product_description = (data.get("product_description") or "").strip()
    size = (data.get("size") or "").strip()

    if not product_name or not product_description:
        return jsonify({"error":"Missing product_name or product_description"}), 400
    if size not in ALLOWED_SIZES:
        return jsonify({"error":f"Invalid size. Allowed: {sorted(ALLOWED_SIZES)}"}), 400

    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {"id": job_id, "status":"queued", "created_at": time.time(), "ads": [], "error": None}

    Thread(target=_run_job, args=(job_id, product_name, product_description, size), daemon=True).start()
    return jsonify({"job_id": job_id})

@app.get("/api/jobs/<job_id>")
def job_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return jsonify({"error":"Job not found"}), 404

        safe = {k:v for k,v in job.items() if k != "ads"}
        safe["ads"] = []
        for ad in job.get("ads", []):
            i = ad["index"]
            safe["ads"].append({
                "index": i,
                "headline": ad["headline"],
                "copy_50_words": ad["copy_50_words"],
                "layout": ad["layout"],
                "image_url": f"/api/jobs/{job_id}/ads/{i}/image",
                "zip_url": f"/api/jobs/{job_id}/ads/{i}/zip",
            })
        return jsonify(safe)

@app.get("/api/jobs/<job_id>/ads/<int:ad_index>/image")
def get_ad_image(job_id: str, ad_index: int):
    d = _job_dir(job_id)
    path = d / f"ad_{ad_index}.jpg"
    if not path.exists():
        return jsonify({"error":"Not found"}), 404
    return send_file(path, mimetype="image/jpeg", as_attachment=False, download_name=path.name)

@app.get("/api/jobs/<job_id>/ads/<int:ad_index>/zip")
def get_ad_zip(job_id: str, ad_index: int):
    d = _job_dir(job_id)
    path = d / f"ad_{ad_index}.zip"
    if not path.exists():
        return jsonify({"error":"Not found"}), 404
    return send_file(path, mimetype="application/zip", as_attachment=True, download_name=path.name)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
