
import os
import time
import uuid
import threading
import base64
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Any

from flask import Flask, request, jsonify
from flask_cors import CORS

# Optional OpenAI dependency (works in "mock mode" without a key)
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore

APP_VERSION = "FINAL-1.0"

# ---------- Config ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini")
IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")

# If you keep hitting 429, increase these.
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "6"))
BASE_BACKOFF_SECONDS = float(os.getenv("BASE_BACKOFF_SECONDS", "2.0"))
MAX_BACKOFF_SECONDS = float(os.getenv("MAX_BACKOFF_SECONDS", "30.0"))

# Basic server-side throttling to reduce account-wide 429 spikes.
# (Not a payment gate — just safety.)
MIN_SECONDS_BETWEEN_OPENAI_CALLS = float(os.getenv("MIN_SECONDS_BETWEEN_OPENAI_CALLS", "2.5"))

# ---------- App ----------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)

_last_openai_call_lock = threading.Lock()
_last_openai_call_ts = 0.0

def _sleep_until_next_openai_slot():
    global _last_openai_call_ts
    with _last_openai_call_lock:
        now = time.time()
        wait = (_last_openai_call_ts + MIN_SECONDS_BETWEEN_OPENAI_CALLS) - now
        if wait > 0:
            time.sleep(wait)
        _last_openai_call_ts = time.time()

@dataclass
class JobState:
    job_id: str
    status: str  # queued | running | ready | error
    phase: str   # text | image | done | error
    message: str
    ready: bool
    retrying: bool
    retry_in_seconds: int
    requested_size: str
    size: str
    ad: Optional[Dict[str, Any]]
    error: Optional[str]
    created_at: float
    updated_at: float

JOBS: Dict[str, JobState] = {}
JOBS_LOCK = threading.Lock()

def _now():
    return time.time()

def _make_placeholder_png_base64(label: str = "AD") -> str:
    # Create a simple SVG and base64 it as data: image/svg+xml; this avoids pillow dependency.
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024">
      <defs>
        <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="#111827"/>
          <stop offset="100%" stop-color="#0f172a"/>
        </linearGradient>
      </defs>
      <rect width="100%" height="100%" fill="url(#g)"/>
      <rect x="80" y="80" width="864" height="864" rx="60" fill="none" stroke="#94a3b8" stroke-width="6"/>
      <text x="512" y="520" font-family="Arial, Helvetica, sans-serif" font-size="120" fill="#e5e7eb" text-anchor="middle">{label}</text>
      <text x="512" y="610" font-family="Arial, Helvetica, sans-serif" font-size="36" fill="#9ca3af" text-anchor="middle">Mock image (no OpenAI key)</text>
    </svg>"""
    b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    return "data:image/svg+xml;base64," + b64

def _openai_client():
    if not OPENAI_API_KEY or OpenAI is None:
        return None
    return OpenAI(api_key=OPENAI_API_KEY)

def _is_retryable_error(err_text: str) -> bool:
    # Handle 429 and transient 5xx/timeout patterns.
    t = (err_text or "").lower()
    return ("429" in t) or ("too many requests" in t) or ("timeout" in t) or ("temporar" in t) or ("502" in t) or ("503" in t) or ("504" in t)

def _retry_sleep_seconds(attempt: int) -> int:
    # Exponential backoff with jitter
    import random
    s = min(MAX_BACKOFF_SECONDS, BASE_BACKOFF_SECONDS * (2 ** max(0, attempt - 1)))
    s = s * (0.7 + random.random() * 0.6)
    return int(max(1, round(s)))

def _generate_marketing_text(client_name: str, prompt: str) -> Dict[str, str]:
    # Produces headline + primary text. One set only per ad.
    client = _openai_client()
    if client is None:
        return {
            "headline": f"{client_name}: results that feel effortless",
            "primary_text": f"{client_name} helps you reach your goal with clarity and speed. {prompt.strip()[:140]}",
            "cta": "Learn more"
        }

    sys = (
        "You are a senior performance marketer. Create ONE ad text set. "
        "Return STRICT JSON with keys: headline, primary_text, cta. "
        "English only. No emojis. Headline <= 40 chars. Primary text 2-3 sentences."
    )
    user = {
        "client_name": client_name,
        "brief": prompt
    }

    _sleep_until_next_openai_slot()
    r = client.responses.create(
        model=TEXT_MODEL,
        input=[
            {"role": "system", "content": sys},
            {"role": "user", "content": json.dumps(user)}
        ],
        response_format={"type": "json_object"},
        temperature=0.8,
    )
    # openai Responses: output_text contains JSON string
    txt = r.output_text
    data = json.loads(txt)
    # guard keys
    return {
        "headline": str(data.get("headline", "")).strip()[:60],
        "primary_text": str(data.get("primary_text", "")).strip(),
        "cta": str(data.get("cta", "Learn more")).strip()[:30],
    }

def _generate_image_data_url(client_name: str, prompt: str, size: str) -> str:
    client = _openai_client()
    if client is None:
        return _make_placeholder_png_base64("AD")

    image_prompt = (
        f"Create a modern, high-converting ad visual for: {client_name}. "
        f"Concept: {prompt}. "
        f"Style: clean, premium, photoreal or high-end design. "
        f"Do NOT include any text, logos, UI elements, or watermarks."
    )

    _sleep_until_next_openai_slot()
    img = client.images.generate(
        model=IMAGE_MODEL,
        prompt=image_prompt,
        size=size,
    )
    # SDK returns base64 in data[0].b64_json for gpt-image-1
    b64 = img.data[0].b64_json
    return "data:image/png;base64," + b64

def _run_job(job_id: str, client_name: str, prompt: str, size: str):
    def update(**kwargs):
        with JOBS_LOCK:
            st = JOBS.get(job_id)
            if not st:
                return
            for k, v in kwargs.items():
                setattr(st, k, v)
            st.updated_at = _now()

    update(status="running", phase="text", message="Generating ad copy...", ready=False, retrying=False, retry_in_seconds=0, error=None, ad=None)

    # --- TEXT with retries ---
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            txt = _generate_marketing_text(client_name, prompt)
            update(phase="image", message="Generating image...", retrying=False, retry_in_seconds=0)
            last_err = None
            break
        except Exception as e:
            last_err = str(e)
            if _is_retryable_error(last_err) and attempt < MAX_RETRIES:
                wait_s = _retry_sleep_seconds(attempt)
                update(
                    status="error",
                    phase="text",
                    message="Retrying due to temporary issue, please wait...",
                    retrying=True,
                    retry_in_seconds=wait_s,
                    error=last_err,
                    ready=False,
                )
                time.sleep(wait_s)
                update(status="running", phase="text", message="Retrying...", retrying=False, retry_in_seconds=0)
                continue
            update(status="error", phase="error", message="Failed. Please try again later.", retrying=False, retry_in_seconds=0, error=last_err, ready=False)
            return

    if last_err is not None:
        # shouldn't happen, but keep safe
        update(status="error", phase="error", message="Failed. Please try again later.", retrying=False, retry_in_seconds=0, error=last_err, ready=False)
        return

    # --- IMAGE with retries ---
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            img_url = _generate_image_data_url(client_name, prompt, size)
            ad = {
                "headline": txt["headline"],
                "primary_text": txt["primary_text"],
                "cta": txt.get("cta", "Learn more"),
                "image_data_url": img_url,
                "size": size,
            }
            update(status="ready", phase="done", message="Ready", ready=True, retrying=False, retry_in_seconds=0, error=None, ad=ad)
            return
        except Exception as e:
            last_err = str(e)
            if _is_retryable_error(last_err) and attempt < MAX_RETRIES:
                wait_s = _retry_sleep_seconds(attempt)
                update(
                    status="error",
                    phase="image",
                    message="Retrying due to temporary issue, please wait...",
                    retrying=True,
                    retry_in_seconds=wait_s,
                    error=last_err,
                    ready=False,
                )
                time.sleep(wait_s)
                update(status="running", phase="image", message="Retrying...", retrying=False, retry_in_seconds=0)
                continue
            update(status="error", phase="error", message="Failed. Please try again later.", retrying=False, retry_in_seconds=0, error=last_err, ready=False)
            return

@app.get("/health")
def health():
    return jsonify({"ok": True, "version": APP_VERSION})

@app.options("/api/generate")
def generate_options():
    return ("", 204)

@app.post("/api/generate")
def generate():
    data = request.get_json(silent=True) or {}
    client_name = str(data.get("client_name", "")).strip() or "Client"
    prompt = str(data.get("prompt", "")).strip() or "Create a compelling ad."
    size = str(data.get("size", "1024x1024")).strip() or "1024x1024"

    # Normalize size to allowed values (gpt-image-1 supports common sizes)
    allowed = {"1024x1024", "1024x1536", "1536x1024"}
    if size not in allowed:
        size = "1024x1024"

    job_id = str(uuid.uuid4())
    st = JobState(
        job_id=job_id,
        status="queued",
        phase="text",
        message="Queued",
        ready=False,
        retrying=False,
        retry_in_seconds=0,
        requested_size=str(data.get("size", size)),
        size=size,
        ad=None,
        error=None,
        created_at=_now(),
        updated_at=_now(),
    )
    with JOBS_LOCK:
        JOBS[job_id] = st

    t = threading.Thread(target=_run_job, args=(job_id, client_name, prompt, size), daemon=True)
    t.start()
    return jsonify({"job_id": job_id})

@app.get("/api/job/<job_id>")
def job(job_id: str):
    with JOBS_LOCK:
        st = JOBS.get(job_id)
        if not st:
            return jsonify({"error": "Job not found"}), 404
        return jsonify(asdict(st))

if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
