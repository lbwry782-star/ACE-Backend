import os
import time
import uuid
from typing import Any, Dict, Optional

from flask import Flask, request, jsonify
from flask_cors import CORS

# Optional OpenAI usage:
# - If OPENAI_API_KEY is missing OR USE_MOCK=1, the server returns a mock ad.
USE_MOCK = os.getenv("USE_MOCK", "1").strip() == "1"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

app = Flask(__name__)

# CORS: allow any origin for /api during testing (you can lock it down later)
CORS(app, resources={r"/api/*": {"origins": "*"}, r"/health": {"origins": "*"}})

# In-memory session store (simple + free tier friendly)
# session_id -> dict(remaining:int, created_at:float)
SESSIONS: Dict[str, Dict[str, Any]] = {}

MAX_ADS_PER_SESSION = int(os.getenv("MAX_ADS_PER_SESSION", "3"))

def _now() -> float:
    return time.time()

def _new_session() -> str:
    sid = str(uuid.uuid4())
    SESSIONS[sid] = {"remaining": MAX_ADS_PER_SESSION, "created_at": _now()}
    return sid

def _get_session(sid: str) -> Optional[Dict[str, Any]]:
    s = SESSIONS.get(sid)
    if not s:
        return None
    # cleanup: 6 hours
    if _now() - float(s.get("created_at", 0)) > 6 * 3600:
        SESSIONS.pop(sid, None)
        return None
    return s

def _mock_image_url(size: str) -> str:
    seed = uuid.uuid4().hex[:8]
    w, h = 1024, 1024
    try:
        if "x" in size:
            w, h = [int(x) for x in size.lower().split("x", 1)]
    except Exception:
        pass
    return f"https://via.placeholder.com/{w}x{h}.png?text=ACE+AD+{seed}"

def _build_mock_ad(brand: str, product: str, size: str, idx: int) -> Dict[str, Any]:
    angles = [
        ("Clarity", "Know exactly what to do next.", "Get started"),
        ("Confidence", "Feel in control with clear guidance.", "Learn more"),
        ("Speed", "Get results faster with less friction.", "Try it"),
    ]
    angle = angles[(idx - 1) % len(angles)]
    headline = f"{brand}: {angle[0]}"
    copy = f"{product} — {angle[1]} Built for busy people who want results without the noise."
    return {
        "headline": headline,
        "copy": copy,
        "cta": angle[2],
        "size": size,
        "image_url": _mock_image_url(size),
    }

def _openai_generate_ad(brand: str, product: str, size: str, idx: int) -> Dict[str, Any]:
    from openai import OpenAI
    import json, re

    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = (
        f"Create ONE distinct ad concept (variant #{idx}).\n"
        f"Brand: {brand}\n"
        f"Product/Offer: {product}\n"
        f"Return STRICT JSON with keys: headline, copy, cta.\n"
        f"Rules: headline <= 8 words, copy <= 35 words, cta <= 3 words."
    )

    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_TEXT_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": "You are a senior direct-response copywriter."},
            {"role": "user", "content": prompt},
        ],
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.9")),
    )
    content = resp.choices[0].message.content or ""
    m = re.search(r"\{[\s\S]*\}", content)
    if not m:
        raise RuntimeError("OpenAI did not return JSON.")
    data = json.loads(m.group(0))

    headline = str(data.get("headline", "")).strip()
    copy = str(data.get("copy", "")).strip()
    cta = str(data.get("cta", "Learn more")).strip()

    # Images are optional/heavy. Default off: we return a placeholder.
    if os.getenv("OPENAI_IMAGES", "0").strip() != "1":
        return {"headline": headline, "copy": copy, "cta": cta, "size": size, "image_url": _mock_image_url(size)}

    try:
        img_prompt = (
            f"Modern minimal advertising image for: {brand}. Offer: {product}. "
            f"High quality, cinematic lighting, no text."
        )
        image_resp = client.images.generate(
            model=os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1"),
            prompt=img_prompt,
            size=size,
        )
        b64 = image_resp.data[0].b64_json
        return {"headline": headline, "copy": copy, "cta": cta, "size": size, "image_base64": b64}
    except Exception:
        return {"headline": headline, "copy": copy, "cta": cta, "size": size, "image_url": _mock_image_url(size)}

@app.get("/health")
def health():
    return jsonify({"ok": True})

@app.post("/api/generate")
def api_generate():
    payload = request.get_json(silent=True) or {}
    brand = str(payload.get("brand", "ACE")).strip() or "ACE"
    product = str(payload.get("product", "An automated creative engine for an advertising agency")).strip()
    size = str(payload.get("size", "1024x1024")).strip() or "1024x1024"

    sid = _new_session()
    s = SESSIONS[sid]

    try:
        idx = MAX_ADS_PER_SESSION - int(s["remaining"]) + 1  # 1
        ad = _openai_generate_ad(brand, product, size, idx) if (OPENAI_API_KEY and not USE_MOCK) else _build_mock_ad(brand, product, size, idx)
        s["remaining"] = max(0, int(s["remaining"]) - 1)
        return jsonify({"status": "ok", "session_id": sid, "remaining": s["remaining"], "ad": ad})
    except Exception as e:
        msg = str(e)
        retry_after = int(os.getenv("DEFAULT_RETRY_AFTER", "30"))
        if "429" in msg or "Too Many Requests" in msg:
            return jsonify({
                "status": "error",
                "session_id": sid,
                "remaining": s["remaining"],
                "error": "rate_limited",
                "message": "נראה שהמערכת עמוסה כרגע (429). נסה שוב בעוד קצת זמן.",
                "retry_in_seconds": retry_after
            }), 429
        return jsonify({
            "status": "error",
            "session_id": sid,
            "remaining": s["remaining"],
            "error": "generation_failed",
            "message": "נכשל. נסה שוב מאוחר יותר.",
            "details": msg[:300]
        }), 500

@app.post("/api/next")
def api_next():
    payload = request.get_json(silent=True) or {}
    sid = str(payload.get("session_id", "")).strip()
    brand = str(payload.get("brand", "ACE")).strip() or "ACE"
    product = str(payload.get("product", "An automated creative engine for an advertising agency")).strip()
    size = str(payload.get("size", "1024x1024")).strip() or "1024x1024"

    s = _get_session(sid)
    if not s:
        return jsonify({"status": "error", "error": "session_not_found", "message": "Session not found"}), 404

    if int(s["remaining"]) <= 0:
        return jsonify({"status": "ok", "session_id": sid, "remaining": 0, "ad": None})

    try:
        idx = MAX_ADS_PER_SESSION - int(s["remaining"]) + 1
        ad = _openai_generate_ad(brand, product, size, idx) if (OPENAI_API_KEY and not USE_MOCK) else _build_mock_ad(brand, product, size, idx)
        s["remaining"] = max(0, int(s["remaining"]) - 1)
        return jsonify({"status": "ok", "session_id": sid, "remaining": s["remaining"], "ad": ad})
    except Exception as e:
        msg = str(e)
        retry_after = int(os.getenv("DEFAULT_RETRY_AFTER", "30"))
        if "429" in msg or "Too Many Requests" in msg:
            return jsonify({
                "status": "error",
                "session_id": sid,
                "remaining": s["remaining"],
                "error": "rate_limited",
                "message": "נראה שהמערכת עמוסה כרגע (429). נסה שוב בעוד קצת זמן.",
                "retry_in_seconds": retry_after
            }), 429
        return jsonify({
            "status": "error",
            "session_id": sid,
            "remaining": s["remaining"],
            "error": "generation_failed",
            "message": "נכשל. נסה שוב מאוחר יותר.",
            "details": msg[:300]
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")), debug=True)
