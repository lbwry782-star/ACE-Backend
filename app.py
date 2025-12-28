import base64
import io
import os
import re
import tempfile
import zipfile
from datetime import datetime
from typing import Any, Dict, Tuple

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

# OpenAI SDK (v1.x)
try:
    from openai import OpenAI
except Exception as e:  # pragma: no cover
    OpenAI = None  # type: ignore


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


OPENAI_API_KEY = env("OPENAI_API_KEY")
OPENAI_IMAGE_MODEL = env("OPENAI_IMAGE_MODEL", "gpt-image-1")
OPENAI_TEXT_MODEL = env("OPENAI_TEXT_MODEL", "gpt-4.1-mini")
FRONTEND_URL = env("FRONTEND_URL")  # used for CORS
BACKEND_URL = env("BACKEND_URL")    # optional
PORT = int(env("PORT", "10000"))

app = Flask(__name__)

# CORS — allow the deployed frontend origin (GitHub Pages / custom domain)
cors_origins = []
if FRONTEND_URL:
    cors_origins.append(FRONTEND_URL.rstrip("/"))
# allow localhost for dev
cors_origins.extend(["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173", "http://127.0.0.1:5173"])
CORS(app, resources={r"/*": {"origins": cors_origins or "*"}})


def _require_fields(payload: Dict[str, Any]) -> Tuple[bool, str]:
    if not payload.get("product"):
        return False, "missing_product"
    if not payload.get("description"):
        return False, "missing_description"
    size = (payload.get("size") or "").lower()
    if size not in {"1024x1024", "1024x1536", "1536x1024"}:
        return False, "invalid_size"
    ad_index = int(payload.get("ad_index") or 1)
    if ad_index not in {1, 2, 3}:
        return False, "invalid_ad_index"
    return True, ""


def _client() -> "OpenAI":
    if OpenAI is None:
        raise RuntimeError("OpenAI SDK not installed")
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing")
    return OpenAI(api_key=OPENAI_API_KEY)


ENGINE_SYSTEM = """You are ACE ENGINE. You MUST follow rules H01-H08 exactly (no shortcuts).
Key non-negotiables:
- Use ONLY visual/shape similarity, never semantic similarity.
- Prefer FULL OVERLAP HYBRID first; if impossible then SIDE BY SIDE with similar silhouettes.
- If similarity is not instantly obvious to an average human eye, REJECT the pair.
- Objects must be physical everyday objects (not abstract). No logos/labels/text unless integral to the object (e.g., playing card ranks as part of the card; dice pips; compass cardinal letters only if engraved/part of object).
- Title (headline) is inside the image, on the background area (not beside the objects), 3-7 words, MUST include the product name, original (not a quote/variation of description), prominent and about same visual weight as the objects.
- Image must be photorealistic (no illustration/3D/vector/AI effect).
- Background must be the classic background of object C (the chosen projection of A), and it must dominate texture/lighting.
- Marketing text is exactly 50 words and EXCLUDES the headline.
Return JSON only when asked.
"""


def build_engine_plan(product: str, description: str, ad_index: int) -> Dict[str, Any]:
    """Use text model to:
    - infer audience profile (H01)
    - create a distinct ad goal for this ad (H02)
    - generate 80 concrete physical objects tied to goal (H04)
    - pick A and B from the list and decide HYBRID vs SIDE_BY_SIDE based on silhouette similarity (H04)
    - define projections C (for A) and D (for B) and classic background for C (H05)
    - produce a headline 3-7 words incl product (H06)
    - produce a 50-word marketing text (H08)
    """
    client = _client()

    prompt = f"""INPUT:
Product name: {product}
Product description: {description}
Ad number: {ad_index} (out of 3)

TASK:
1) Infer a target audience profile ONLY from the product name+description (H01): age range, lifestyle, needs/pain points, knowledge level.
2) Define a distinct advertising goal for THIS ad (H02). Make it clearly different from the other two likely goals.
3) Produce a list of 80 physical, concrete, everyday objects associated with the goal (H04). No abstract concepts. No logos/labels/text.
4) Choose A (central meaning object) and B (emphasis object) from the list (H04).
5) Choose one projection (silhouette) for A and one projection for B that:
   - First try to find silhouettes that can be perfectly matched for FULL OVERLAP HYBRID.
   - If no obvious match exists, choose SIDE_BY_SIDE with clearly similar silhouettes (obvious to average human eye).
   - If similarity is not obvious, reject and pick a new pair. Try multiple attempts up to 12. If repeated failures, expand list size by +10 (policy) and continue.
6) Output final decision:
   - layout: "HYBRID" or "SIDE_BY_SIDE"
   - A_object, B_object
   - A_projection_description (C) and B_projection_description (D)
   - classic_background_for_C (must be specific, not generic like "nature")
7) Create the headline: 3-7 words, includes product name, original, prominent.
8) Create marketing_text: EXACTLY 50 words, not including the headline.

OUTPUT JSON ONLY with keys:
audience, goal, objects, A_object, B_object, layout, A_projection, B_projection, classic_background, headline, marketing_text
"""

    resp = client.chat.completions.create(
        model=OPENAI_TEXT_MODEL,
        messages=[
            {"role": "system", "content": ENGINE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        response_format={"type": "json_object"},
    )
    data = resp.choices[0].message.content
    return _safe_json_loads(data)


def _safe_json_loads(s: str) -> Dict[str, Any]:
    try:
        import json
        return json.loads(s)
    except Exception:
        # very defensive: try to extract first JSON object
        m = re.search(r"\{.*\}", s, flags=re.S)
        if not m:
            raise
        import json
        return json.loads(m.group(0))


def build_image_prompt(plan: Dict[str, Any], size: str) -> str:
    product = plan.get("product_name", "")
    headline = plan["headline"]
    layout = plan["layout"]
    A = plan["A_object"]
    B = plan["B_object"]
    Aproj = plan["A_projection"]
    Bproj = plan["B_projection"]
    bg = plan["classic_background"]

    # Keep prompt strict: photorealistic, no extra text, headline only, blackish clean ad style
    return f"""Create a single vertical/horizontal photorealistic advertising image (size {size}).
Scene rules (MUST follow):
- Two main physical objects: A = {A}, B = {B}.
- Use projections/silhouettes:
  - A projection (C): {Aproj}
  - B projection (D): {Bproj}
- Composition layout: {layout}. If HYBRID: full overlap where D is perfectly embedded into C while keeping realistic photo lighting and perspective. If SIDE_BY_SIDE: place A and B close, clearly similar silhouettes.
- Background: {bg}. It must dominate the composition's texture and lighting, and match A's classic background. Do NOT use generic "nature".
- Style: real photography, sharp, realistic materials, no illustration, no 3D render, no vector, no AI artifacts.
- Text: ONLY one headline inside the image on the background area (not beside the objects), prominent and large, 3–7 words, MUST include the product name: "{headline}".
- No other text, no logos, no labels.
"""


def generate_image_bytes(prompt: str, size: str) -> bytes:
    client = _client()
    # OpenAI Images API in SDK: images.generate returns base64
    img = client.images.generate(
        model=OPENAI_IMAGE_MODEL,
        prompt=prompt,
        size=size,
    )
    b64 = img.data[0].b64_json
    return base64.b64decode(b64)


def ensure_50_words(text: str) -> str:
    words = re.findall(r"\b\w+[\w'’-]*\b", text.strip())
    if len(words) == 50:
        return text.strip()

    # If off by a little, ask the model to fix precisely to 50 words.
    client = _client()
    fix_prompt = f"""Rewrite the following marketing text to EXACTLY 50 words.
Rules:
- Keep meaning consistent.
- Do NOT include the headline.
- Plain English.
Text:
{text}
"""
    resp = client.chat.completions.create(
        model=OPENAI_TEXT_MODEL,
        messages=[
            {"role": "system", "content": "You output ONLY the corrected 50-word text, no quotes, no extra lines."},
            {"role": "user", "content": fix_prompt},
        ],
        temperature=0.3,
    )
    fixed = resp.choices[0].message.content.strip()
    return fixed


def make_zip(image_bytes: bytes, marketing_text: str, ad_index: int) -> bytes:
    img_name = f"ad_{ad_index}.jpg"
    txt_name = f"ad_{ad_index}.txt"

    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(img_name, image_bytes)
        z.writestr(txt_name, marketing_text)
    mem.seek(0)
    return mem.read()


# In-memory store for generated files (simple and stateless enough for small usage)
# For production you might persist, but this matches the project scope.
FILE_STORE: Dict[str, bytes] = {}


@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.post("/generate")
def generate():
    payload = request.get_json(silent=True) or {}
    ok, err = _require_fields(payload)
    if not ok:
        # Minimal error response (frontend is mostly silent by design)
        return jsonify({"ok": False, "error": err}), 400

    product = payload["product"].strip()
    description = payload["description"].strip()
    size = payload["size"].lower()
    ad_index = int(payload.get("ad_index") or 1)

    # Plan with text model
    plan = build_engine_plan(product, description, ad_index)
    plan["product_name"] = product

    # Ensure exact 50 words
    marketing_text = ensure_50_words(plan.get("marketing_text", ""))

    # Generate image
    img_prompt = build_image_prompt(plan, size)
    image_bytes = generate_image_bytes(img_prompt, size)

    # Build zip and store with a key
    key = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{ad_index}_{abs(hash(product))%100000}"
    zip_bytes = make_zip(image_bytes, marketing_text, ad_index)

    FILE_STORE[f"{key}.zip"] = zip_bytes
    FILE_STORE[f"{key}.jpg"] = image_bytes

    base = BACKEND_URL.rstrip("/") if BACKEND_URL else ""
    image_url = f"{base}/file/{key}.jpg" if base else f"/file/{key}.jpg"
    zip_url = f"{base}/file/{key}.zip" if base else f"/file/{key}.zip"

    return jsonify({
        "ok": True,
        "ad_index": ad_index,
        "image_url": image_url,
        "zip_url": zip_url,
        "marketing_text": marketing_text,
        # Optional debug fields for developers (frontend ignores):
        "layout": plan.get("layout"),
        "A_object": plan.get("A_object"),
        "B_object": plan.get("B_object"),
        "classic_background": plan.get("classic_background"),
        "headline": plan.get("headline"),
    })


@app.get("/file/<path:name>")
def file_get(name: str):
    blob = FILE_STORE.get(name)
    if blob is None:
        return jsonify({"ok": False, "error": "not_found"}), 404

    if name.endswith(".zip"):
        return send_file(
            io.BytesIO(blob),
            mimetype="application/zip",
            as_attachment=True,
            download_name=name,
        )
    if name.endswith(".jpg"):
        return send_file(
            io.BytesIO(blob),
            mimetype="image/jpeg",
            as_attachment=False,
            download_name=name,
        )
    return jsonify({"ok": False, "error": "unsupported"}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
