
import os
import io
import uuid
import json
import base64
from typing import Dict, Any, List

from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from openai import OpenAI
from PIL import Image

# --------- Basic setup ---------
app = Flask(__name__)

# CORS: for פיתוח נשחרר לכל מקורות; אפשר להקשיח אחר כך לדומיין יחיד
CORS(app, resources={r"/*": {"origins": "*"}})

client = OpenAI()

TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini")
IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")

# נזכור את התוצאות האחרונות בזיכרון עבור DOWNLOAD
_ads_store: Dict[str, List[Dict[str, Any]]] = {}
_last_request_id: str | None = None


# --------- Helpers ---------
def normalize_size(size: str) -> str:
    """
    OpenAI image sizes allowed:
    - 1024x1024
    - 1024x1536 (portrait)
    - 1536x1024 (landscape)
    - auto
    """
    size = (size or "").strip().lower()
    if size == "1024x1792":
        return "1024x1536"
    if size == "1792x1024":
        return "1536x1024"
    allowed = {"1024x1024", "1024x1536", "1536x1024", "auto"}
    if size not in allowed:
        return "1024x1024"
    return size


def ensure_english_text(text: str) -> str:
    # אין כאן בדיקת שפה חזקה – רק טרימינג בסיסי
    return (text or "").strip()


def build_image_prompt(product_text: str) -> str:
    """
    Prompt לתמונה – ללא HYBRID, ללא לוגו, ללא מים/מותגים.
    """
    return f"""
You are an advertising art director.

Create one strong **photographic** advertising image for the following product description:

\"\"\"
{product_text}
\"\"\"

Guidelines:
- Show a single, clear visual scene.
- Use real objects and lighting, no illustrations, no sketches.
- The composition should feel like a finished advertisement.
- Include one short English headline (3–7 words) **inside** the image, in a clean modern font.
- Do NOT include any additional text besides that one headline.
- Do NOT show or suggest any logo, brand mark, watermark, or app interface.
- Do NOT show celebrities, brand names, trademarks, or UI elements.
- The image should work as a social media ad: clean, focused, high contrast.
""".strip()


def build_copy_prompt(product_text: str) -> str:
    """
    Prompt לקופי – 3 וריאציות, כל אחת עם כותרת קצרה וטקסט שיווקי.
    """
    return f"""
You are an advertising copywriter.

Based on the following product description, create copy for **3 different ad variations**:

\"\"\"
{product_text}
\"\"\"

For each variation, write:
- "headline": a short English headline (3–7 words)
- "body": a single marketing paragraph in English, around 40–60 words.

Rules:
- Do NOT mention any brand or logo name.
- Do NOT mention ACE, app, "hybrid", AI model names, or any engine.
- Speak directly to the potential customer and focus on benefits.
- Return your answer as pure JSON with this structure:

{{
  "variants": [
    {{"headline": "...","body": "..."}},
    {{"headline": "...","body": "..."}},
    {{"headline": "...","body": "..."}} 
  ]
}}
""".strip()


def create_jpg_from_base64(b64_data: str) -> bytes:
    img_bytes = base64.b64decode(b64_data)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


# --------- Routes ---------
@app.route("/health", methods=["GET"])
def health() -> Any:
    return jsonify({"status": "ok"})


@app.route("/generate", methods=["POST", "OPTIONS"])
def generate() -> Any:
    global _ads_store, _last_request_id

    if request.method == "OPTIONS":
        # Preflight handled by Flask-CORS, אבל נחזיר תשובה מפורשת
        resp = make_response("", 204)
        return resp

    try:
        data = request.get_json(force=True, silent=True) or {}
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    product_text = ensure_english_text(data.get("product") or data.get("description") or "")
    size = normalize_size(data.get("size", "1024x1024"))

    if not product_text:
        return jsonify({"error": "Missing 'product' description in request body"}), 400

    # 1) TEXT – produce 3 variants of headline+body
    try:
        copy_prompt = build_copy_prompt(product_text)
        copy_resp = client.responses.create(
            model=TEXT_MODEL,
            input=copy_prompt,
            response_format={"type": "json_object"},
        )
        text_content = copy_resp.output[0].content[0].text
        copy_data = json.loads(text_content)
        variants_text = copy_data.get("variants", [])
    except Exception as e:
        return jsonify({
            "error": "Text generation failed",
            "details": str(e),
        }), 500

    # Ensure exactly 3 entries
    while len(variants_text) < 3:
        variants_text.append({"headline": "New Creative Idea", "body": product_text})
    variants_text = variants_text[:3]

    # 2) IMAGES – 3 images using same base prompt
    try:
        image_prompt = build_image_prompt(product_text)
        img_resp = client.images.generate(
            model=IMAGE_MODEL,
            prompt=image_prompt,
            n=3,
            size=size,
        )
        image_datas = [d.b64_json for d in img_resp.data]
    except Exception as e:
        return jsonify({
            "error": "Image generation failed",
            "details": str(e),
        }), 500

    # 3) Build in-memory store for DOWNLOAD
    request_id = str(uuid.uuid4())
    prepared_variants: List[Dict[str, Any]] = []

    for idx in range(3):
        text_variant = variants_text[idx]
        b64_img = image_datas[idx]

        jpg_bytes = create_jpg_from_base64(b64_img)
        # Data URL for immediate display
        data_url = "data:image/jpeg;base64," + base64.b64encode(jpg_bytes).decode("ascii")

        prepared_variants.append({
            "image_bytes": jpg_bytes,
            "image_data": data_url,
            "headline": text_variant.get("headline", "").strip(),
            "copy": text_variant.get("body", "").strip(),
        })

    _ads_store[request_id] = prepared_variants
    _last_request_id = request_id

    # 4) Response to frontend
    response_variants = []
    for v in prepared_variants:
        response_variants.append({
            "image_data": v["image_data"],
            "headline": v["headline"],
            "copy": v["copy"],
        })

    return jsonify({
        "request_id": request_id,
        "variants": response_variants,
    })


@app.route("/download", methods=["GET"])
def download() -> Any:
    global _ads_store, _last_request_id

    req_id = request.args.get("request_id") or ""
    index_str = request.args.get("index") or "1"

    # Fallback ל-request האחרון אם ביקשו request_id לא קיים
    variants = _ads_store.get(req_id)
    if not variants and _last_request_id:
        variants = _ads_store.get(_last_request_id)
    if not variants:
        return jsonify({"error": "No generated ads available for download"}), 400

    try:
        idx = int(index_str) - 1
    except ValueError:
        idx = 0
    if idx < 0 or idx >= len(variants):
        idx = 0

    variant = variants[idx]
    jpg_bytes: bytes = variant["image_bytes"]
    headline: str = variant.get("headline") or ""
    copy_text: str = variant.get("copy") or ""

    # Build ZIP in-memory
    zip_buf = io.BytesIO()
    import zipfile
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("ad.jpg", jpg_bytes)
        # קובץ טקסט – כותרת ושורה ריקה ואז הקופי
        text_content = headline.strip()
        if text_content:
            text_content += "\n\n"
        text_content += copy_text
        zf.writestr("copy.txt", text_content)

    zip_bytes = zip_buf.getvalue()

    resp = make_response(zip_bytes)
    resp.headers["Content-Type"] = "application/zip"
    resp.headers["Content-Disposition"] = f'attachment; filename="ACE_Ad_{idx+1}.zip"'
    return resp


if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
