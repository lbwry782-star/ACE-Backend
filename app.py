
import os
import io
import base64
import uuid
from datetime import datetime, timedelta

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from openai import OpenAI
from PIL import Image

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

client = OpenAI()
app = Flask(__name__)

FRONTEND_URL = os.getenv("FRONTEND_URL", "*")
CORS(
    app,
    resources={r"/*": {"origins": FRONTEND_URL}},
    supports_credentials=False,
)

IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini")

SESSION_TTL_HOURS = int(os.getenv("SESSION_TTL_HOURS", "48"))
MAX_ATTEMPTS_PER_SESSION = 1

sessions = {}        # sid -> {attempts_used, expires_at}
requests_store = {}  # request_id -> data
last_request = None  # fallback for download


def _cleanup_sessions():
    now = datetime.utcnow()
    expired = [sid for sid, data in sessions.items() if data["expires_at"] < now]
    for sid in expired:
        sessions.pop(sid, None)


def ensure_session(sid: str):
    _cleanup_sessions()
    if sid not in sessions:
        sessions[sid] = {
            "attempts_used": 0,
            "expires_at": datetime.utcnow() + timedelta(hours=SESSION_TTL_HOURS),
        }
    return sessions[sid]


def has_attempts_left(sid: str) -> bool:
    data = ensure_session(sid)
    return data["attempts_used"] < MAX_ATTEMPTS_PER_SESSION


def consume_attempt(sid: str):
    data = ensure_session(sid)
    data["attempts_used"] += 1


def build_image_prompt(product_text: str) -> str:
    """Image prompt WITHOUT the word 'hybrid' and WITHOUT logos."""
    return (
        "Create a real photographic advertising image for the following product. "
        "Use one clear visual scene that expresses the idea of the product and its benefit. "
        "You may include everyday objects, but do not mention or show any logo, brand mark, "
        "app interface, watermark, or trademark. The visual should feel like a real photo, "
        "with realistic lighting and depth.\n\n"
        f"Product & description: {product_text}\n\n"
        "Embed a short English headline (3–7 words) inside the image only. "
        "Do NOT add any other text, labels, buttons, or logos anywhere in the image."
    )


def build_text_prompt(product_text: str) -> str:
    """Text prompt – no mention of 'hybrid', only copy rules."""
    return (
        "You are an advertising copywriter. Create 3 distinct ad variations for "
        "the following product description. For each variation, produce:\n"
        "1) headline – 3–7 English words\n"
        "2) copy – exactly 50 English words, no bullet points.\n\n"
        f"Product & description: {product_text}\n\n"
        "Do not use brand names, app names, or trademarks. "
        "Write in neutral, clear English only.\n\n"
        "Respond in strict JSON with this structure:\n"
        "{\n"
        "  \"variants\": [\n"
        "    {\"headline\": \"...\", \"copy\": \"...\"},\n"
        "    {\"headline\": \"...\", \"copy\": \"...\"},\n"
        "    {\"headline\": \"...\", \"copy\": \"...\"}\n"
        "  ]\n"
        "}"
    )


def pil_image_from_b64(b64_data: str) -> Image.Image:
    raw = base64.b64decode(b64_data)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def fallback_variants_text():
    base_copy = (
        "This is a 50-word placeholder marketing paragraph for the ACE Engine demo. "
        "It describes the advertised product in clear, persuasive language and "
        "invites the viewer to explore more and take action after seeing this "
        "automatically generated ad. The final system will replace this text."
    )
    return [
        {"headline": "ACE Ad Variant 1", "copy": base_copy},
        {"headline": "ACE Ad Variant 2", "copy": base_copy},
        {"headline": "ACE Ad Variant 3", "copy": base_copy},
    ]


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/start-session", methods=["POST"])
def start_session():
    data = request.get_json(silent=True) or {}
    sid = data.get("sid")
    if not sid:
        return jsonify({"error": "Missing sid"}), 400

    sess = ensure_session(sid)
    attempts_left = max(0, MAX_ATTEMPTS_PER_SESSION - sess["attempts_used"])
    return jsonify({"attempts_left": attempts_left}), 200


@app.route("/generate", methods=["POST"])
def generate():
    global last_request

    body = request.get_json(silent=True) or {}
    product = (body.get("product") or "").strip()
    size = (body.get("size") or "1024x1024").strip()
    sid = body.get("sid")  # optional

    if not product:
        return jsonify({"error": "Missing 'product' in request body"}), 400

    dev_mode = product == "4242"

    # Optional token system: only enforce if sid is provided
    if not dev_mode and sid:
        if not has_attempts_left(sid):
            return jsonify({"error": "No attempts left for this session"}), 403

    # 1) IMAGES
    try:
        image_prompt = build_image_prompt(product)
        image_b64_list = []
        for _ in range(3):
            img_resp = client.images.generate(
                model=IMAGE_MODEL,
                prompt=image_prompt,
                size=size,
                quality="high",
            )
            # New Images API still returns base64 field for compat
            b64 = img_resp.data[0].b64_json
            image_b64_list.append(b64)
    except Exception as e:
        return jsonify({"error": "Image generation failed", "details": str(e)}), 500

    # 2) TEXT (robust)
    try:
        text_prompt = build_text_prompt(product)
        txt_resp = client.responses.create(
            model=TEXT_MODEL,
            input=text_prompt,
        )

        raw_text = ""
        try:
            # New Responses format
            raw_text = txt_resp.output[0].content[0].text
        except Exception:
            try:
                raw_text = txt_resp.choices[0].message["content"]
            except Exception:
                raw_text = ""

        import json
        try:
            parsed = json.loads(raw_text)
            variants_text = parsed.get("variants", [])
        except Exception:
            variants_text = fallback_variants_text()
    except Exception as e:
        print("Text generation failed, using placeholders:", str(e))
        variants_text = fallback_variants_text()

    # Normalize to exactly 3 variants
    while len(variants_text) < 3:
        variants_text.append(fallback_variants_text()[0])
    variants_text = variants_text[:3]

    # 3) Combine
    variants = []
    for i in range(3):
        b64 = image_b64_list[i]
        txt = variants_text[i]
        headline = txt.get("headline", "").strip()
        copy = txt.get("copy", "").strip()

        data_uri = f"data:image/jpeg;base64,{b64}"
        variants.append(
            {
                "image_data": data_uri,
                "headline": headline,
                "copy": copy,
            }
        )

    request_id = str(uuid.uuid4())
    stored = {
        "variants": variants,
        "images_b64": image_b64_list,
    }
    requests_store[request_id] = stored
    last_request = stored  # fallback

    if not dev_mode and sid:
        consume_attempt(sid)

    return jsonify({"request_id": request_id, "variants": variants}), 200


@app.route("/download", methods=["GET"])
def download():
    global last_request

    request_id = request.args.get("request_id")
    index_str = request.args.get("index")

    if not index_str:
        return jsonify({"error": "Missing index"}), 400

    try:
        index = int(index_str) - 1
    except ValueError:
        return jsonify({"error": "Invalid index"}), 400

    stored = None
    if request_id:
        stored = requests_store.get(request_id)

    # Fallback: if request_id is unknown, use last_request (most recent ads)
    if stored is None:
        stored = last_request

    if stored is None:
        return jsonify({"error": "Unknown request_id and no recent generation found"}), 404

    if index < 0 or index >= len(stored["variants"]):
        return jsonify({"error": "Index out of range"}), 400

    variant = stored["variants"][index]
    image_b64 = stored["images_b64"][index]

    img = pil_image_from_b64(image_b64)
    img_buf = io.BytesIO()
    img.save(img_buf, format="JPEG", quality=95)
    img_bytes = img_buf.getvalue()

    text_content = f"{variant.get('headline','').strip()}\n\n{variant.get('copy','').strip()}\n"

    import zipfile

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("ad.jpg", img_bytes)
        zf.writestr("copy.txt", text_content.encode("utf-8"))
    zip_buf.seek(0)

    filename = f"ace_ad_{index+1}.zip"
    return send_file(
        zip_buf,
        mimetype="application/zip",
        as_attachment=True,
        download_name=filename,
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)
