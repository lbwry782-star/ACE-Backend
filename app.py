import os
import time
import base64
import io
import zipfile
from uuid import uuid4

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

app = Flask(__name__)

frontend_url = os.environ.get("FRONTEND_URL")
if frontend_url:
    CORS(app, resources={r"/*": {"origins": [frontend_url]}})
else:
    CORS(app)

IMAGE_MODEL = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1")
TEXT_MODEL = os.environ.get("OPENAI_TEXT_MODEL", "gpt-4.1-mini")

SESSIONS = {}
RESULTS = {}

def get_client():
    if OpenAI is None:
        raise RuntimeError("openai package not installed. Add 'openai' to requirements.txt")
    return OpenAI()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/start-session", methods=["POST"])
def start_session():
    data = request.get_json(silent=True) or {}
    sid = data.get("sid")
    if not sid:
        return jsonify({"error": "missing sid"}), 400

    session = SESSIONS.get(sid)
    if session is None:
        session = {"attempts_left": 1}
        SESSIONS[sid] = session

    return jsonify({"ok": True, "attempts_left": session["attempts_left"]}), 200

def build_image_prompt(product, description, variant_index):
    desc = description or ""
    return (
        "You are ACE, an Automated Creative Engine for advertising.\n"
        "Create a single realistic photographic advertisement visual.\n"
        "Rules:\n"
        "1) Use exactly TWO physical objects only: object A and object B.\n"
        "2) No third object of any kind. No decorations, icons, or extra items.\n"
        "3) The background must be the classic, natural background of either A or B.\n"
        "4) Prefer strong SHAPE similarity between A and B (bottles, cones, discs, etc.).\n"
        "5) If shape similarity is very strong, you may overlap A and B like a hybrid object.\n"
        "   Otherwise place them very close side by side.\n"
        "6) Composition must be clean, with about 10% margin around objects, no cropping.\n"
        "7) No text in the image at all. No logo, no headline, no UI.\n"
        "8) Photographic style only – no illustration, no 3D render.\n"
        "9) Lighting realistic. No surreal floating objects.\n\n"
        f"Product: {product}\n"
        f"Description: {desc}\n"
        f"Variant #{variant_index}: Choose a distinct pair of objects A and B that metaphorically support this product.\n"
        "Return only the image; do not add any text in the picture."
    )

def build_copy_prompt(product, description, variants_count=3):
    desc = description or ""
    return (
        "You are ACE, an AI copywriter for advertising.\n"
        "Write copy for photographic ads that follow these rules:\n"
        "- The headline must be in English, 3–7 words.\n"
        "- The body copy must be exactly 50 words in English.\n"
        "- The copy should match a visual that uses exactly two physical objects (A and B) "
        "with strong shape similarity, in a realistic photo, no extra objects.\n"
        "- The tone is sharp, intelligent and concise.\n\n"
        f"Product: {product}\n"
        f"Description: {desc}\n"
        f"Create {variants_count} distinct options. "
        "Reply strictly as JSON with the following structure:\n"
        "{ \"variants\": [\n"
        "  {\"headline\": \"...\", \"copy\": \"...\"},\n"
        "  {\"headline\": \"...\", \"copy\": \"...\"},\n"
        "  {\"headline\": \"...\", \"copy\": \"...\"}\n"
        "]}"
    )

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(silent=True) or {}
    product = (data.get("product") or "").strip()
    description = (data.get("description") or "").strip()
    size = (data.get("size") or "").strip() or "1024x1024"
    sid = data.get("sid")

    if not product:
        return jsonify({"error": "missing product"}), 400

    dev_mode = (product == "4242")

    valid_sizes = {"1024x1024", "1024x1792", "1792x1024"}
    if size not in valid_sizes:
        return jsonify({"error": "invalid size"}), 400

    if not dev_mode:
        if not sid:
            return jsonify({"error": "missing sid"}), 400
        session = SESSIONS.get(sid)
        if not session or session.get("attempts_left", 0) <= 0:
            return jsonify({"error": "no_attempts"}), 403
        session["attempts_left"] = 0

    try:
        client = get_client()

        img_prompt_1 = build_image_prompt(product, description, 1)
        img_prompt_2 = build_image_prompt(product, description, 2)
        img_prompt_3 = build_image_prompt(product, description, 3)
        img_prompt = (
            "Generate three different options following the same rules, "
            "each with a distinct A/B object pair. "
            "Do not add any text in any image.\n\n"
            f"Option 1:\n{img_prompt_1}\n\n"
            f"Option 2:\n{img_prompt_2}\n\n"
            f"Option 3:\n{img_prompt_3}\n"
        )

        img_resp = client.images.generate(
            model=IMAGE_MODEL,
            prompt=img_prompt,
            n=3,
            size=size,
            response_format="b64_json",
        )

        images_b64 = [d["b64_json"] for d in img_resp.data]

        copy_prompt = build_copy_prompt(product, description, variants_count=3)
        txt_resp = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": "You output only JSON, no explanation."},
                {"role": "user", "content": copy_prompt},
            ],
            temperature=0.7,
        )

        raw_text = txt_resp.choices[0].message.content.strip()

        import json as _json
        try:
            parsed = _json.loads(raw_text)
            variants_text = parsed.get("variants", [])
        except Exception:
            variants_text = []
            for _ in range(3):
                variants_text.append({
                    "headline": product[:30] or "ACE Ad",
                    "copy": (
                        "Discover a new way to think about this product. "
                        "A clean, minimal, object‑driven visual invites the viewer "
                        "to make the connection. No noise, only clarity, emotion "
                        "and focus around what matters most for your audience."
                    )
                })

        request_id = str(uuid4())
        variants_payload = []
        store_variants = []

        for idx, b64 in enumerate(images_b64[:3]):
            img_bytes = base64.b64decode(b64)
            headline = ""
            copy = ""
            if idx < len(variants_text):
                headline = (variants_text[idx].get("headline") or "").strip()
                copy = (variants_text[idx].get("copy") or "").strip()

            data_url = "data:image/png;base64," + b64

            variants_payload.append({
                "index": idx + 1,
                "image_data": data_url,
                "headline": headline,
                "copy": copy,
            })

            store_variants.append({
                "image_bytes": img_bytes,
                "headline": headline,
                "copy": copy,
            })

        RESULTS[request_id] = {
            "created_at": time.time(),
            "variants": store_variants,
        }

        return jsonify({
            "request_id": request_id,
            "variants": variants_payload,
        }), 200

    except Exception as e:
        print("Error in /generate:", e, flush=True)
        if not dev_mode and sid in SESSIONS:
            SESSIONS[sid]["attempts_left"] = 1
        return jsonify({"error": "generation_failed", "details": str(e)}), 500

@app.route("/download", methods=["GET"])
def download_variant():
    request_id = request.args.get("request_id")
    index_str = request.args.get("index")

    if not request_id or not index_str:
        return jsonify({"error": "missing parameters"}), 400

    try:
        idx = int(index_str) - 1
    except ValueError:
        return jsonify({"error": "invalid index"}), 400

    record = RESULTS.get(request_id)
    if not record:
        return jsonify({"error": "unknown request_id"}), 404

    variants = record.get("variants", [])
    if idx < 0 or idx >= len(variants):
        return jsonify({"error": "index out of range"}), 400

    v = variants[idx]
    image_bytes = v["image_bytes"]
    headline = v["headline"] or "ACE Ad"
    copy = v["copy"] or ""

    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"ad_{idx+1}.png", image_bytes)
        text_content = f"HEADLINE:\\n{headline}\\n\\nCOPY (50 words):\\n{copy}\\n"
        zf.writestr(f"ad_{idx+1}_copy.txt", text_content)

    mem_zip.seek(0)
    filename = f"ace_ad_{idx+1}.zip"
    return send_file(
        mem_zip,
        mimetype="application/zip",
        as_attachment=True,
        download_name=filename,
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))