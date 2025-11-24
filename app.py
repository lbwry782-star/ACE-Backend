import base64
import io
import json
import os
import uuid
from datetime import datetime
from zipfile import ZipFile
from io import BytesIO

from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from openai import OpenAI
from PIL import Image

app = Flask(__name__)

# Allow CORS from your frontend
frontend_url = os.environ.get("FRONTEND_URL")
if frontend_url:
    CORS(app, origins=[frontend_url], supports_credentials=True)
else:
    CORS(app, origins="*", supports_credentials=True)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

TEXT_MODEL = os.environ.get("OPENAI_TEXT_MODEL", "gpt-4.1-mini")
IMAGE_MODEL = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1")

# In-memory store for generated ZIPs (per session token)
GENERATED_SESSIONS = {}


def parse_size(size_str: str):
    try:
        w, h = size_str.lower().split("x")
        return int(w), int(h)
    except Exception:
        return 1024, 1024


def choose_openai_image_size(width: int, height: int) -> str:
    """
    If requested size is already a generic OpenAI size, use it as-is.
    Otherwise choose by orientation.
    Allowed:
      - 1024x1024
      - 1536x1024
      - 1024x1536
    """
    if (width, height) in [(1024, 1024), (1536, 1024), (1024, 1536)]:
        return f"{width}x{height}"
    if width == height:
        return "1024x1024"
    if width > height:
        return "1536x1024"
    return "1024x1536"


def normalize_copy_to_50_words(text: str) -> str:
    words = text.strip().split()
    if len(words) > 50:
        words = words[:50]
    elif len(words) < 50:
        words += ["more"] * (50 - len(words))
    return " ".join(words)


def extract_json_from_text(raw: str):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        return None


def generate_text_variation(product_name: str, description: str, idx: int):
    system_msg = (
        "You are an advertising copywriter for the ACE Advertising Engine. "
        "Follow these strict rules:\n"
        "\u2022 English only.\n"
        "\u2022 Headline: 3–7 words, punchy, based on the audience mindset.\n"
        "\u2022 The headline must NOT literally describe the hybrid visual object; "
        "focus on promise, benefit, or idea instead.\n"
        "\u2022 Marketing copy: exactly 50 words, English only, no lists, no bullets.\n"
        "\u2022 Copy must focus on benefits, promise, or solution – not on how the AI works.\n"
        "\u2022 Do not mention AI, models, prompts, or automation.\n"
        "\u2022 Do not mention any trademarks, brands, or celebrities.\n"
    )

    user_msg = (
        "Product information:\n"
        f"- Product name: {product_name}\n"
        f"- Product description: {description or '(no extra description)'}\n\n"
        "Infer the target audience and main advertising goal from this.\n"
        "Then return a SINGLE JSON object ONLY, with this exact structure:\n"
        "{\n"
        '  "headline": "Your 3–7 word headline here",\n'
        '  "copy": "Exactly 50 words of marketing copy here"\n'
        "}\n"
        "No extra commentary, no markdown, no explanations – JSON only."
    )

    completion = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.8,
    )

    content = completion.choices[0].message.content or ""
    data = extract_json_from_text(content)

    if not isinstance(data, dict):
        lines = [l.strip() for l in content.splitlines() if l.strip()]
        headline = lines[0] if lines else "Fresh ideas that sell"
        rest = " ".join(lines[1:]) if len(lines) > 1 else (
            f"Discover {product_name} and see how it transforms everyday moments into something sharper, "
            "smarter, and more effective for your life and work."
        )
        copy_text = normalize_copy_to_50_words(rest)
        return headline, copy_text

    headline = data.get("headline") or "Fresh ideas that sell"
    copy_text = data.get("copy") or (
        f"Discover {product_name} and see how it transforms everyday moments into something sharper, "
        "smarter, and more effective for your life and work."
    )

    headline = " ".join(headline.strip().split())
    copy_text = normalize_copy_to_50_words(copy_text)
    return headline, copy_text


def build_image_prompt(product_name: str, description: str, headline: str, ad_size: str, idx: int) -> str:
    variation_label = f"Variation {idx + 1}"
    return (
        f"{variation_label}: Photorealistic advertising image for the product '{product_name}'. "
        f"Short context: {description or 'No extra description provided; infer from product name only.'} "
        "Infer the target audience and the main advertising goal from this.\n\n"
        "Create a SINGLE central hybrid object made from exactly two real-world objects that logically symbolize "
        "the advertising goal (first by conceptual association, then by shape similarity). One object should visually "
        "eclipse or overlap the other, like a solar eclipse, forming a clear super-shape.\n\n"
        "Rules:\n"
        "\u2022 Photography style only, no illustration, no cartoons.\n"
        "\u2022 Minimal, clean photographic background – no clutter, no extra props.\n"
        "\u2022 The hybrid object is clearly centered and dominant, respecting safe margins.\n"
        "\u2022 No logos, no brand names, no recognizable celebrities, no UI screenshots.\n"
        "\u2022 Integrate the headline text near the bottom area: "
        f"'{headline}'. Use simple, clean typography.\n"
        "\u2022 Do NOT include any additional body copy, paragraphs, or long text beyond this headline.\n"
        "\u2022 Overall look: polished, modern advertising visual suitable for social media.\n"
        f"Requested frame aspect based on ad size: {ad_size}. Keep the composition usable in that framing."
    )


def generate_image_pil(product_name: str, description: str, headline: str, ad_size: str, idx: int) -> Image.Image:
    target_w, target_h = parse_size(ad_size)
    api_size = choose_openai_image_size(target_w, target_h)

    prompt = build_image_prompt(product_name, description, headline, ad_size, idx)

    result = client.images.generate(
        model=IMAGE_MODEL,
        prompt=prompt,
        size=api_size,
        output_format="jpeg",
    )

    image_base64 = result.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)

    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    if (img.width, img.height) != (target_w, target_h):
        img = img.resize((target_w, target_h), Image.LANCZOS)
    return img


def image_to_jpeg_bytes(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=92, optimize=True)
    buf.seek(0)
    return buf.getvalue()


def make_thumbnail_bytes(img: Image.Image, max_width: int = 420) -> bytes:
    w, h = img.size
    if w > max_width:
        scale = max_width / float(w)
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85, optimize=True)
    buf.seek(0)
    return buf.getvalue()


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "time": datetime.utcnow().isoformat() + "Z",
            "text_model": TEXT_MODEL,
            "image_model": IMAGE_MODEL,
        }
    )


@app.route("/generate", methods=["POST"])
def generate():
    """
    Generates 3 variations and returns JSON with thumbnails + copy and a session token
    for per-variation ZIP download.
    """
    data = request.get_json(force=True, silent=True) or {}

    product_name = (data.get("product_name") or "").strip()
    description = (data.get("product_description") or "").strip()
    ad_size = (data.get("ad_size") or "1024x1024").strip()

    if not product_name:
        return jsonify({"error": "product_name is required"}), 400

    words = [w for w in product_name.split() if w.strip()]
    if len(words) > 15:
        product_name = " ".join(words[:15])

    token = uuid.uuid4().hex
    zips_for_token = []
    variations_payload = []

    for idx in range(3):
        headline, copy_text = generate_text_variation(product_name, description, idx)
        img = generate_image_pil(product_name, description, headline, ad_size, idx)

        full_jpeg = image_to_jpeg_bytes(img)
        thumb_jpeg = make_thumbnail_bytes(img)

        # Build per-variation ZIP
        mem_zip = BytesIO()
        with ZipFile(mem_zip, mode="w") as zf:
            zf.writestr(f"ad_{idx+1}.jpg", full_jpeg)
            zf.writestr(f"ad_{idx+1}_copy.txt", copy_text)
        mem_zip.seek(0)
        zips_for_token.append(mem_zip.read())

        thumb_b64 = base64.b64encode(thumb_jpeg).decode("ascii")
        thumb_data_url = f"data:image/jpeg;base64,{thumb_b64}"

        variations_payload.append(
            {
                "index": idx + 1,
                "headline": headline,
                "copy": copy_text,
                "thumb": thumb_data_url,
            }
        )

    GENERATED_SESSIONS[token] = zips_for_token

    return jsonify({"token": token, "variations": variations_payload})


@app.route("/download/<token>/<int:index>", methods=["GET"])
def download_variation(token: str, index: int):
    """
    Downloads a single variation ZIP by token and index (1-based).
    """
    if token not in GENERATED_SESSIONS:
        return jsonify({"error": "invalid or expired token"}), 404

    zips_for_token = GENERATED_SESSIONS[token]
    if index < 1 or index > len(zips_for_token):
        return jsonify({"error": "invalid variation index"}), 400

    zip_bytes = zips_for_token[index - 1]
    mem_file = BytesIO(zip_bytes)
    filename = f"ace_ad_variation_{index}.zip"
    return send_file(
        mem_file,
        mimetype="application/zip",
        as_attachment=True,
        download_name=filename,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
