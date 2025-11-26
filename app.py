
import os
import io
import json
import zipfile
from datetime import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import base64

app = Flask(__name__)

# Allow all origins so GitHub Pages (and others) can call the API without CORS errors
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini")
OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment.")

client = OpenAI(api_key=OPENAI_API_KEY)

SIZE_MAP = {
    "1024 x 1024 – Square": "1024x1024",
    "1024 x 1536 – Portrait": "1024x1536",
    "1536 x 1024 – Landscape": "1536x1024",
}


def build_prompt(product_name: str, product_description: str) -> str:
    desc = product_description.strip() or "No additional description provided."
    return (
        "You are the text engine for the ACE Advertising Engine. "
        "You receive a product name and an optional description. "
        "Generate three distinct ad concepts for the same product. "
        "For each concept, return a JSON object with:\n"
        " - headline: a short, punchy English headline (3–7 words)\n"
        " - copy: exactly 50 English words of marketing text describing the ad.\n\n"
        f"Product name: {product_name}\n"
        f"Product description: {desc}\n\n"
        "Return a JSON array with exactly 3 objects, no prose, no backticks."
    )


def parse_concepts(raw_text: str):
    try:
        data = json.loads(raw_text)
        if isinstance(data, list) and len(data) == 3:
            out = []
            for item in data:
                out.append(
                    {
                        "headline": str(item.get("headline", "")).strip() or "ACE Ad",
                        "copy": str(item.get("copy", "")).strip(),
                    }
                )
            return out
    except Exception:
        pass

    return [
        {"headline": "ACE Ad Variation 1", "copy": raw_text.strip()[:350]},
        {"headline": "ACE Ad Variation 2", "copy": raw_text.strip()[:350]},
        {"headline": "ACE Ad Variation 3", "copy": raw_text.strip()[:350]},
    ]


def make_zip_bytes(image_bytes: bytes, headline: str, copy: str) -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("ad.jpg", image_bytes)
        txt = f"{headline}\n\n{copy}\n"
        zf.writestr("copy.txt", txt)
    mem.seek(0)
    return mem.read()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "time": datetime.utcnow().isoformat() + "Z"})


@app.route("/generate_ads", methods=["POST", "OPTIONS"])
def generate_ads():
    if request.method == "OPTIONS":
        return ("", 200)

    try:
        data = request.get_json(force=True, silent=False)
    except Exception:
        return jsonify({"success": False, "error": "Invalid JSON in request body."}), 400

    product_name = (data.get("productName") or "").strip()
    product_description = (data.get("productDescription") or "").strip()
    ad_size_label = data.get("adSize") or "1024 x 1024 – Square"

    if not product_name:
        return jsonify({"success": False, "error": "Missing 'productName' in request body."}), 400

    size = SIZE_MAP.get(ad_size_label, "1024x1024")

    system_prompt = (
        "You are an advertising expert. Always respond with valid JSON ONLY, no explanations."
    )
    user_prompt = build_prompt(product_name, product_description)

    text_resp = client.responses.create(
        model=OPENAI_TEXT_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_output_tokens=800,
    )

    # newer client helper
    raw_text = text_resp.output[0].content[0].text

    concepts = parse_concepts(raw_text)

    variations_payload = []
    for idx, concept in enumerate(concepts, start=1):
        headline = concept["headline"]
        copy = concept["copy"]

        img_prompt = (
            f"Photographic advertising image as a hybrid object representing: {product_name}. "
            f"Headline to overlay in the composition (but DO NOT rasterize the text): '{headline}'. "
            "High detail, cinematic lighting, black background, strong focal hybrid object."
        )
        img_resp = client.images.generate(
            model=OPENAI_IMAGE_MODEL,
            prompt=img_prompt,
            size=size,
            n=1,
            response_format="b64_json",
        )
        b64 = img_resp.data[0].b64_json
        image_bytes = base64.b64decode(b64)

        zip_bytes = make_zip_bytes(image_bytes, headline, copy)
        zip_b64 = base64.b64encode(zip_bytes).decode("utf-8")

        variations_payload.append(
            {
                "headline": headline,
                "copy": copy,
                "image_b64": f"data:image/jpeg;base64,{b64}",
                "zip_b64": zip_b64,
                "zip_filename": f"ace_ad_{idx}.zip",
            }
        )

    return jsonify(
        {
            "success": True,
            "size": size,
            "variations": variations_payload,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
