
import os
import base64
from io import BytesIO

from flask import Flask, request, jsonify
from flask_cors import CORS
import openai

# --- OpenAI setup ---
openai.api_key = os.getenv("OPENAI_API_KEY")

IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini")

# --- Flask app ---
app = Flask(__name__)

FRONTEND_URL = os.getenv("FRONTEND_URL", "*")

# CORS: allow only the configured frontend in production,
# but fall back to * if not set so it will still work in tests.
if FRONTEND_URL == "*" or FRONTEND_URL == "":
    CORS(app)
else:
    CORS(app, resources={r"/*": {"origins": FRONTEND_URL}})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


def clamp_image_size(requested_size):
    """Return a valid OpenAI size string.

    If the requested_size is not one of the official sizes,
    we fall back to 1024x1024 so the call will not fail.
    """
    valid_sizes = {"1024x1024", "1792x1024", "1024x1792"}
    if requested_size in valid_sizes:
        return requested_size
    return "1024x1024"


def build_system_prompt():
    return (
        "You are an ad copywriter. The user will give you a product name, an optional "
        "short description, and an image size. You must return exactly three different "
        "ad concepts for this product. Each concept must have a short English headline "
        "(3–7 words) and a short marketing text (about 50 English words). "
        "Do not mention AI, do not mention that the ad was generated, and do not use brand names."
    )


def parse_text_response(text):
    """Parse a simple JSON or bullet-style response into a list of 3 ads.

    To keep things robust, we try JSON first. If that fails, we fall back to a very
    simple line-based split.
    """
    import json

    # Try JSON first
    try:
        data = json.loads(text)
        ads = []
        for item in data:
            headline = item.get("headline", "").strip()
            marketing_text = item.get("text", "").strip() or item.get("copy", "").strip()
            if headline or marketing_text:
                ads.append({
                    "headline": headline,
                    "marketing_text": marketing_text,
                })
        if len(ads) >= 3:
            return ads[:3]
    except Exception:
        pass

    # Fallback: split by lines and group roughly into 3 chunks
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    chunks = [[]]
    for line in lines:
        if line.lower().startswith(("1.", "2.", "3.")) and len(chunks[-1]) > 0:
            chunks.append([line])
        else:
            chunks[-1].append(line)
    if len(chunks) < 3:
        while len(chunks) < 3:
            chunks.append([])

    ads = []
    for chunk in chunks[:3]:
        if not chunk:
            ads.append({"headline": "", "marketing_text": ""})
            continue
        headline = chunk[0]
        marketing_text = " ".join(chunk[1:]) if len(chunk) > 1 else ""
        ads.append({"headline": headline, "marketing_text": marketing_text})
    return ads


def generate_copy_with_openai(product, description, size):
    """Ask OpenAI for three ad concepts (headline + text)."""
    prompt = (
        f"Product name: {product}\n"
        f"Description: {description or '(none)'}\n"
        f"Image size: {size}\n\n"
        "Return the answer as a JSON array of 3 objects, each with keys "
        "'headline' (short English headline, 3-7 words) and 'text' (about 50 English words)."
    )

    try:
        # Using the classic ChatCompletion API so it works with openai==0.28
        response = openai.ChatCompletion.create(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": prompt},
            ],
            temperature=0.9,
        )
        text = response["choices"][0]["message"]["content"]
    except Exception as e:
        # If text generation fails, fall back to a very simple template
        text = (
            f"[FALLBACK]\n"
            f"1. {product} – Powerful results\n"
            f"Discover how {product} can help your clients. "
            f"Clear benefits, simple communication, and a focused promise.\n"
            f"2. {product} – Smarter choice\n"
            f"A short, confident message that highlights the main benefit.\n"
            f"3. {product} – Stand out\n"
            f"Simple, memorable, and easy to use in any ad format."
        )

    return parse_text_response(text)


def generate_image_b64(product, description, size_str, variation_index):
    """Generate a single image and return base64 string."""
    prompt = (
        f"Create a photographic advertisement image for the product: {product}. "
        f"The ad is for social media. Avoid logos and real brand names. "
        f"Focus on a clear, striking composition with a hybrid of two real-world objects "
        f"that visually express the product benefit. "
    )
    if description:
        prompt += f" Extra product context: {description}. "

    size_for_openai = clamp_image_size(size_str)

    try:
        result = openai.Image.create(
            model=IMAGE_MODEL,
            prompt=prompt,
            n=1,
            size=size_for_openai,
            response_format="b64_json",
        )
        b64 = result["data"][0]["b64_json"]
        return b64
    except Exception as e:
        # If image generation fails, return an empty string; the frontend
        # will still show the headline and text.
        return ""


@app.route("/generate", methods=["POST"])
def generate():
    """Main generation endpoint: returns 3 ads (image + headline + text)."""
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    product = (data.get("product") or "").strip()
    description = (data.get("description") or "").strip()
    size = (data.get("size") or "1024x1024").strip()

    if not product:
        return jsonify({"error": "Missing 'product' field"}), 400

    # 1) Create copy for three ads
    ads_copy = generate_copy_with_openai(product, description, size)

    # 2) For each ad, generate an image
    ads_payload = []
    for idx, copy_obj in enumerate(ads_copy, start=1):
        b64_img = generate_image_b64(product, description, size, idx)
        ads_payload.append({
            "headline": copy_obj.get("headline", ""),
            "marketing_text": copy_obj.get("marketing_text", ""),
            "image_base64": b64_img,
            "size": clamp_image_size(size),
        })

    return jsonify({"ads": ads_payload}), 200


if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
