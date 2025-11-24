
import os
import io
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)

FRONTEND_URL = os.getenv("FRONTEND_URL", "*")
if FRONTEND_URL == "*":
    CORS(app)
else:
    CORS(app, resources={r"/*": {"origins": [FRONTEND_URL]}})

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini")
IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")

# UI sizes (what the frontend sends)
UI_SIZES = {
    "1024x1024": (1024, 1024),
    "1024x1536": (1024, 1536),
    "1536x1024": (1536, 1024),
}

# Mapping from UI size -> OpenAI generic image size
OPENAI_SIZE_MAP = {
    "1024x1024": "1024x1024",
    "1024x1536": "1024x1792",
    "1536x1024": "1792x1024",
}


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


def make_fallback_image(headline: str, size_key: str) -> str:
    """Fallback: create a simple black image with the headline in the center and return as data URL."""
    width, height = UI_SIZES.get(size_key, (1024, 1024))
    img = Image.new("RGB", (width, height), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    font = ImageFont.load_default()
    text = headline or "ACE Ad"
    max_width = int(width * 0.8)

    words = text.split()
    lines = []
    current = ""
    for w in words:
        candidate = (current + " " + w).strip()
        bbox = draw.textbbox((0, 0), candidate, font=font)
        line_width = bbox[2] - bbox[0]
        if line_width <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = w
    if current:
        lines.append(current)

    if not lines:
        lines = ["ACE Ad"]

    line_heights = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_heights.append(bbox[3] - bbox[1] + 4)

    total_text_height = sum(line_heights)
    y = (height - total_text_height) // 2

    for line, lh in zip(lines, line_heights):
        bbox = draw.textbbox((0, 0), line, font=font)
        line_width = bbox[2] - bbox[0]
        x = (width - line_width) // 2
        draw.text((x, y), line, font=font, fill=(255, 255, 255))
        y += lh

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=90)
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


@app.route("/generate_ads", methods=["POST"])
def generate_ads():
    import json as json_lib

    try:
        data = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"success": False, "error": "Invalid JSON"}), 400

    product_name = (data.get("product_name") or "").strip()
    product_description = (data.get("product_description") or "").strip()
    size_key = (data.get("size") or "1024x1024").strip()

    if not product_name:
        return jsonify({"success": False, "error": "Missing product_name"}), 400

    if size_key not in UI_SIZES:
        size_key = "1024x1024"

    openai_size = OPENAI_SIZE_MAP.get(size_key, "1024x1024")

    # ---- 1) TEXT GENERATION ----
    prompt = f"""You are the text engine for the ACE Advertising Engine.

You receive a product name and an optional description.
Your task: generate EXACTLY 3 advertising variations.

For EACH variation you MUST output:
- "headline": a short English headline, 3–7 words, focused on the audience mindset and benefit. Do NOT describe any hybrid object literally.
- "copy": exactly 50 words of English marketing copy. It should focus on benefit, promise, or solution. No bullet lists.

Product name: "{product_name}"
Product description: "{product_description}"

Respond ONLY as JSON in the following format (no extra text):

[
  {{"headline": "...", "copy": "..."}},
  {{"headline": "...", "copy": "..."}},
  {{"headline": "...", "copy": "..."}}
]
"""  # noqa: E501

    try:
        chat_response = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise advertising copy generator that always follows instructions and JSON format strictly.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.7,
        )
        raw_text = chat_response.choices[0].message.content
    except Exception as e:
        return jsonify({"success": False, "error": f"Text generation failed: {e}"}), 500

    try:
        variations = json_lib.loads(raw_text)
        if not isinstance(variations, list) or len(variations) != 3:
            raise ValueError("Model did not return 3 variations.")
    except Exception:
        variations = [
            {"headline": product_name[:60] or "ACE Ad Variation 1", "copy": raw_text or ""},
            {"headline": product_name[:60] or "ACE Ad Variation 2", "copy": raw_text or ""},
            {"headline": product_name[:60] or "ACE Ad Variation 3", "copy": raw_text or ""},
        ]

    # ---- 2) IMAGE GENERATION ----
    results = []
    for item in variations:
        headline = (item.get("headline") or "").strip()
        copy_text = (item.get("copy") or "").strip()

        img_data_url = None

        visual_prompt = f"""Create a single photographic advertising image.

Brand: ACE Advertising Engine (do not write the brand name in the image).
Product: {product_name}
Description: {product_description}

Headline text to EMBED in the image: "{headline}"

Visual rules:
- Use a clear hybrid object composed from two real-world elements that symbolise the product and its benefit.
- Photographic, realistic lighting and materials. No illustration, no icons, no logos.
- Dark, elegant background suitable for a premium ad.
- Do NOT include any marketing copy in the image, only the short English headline.
- No additional UI elements, no borders, no watermarks."""  # noqa: E501

        try:
            img_resp = client.images.generate(
                model=IMAGE_MODEL,
                prompt=visual_prompt,
                size=openai_size,
                n=1,
                response_format="b64_json",
            )
            b64 = img_resp.data[0].b64_json
            img_data_url = f"data:image/png;base64,{b64}"
        except Exception as e:
            # On any error, fall back to local text-on-black image
            img_data_url = make_fallback_image(headline, size_key)

        results.append(
            {
                "headline": headline,
                "copy": copy_text,
                "image_url": img_data_url,
                "size": size_key,
            }
        )

    return jsonify({"success": True, "variations": results}), 200


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
