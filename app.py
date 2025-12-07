import os
import io
import json
import base64

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI

app = Flask(__name__)
CORS(app)

# --- OpenAI client ---
# Uses OPENAI_API_KEY from environment automatically.
client = OpenAI()

OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
OPENAI_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini")

# Application-level supported sizes (what the FRONTEND sends)
APP_SIZES = {
    "1024x1024": (1024, 1024),
    "1024x1792": (1024, 1792),
    "1792x1024": (1792, 1024),
}


def map_size_for_image_model(size_str: str) -> str:
    """
    Map app size to a legal size for the image model.
    For gpt-image-1 we must use one of: 1024x1024, 1024x1536, 1536x1024.
    For other models (like dall-e-3) we pass the size through.
    """
    if OPENAI_IMAGE_MODEL.startswith("gpt-image-1"):
        mapping = {
            "1024x1024": "1024x1024",
            "1024x1792": "1024x1536",
            "1792x1024": "1536x1024",
        }
        return mapping.get(size_str, "1024x1024")
    # Fallback: pass-through (dall-e-3 supports 1024x1024, 1024x1792, 1792x1024)
    return size_str if size_str in ("1024x1024", "1024x1792", "1792x1024") else "1024x1024"


def _draw_headline_on_image(img: Image.Image, headline: str) -> Image.Image:
    """
    Draw the headline text on top of the image (bottom center) with a dark translucent strip.
    Headline is assumed to be short (3–7 words) in English.
    """
    img = img.convert("RGBA")
    w, h = img.size
    draw = ImageDraw.Draw(img)

    # Choose font size relative to image height
    font_size = max(28, h // 18)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    text = headline.strip()
    if not text:
        return img.convert("RGB")

    # Measure text, with a simple retry if too wide
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    padding_x = 40
    padding_y = 25
    if text_w > (w - 2 * padding_x):
        # Reduce font size a bit and re-measure
        font_size = max(20, font_size - 8)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

    x = (w - text_w) // 2
    y = h - text_h - padding_y

    # Dark translucent rounded rectangle behind text
    rect_margin = 18
    rect_coords = (x - rect_margin, y - rect_margin,
                   x + text_w + rect_margin, y + text_h + rect_margin)

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    try:
        od.rounded_rectangle(rect_coords, radius=12, fill=(0, 0, 0, 160))
    except Exception:
        # Fallback if rounded_rectangle is unavailable
        od.rectangle(rect_coords, fill=(0, 0, 0, 160))

    img = Image.alpha_composite(img, overlay)

    # White text
    draw = ImageDraw.Draw(img)
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))

    return img.convert("RGB")


def _fit_image_to_canvas(image_b64: str, target_size):
    """
    Decode base64 PNG/JPEG, fit proportionally into a target canvas size with black bars,
    then return base64-encoded JPEG to keep file size reasonable.
    """
    data = base64.b64decode(image_b64)
    with Image.open(io.BytesIO(data)) as im:
        im = im.convert("RGB")
        tw, th = target_size
        iw, ih = im.size

        # Keep aspect ratio, fit inside target
        ratio = min(tw / iw, th / ih)
        new_w = int(iw * ratio)
        new_h = int(ih * ratio)
        im_resized = im.resize((new_w, new_h), Image.LANCZOS)

        canvas = Image.new("RGB", (tw, th), (0, 0, 0))
        offset = ((tw - new_w) // 2, (th - new_h) // 2)
        canvas.paste(im_resized, offset)

        buffer = io.BytesIO()
        canvas.save(buffer, format="JPEG", quality=95)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _call_ace_engine(product: str, description: str):
    """
    Call GPT text model once to get 3 ad specs (headline, copy, visual_prompt).
    We let the model implement the ACE Engine logic internally.
    """
    # Single prompt; Responses API will return plain text we parse as JSON.
    prompt = f"""
You are the ACE advertising engine.
Your job is to design 3 different ad concepts for a single product.

INPUT:
- Product name: "{product}"
- Product description: "{description}"

Internally (do NOT describe these steps in the output) you MUST:
1) Infer ONE coherent target audience from the product (age, lifestyle, needs, familiarity with the category).
2) Define 3 different advertising objectives (one objective for each ad).
3) For each objective, imagine a list of 80 concrete physical objects (no abstract ideas or feelings).
   From this list, choose a pair of objects:
   - A = main object
   - B = supporting object
4) Decide if the pair is used as:
   - "fusion"  → shapes overlap, a single hybrid object is formed
   - "placement" → two separate objects positioned side-by-side with a clear relationship.

Then OUTPUT must be valid JSON only (no commentary, no markdown) with this exact structure:

[
  {{
    "headline": "...",        // 3–7 English words, suitable to be written on the image, no brand names
    "copy": "...",            // exactly 50 English words, English marketing copy suitable for a social media post
    "visual_prompt": "...",   // detailed English prompt for a PHOTOGRAPHIC image model
    "objective": "...",       // very short phrase naming the ad objective
    "mode": "fusion"          // or "placement"
  }},
  {{
    "headline": "...",
    "copy": "...",
    "visual_prompt": "...",
    "objective": "...",
    "mode": "placement"
  }},
  {{
    "headline": "...",
    "copy": "...",
    "visual_prompt": "...",
    "objective": "...",
    "mode": "fusion"
  }}
]

Rules for visual_prompt:
- Describe objects A and B, their relationship and background.
- Make the result a realistic photograph (not illustration, not drawing).
- Do NOT include any written text, letters, numbers or logos in the image.
- Obey real-world lighting and physics (no impossible floating objects without support).
- If mode = "fusion" → clearly describe the single hybrid object so it can exist in real life.
- If mode = "placement" → clearly describe the two separate objects and how they are positioned.

Important:
- Output MUST be valid JSON only.
- Do NOT mention "JSON" or these rules inside the JSON itself.
"""

    response = client.responses.create(
        model=OPENAI_TEXT_MODEL,
        input=prompt,
        max_output_tokens=1500,
    )

    raw = response.output_text
    try:
        data = json.loads(raw)
        if not isinstance(data, list) or len(data) != 3:
            raise ValueError("Engine JSON is not a list of 3 items")
        return data
    except Exception as e:
        raise RuntimeError(f"Failed to parse ACE engine JSON: {e}")


def _generate_single_image(visual_prompt: str, headline: str, app_size_key: str) -> str:
    """
    Use the image model to generate a base image from the visual prompt,
    then fit it to the requested canvas and draw the headline on top.
    Returns base64-encoded JPEG.
    """
    model_size = map_size_for_image_model(app_size_key)
    target_size = APP_SIZES[app_size_key]

    # Extra safety text to avoid text inside the raw image.
    full_prompt = (
        "Photographic advertising image. Ultra realistic, high quality. "
        "Do NOT include any written text, letters, logos or numbers inside the image. "
        + visual_prompt.strip()
    )

    img_response = client.images.generate(
        model=OPENAI_IMAGE_MODEL,
        prompt=full_prompt,
        n=1,
        size=model_size,
    )

    # For gpt-image-1 and modern image APIs we always get base64 encoded images in data[0].b64_json
    try:
        base_png_b64 = img_response.data[0].b64_json
    except Exception as e:
        raise RuntimeError(f"Image API did not return expected data: {e}")

    # Fit to our canvas
    fitted_b64 = _fit_image_to_canvas(base_png_b64, target_size)

    # Draw headline on top of the fitted image
    data = base64.b64decode(fitted_b64)
    with Image.open(io.BytesIO(data)) as im:
        im = _draw_headline_on_image(im, headline)
        buffer = io.BytesIO()
        im.save(buffer, format="JPEG", quality=95)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/generate", methods=["POST"])
def generate():
    """
    Main generation endpoint.

    Expected JSON body:
    {
        "product": "Name",
        "description": "Line or two",
        "size": "1024x1024" | "1024x1792" | "1792x1024",
        "token": true/false   // must be true or product==\"4242\" for override
    }
    """
    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON body"}), 400

    product = (payload or {}).get("product", "").strip()
    description = (payload or {}).get("description", "").strip()
    size = (payload or {}).get("size", "").strip()
    token = bool((payload or {}).get("token", False))

    # Basic validation
    if not product:
        return jsonify({"error": "Missing 'product'"}), 400
    if not description:
        return jsonify({"error": "Missing 'description'"}), 400
    if size not in APP_SIZES:
        return jsonify({"error": "Unsupported 'size' value"}), 400

    # OVERRIDE MODE (4242)
    override_mode = product == "4242"

    # TOKEN MODE — must send token=true from FRONTEND after payment
    if (not override_mode) and not token:
        return jsonify({"error": "Token missing or false. Generation is not allowed."}), 403

    try:
        # Get 3 ad specs from ACE engine (headlines, copy, visual prompts)
        specs = _call_ace_engine(product, description)

        ads_output = []
        for spec in specs:
            headline = str(spec.get("headline", "")).strip()
            copy = str(spec.get("copy", "")).strip()
            visual_prompt = str(spec.get("visual_prompt", "")).strip()

            if not visual_prompt or not headline or not copy:
                raise RuntimeError("ACE engine returned incomplete ad spec")

            img_b64 = _generate_single_image(visual_prompt, headline, size)

            ads_output.append(
                {
                    "image_base64": img_b64,
                    "headline": headline,
                    "copy": copy,
                    "objective": spec.get("objective", ""),
                    "mode": spec.get("mode", ""),
                }
            )

        return jsonify({"ads": ads_output}), 200

    except RuntimeError as e:
        # Engine or image-specific error
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        # Catch-all
        return jsonify({"error": "Unexpected server error", "details": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
