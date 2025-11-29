
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
import base64
import io
import zipfile
from openai import OpenAI

app = Flask(__name__)
CORS(app)

# Simple in-memory token store: {token: {"used": bool}}
TOKENS = {}

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

TEXT_MODEL = os.environ.get("OPENAI_TEXT_MODEL", "gpt-4.1-mini")
IMAGE_MODEL = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1")


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/create-token")
def create_token():
    token = str(uuid.uuid4())
    TOKENS[token] = {"used": False}
    return jsonify({"token": token})


def build_text_prompt(product: str, description: str) -> str:
    return f"""You are the ACE advertising engine.

PRODUCT: {product}
DESCRIPTION: {description}

ENGINE RULES (follow strictly):
1. Always use TWO concrete objects from a fixed internal list of 100 associative objects (do not invent abstract concepts).
2. You must NEVER show a single object. The ad always includes TWO real objects.
3. Decide between:
   a) HYBRID OBJECT
   b) TWO OBJECTS SIDE-BY-SIDE

HYBRID OBJECT RULES:
- Use when the SHAPE SIMILARITY between the two objects is HIGH.
- Hybrid means:
  * Object A in the foreground with its classic natural background.
  * Object B still visible in the background with its own classic natural background.
- You can overlap shapes, like an eclipse, but both objects and their environments must be readable.
- Do NOT create a vague or abstract merge; keep both objects concrete and recognizable.

SIDE-BY-SIDE RULES:
- Use when SHAPE SIMILARITY is LOW.
- Show Object A and Object B as two separate objects in the same frame.
- Adjust their angles and perspective so they appear as visually related as possible.
- You may rotate, tilt, or re-frame them to emphasize shape correspondence.

GENERAL VISUAL RULES:
- Photographic style only (no illustration, no cartoons).
- Respect a clean composition with safe margins.
- Do NOT add any marketing copy, slogans, or small text in the image.
- The only text that may appear inside the frame is a short HEADLINE (3–7 words), in English.

TASK:
1. Choose Object A and Object B from your internal list, associatively relevant to the PRODUCT and its goal.
2. Decide if this variation should be HYBRID or SIDE-BY-SIDE, based on shape similarity.
3. Describe the visual in one detailed sentence, including:
   - whether it is HYBRID or SIDE-BY-SIDE,
   - what Object A is and its classic environment,
   - what Object B is and its classic environment,
   - how they are arranged in the frame (foreground/background or side-by-side).
4. Create a short English HEADLINE (3–7 words) suitable to be embedded inside the image near the objects.
5. Create a 50-word English marketing copy for social media (outside the image).

Return a JSON object with EXACTLY this schema:
{{
  "ads": [
    {{
      "layout": "hybrid" or "side-by-side",
      "object_a": "short name of object A",
      "object_b": "short name of object B",
      "visual_prompt": "one-sentence visual description for the image generation, including layout, foreground/background logic and classic environments",
      "headline": "3-7 word headline for the image",
      "copy": "exactly 50 words of English marketing text for social media"
    }},
    {{
      "layout": "...",
      "object_a": "...",
      "object_b": "...",
      "visual_prompt": "...",
      "headline": "...",
      "copy": "..."
    }},
    {{
      "layout": "...",
      "object_a": "...",
      "object_b": "...",
      "visual_prompt": "...",
      "headline": "...",
      "copy": "..."
    }}
  ]
}}"""


@app.post("/generate")
def generate():
    data = request.get_json(force=True, silent=True) or {}
    product = (data.get("product") or "").strip()
    description = (data.get("description") or "").strip()
    size = (data.get("size") or "1080x1080").strip()
    token = (data.get("token") or "").strip()

    if not product:
        return jsonify({"error": "missing product"}), 400

    dev_mode = (product == "4242")

    # Token enforcement: one generation per token (unless dev mode)
    if not dev_mode:
        info = TOKENS.get(token)
        if not token or not info:
            return jsonify({"error": "unknown or missing token"}), 404
        if info.get("used"):
            return jsonify({"error": "token already used"}), 403

    # Build text prompt according to ACE engine rules
    prompt = build_text_prompt(product, description)

    try:
        txt = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8
        )
    except Exception as e:
        return jsonify({"error": f"text generation failed: {e}"}), 500

    content = txt.choices[0].message.content

    # Parse the JSON returned by the model
    import json
    try:
        parsed = json.loads(content)
        ads_spec = parsed.get("ads", [])
    except Exception as e:
        return jsonify({"error": f"failed to parse JSON from text model: {e}", "raw": content}), 500

    if not isinstance(ads_spec, list) or len(ads_spec) == 0:
        return jsonify({"error": "no ads definition returned from text model", "raw": content}), 500

    # Convert ACE sizes to OpenAI image sizes (use closest)
    def size_to_openai(sz: str) -> str:
        s = sz.lower().strip()
        if "1920" in s:
            return "1080x1920"
        if "1350" in s:
            return "1080x1350"
        if "566" in s or "628" in s or "856" in s:
            return "1200x628"
        return "1024x1024"

    openai_size = size_to_openai(size)

    ads_out = []

    for idx, spec in enumerate(ads_spec[:3], start=1):
        visual_prompt = spec.get("visual_prompt") or ""
        headline = spec.get("headline") or ""
        copy_text = spec.get("copy") or ""

        # Build final image prompt: include headline rule clearly
        img_prompt = (
            f"Photographic advertisement image for product '{product}'. "
            f"{visual_prompt} "
            f"Embed this short English headline inside the image near the hybrid/objects: '{headline}'. "
            "Do not include any additional text. No logos, no brands, no celebrities."
        )

        try:
            img = client.images.generate(
                model=IMAGE_MODEL,
                prompt=img_prompt,
                size=openai_size,
                n=1
            )
        except Exception as e:
            return jsonify({"error": f"image generation failed: {e}"}), 500

        b64_image = img.data[0].b64_json
        image_bytes = base64.b64decode(b64_image)

        # Build ZIP in memory: image + copy.txt
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"ace_ad_{idx}.png", image_bytes)
            zf.writestr("copy.txt", copy_text)
        zip_bytes = zip_buffer.getvalue()
        zip_b64 = base64.b64encode(zip_bytes).decode("utf-8")

        # For immediate preview: use data URL
        image_data_url = f"data:image/png;base64,{b64_image}"

        ads_out.append({
            "headline": headline,
            "copy": copy_text,
            "image_url": image_data_url,
            "zip_base64": zip_b64,
            "filename": f"ace_ad_{idx}.zip"
        })

    if not dev_mode and token in TOKENS:
        TOKENS[token]["used"] = True

    return jsonify({"ads": ads_out})
