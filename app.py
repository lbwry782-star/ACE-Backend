import os
import json
import base64
import io
import zipfile
import random

from flask import Flask, request, jsonify
from flask_cors import CORS

import openai

# Configure OpenAI (legacy 0.28 style)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Use models that are definitely supported by openai==0.28
# Text: gpt-3.5-turbo
# Image: dall-e-2  (sizes: 256x256, 512x512, 1024x1024)
OPENAI_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-3.5-turbo")
OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "dall-e-2")

app = Flask(__name__)

frontend_url = os.getenv("FRONTEND_URL")
if frontend_url:
    CORS(app, resources={r"/*": {"origins": [frontend_url]}})
else:
    CORS(app)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


# ---- Helper: call text model to get audience, goal, 100 associations ----

PLANNER_SYSTEM_PROMPT = """You are the ACE Engine text planner.

Your job:
1. Infer the target audience profile from the product name and description.
2. Infer ONE clear campaign goal.
3. Generate EXACTLY 100 visual associations for that goal.

CRITICAL RULES FOR ASSOCIATIONS:
- You MUST generate EXACTLY 100 associations.
- Each association MUST be a concrete, physical, photographable object.
- Objects must be tangible, real-world things that can appear in a photo:
  Examples: "glass bottle", "aluminum ladder", "wristwatch", "yellow umbrella",
            "car headlight", "pineapple slice", "coffee mug", "ice cube",
            "magnifying glass", "stethoscope".
- Do NOT use abstract concepts (success, freedom, innovation, clarity, luxury),
  emotions (joy, fear, trust), qualities (strength, freshness, speed),
  or metaphors without objects (new beginnings, high performance, productivity),
  or actions/verbs (running fast, competing, winning).
- Each association MUST be 1–4 English words and refer only to an actual object.

Return valid JSON with this exact structure:
{
  "audience": "...",
  "goal": "...",
  "associations": [
    "object 1",
    "object 2",
    ...
    "object 100"
  ]
}
"""


def call_text_planner(product: str, description: str):
    """Call the OpenAI text model to get audience, goal, and 100 associations."""
    user_prompt = f"""
Product: {product}
Description: {description}

Infer audience + campaign goal and then produce EXACTLY 100 concrete object associations as specified.
Respond ONLY with the JSON object, no extra text.
""".strip()

    try:
        resp = openai.ChatCompletion.create(
            model=OPENAI_TEXT_MODEL,
            messages=[
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.9,
        )
        content = resp["choices"][0]["message"]["content"]
        data = json.loads(content)
        audience = data.get("audience", "")
        goal = data.get("goal", "")
        associations = data.get("associations", [])
        # Ensure exactly 100 strings
        associations = [str(a) for a in associations][:100]
        if len(associations) < 100 and associations:
            while len(associations) < 100:
                associations.append(associations[-1])
        if not associations:
            # very defensive: fallback
            raise ValueError("No associations returned")
        return audience, goal, associations
    except Exception as e:
        print("Text planner error:", repr(e))
        audience = "general audience"
        goal = "increase interest in the product"
        fallback_objects = [
            "glass bottle",
            "ice cube",
            "coffee mug",
            "yellow umbrella",
            "car headlight",
            "pineapple slice",
            "wristwatch",
            "shopping bag",
            "spotlight",
            "magnifying glass",
        ]
        associations = (fallback_objects * 10)[:100]
        return audience, goal, associations


# ---- Helper: choose hybrid object pair from associations ----

def choose_hybrid_pairs(associations, count=3):
    """
    Choose 'count' pairs (A,B) of objects from the association list.
    Simple random pairing respecting diversity.
    """
    pairs = []
    pool = associations[:]
    random.shuffle(pool)
    for i in range(count):
        if len(pool) < 2:
            pool = associations[:]
            random.shuffle(pool)
        a = pool.pop()
        b = pool.pop()
        pairs.append((a, b))
    return pairs


# ---- Helper: call image model to create one hybrid image ----

def map_size_to_dalle2(size_str: str) -> str:
    """
    DALL·E 2 only supports 256x256, 512x512, 1024x1024.
    We always map to 1024x1024 (the highest quality) regardless of input.
    """
    return "1024x1024"


def generate_hybrid_image(product, audience, goal, object_a, object_b, requested_size):
    """
    Calls the image model to generate one hybrid object image and returns base64 JPEG.
    Uses DALL·E 2 (dall-e-2). Frontend sizes are mapped to 1024x1024.
    """
    prompt = f"""
Ultra-realistic photographic advertisement for: {product}.

Target audience: {audience}.
Campaign goal: {goal}.

Create a single clear hybrid object combining:
- Object A: {object_a}
- Object B: {object_b}

Composition rules:
- Show ONE hybrid object in the center of the frame.
- The hybrid must clearly merge BOTH objects, not just show them side by side.
- Use classic, natural background appropriate to the objects, not a plain studio backdrop.
- Cinematic lighting, subtle depth of field, no text in the image.
- No logos, no brands, no watermarks, no extra props that distract from the hybrid.

Output style: hyper-realistic color photograph.
""".strip()

    dalle_size = map_size_to_dalle2(requested_size)

    try:
        img_resp = openai.Image.create(
            model=OPENAI_IMAGE_MODEL,
            prompt=prompt,
            size=dalle_size,
            n=1,
            response_format="b64_json",
        )
        b64 = img_resp["data"][0]["b64_json"]
        return b64
    except Exception as e:
        print("Image generation error:", repr(e))
        return ""


# ---- Helper: ZIP packaging ----

def build_zip_from_image_and_copy(image_b64: str, headline: str, copy: str, index: int) -> str:
    """
    Create an in-memory ZIP with:
    - ad{index}.jpg  (decoded from base64)
    - copy.txt       (headline + copy)
    Return base64-encoded ZIP.
    """
    mem_file = io.BytesIO()
    with zipfile.ZipFile(mem_file, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        if image_b64:
            try:
                img_bytes = base64.b64decode(image_b64)
                z.writestr(f"ad{index}.jpg", img_bytes)
            except Exception as e:
                print("ZIP image decode error:", repr(e))
        text_content = f"{headline}\n\n{copy}"
        z.writestr("copy.txt", text_content.encode("utf-8"))

    mem_file.seek(0)
    zip_bytes = mem_file.read()
    zip_b64 = base64.b64encode(zip_bytes).decode("utf-8")
    return zip_b64


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(force=True, silent=True) or {}
    product = data.get("product", "").strip()
    description = data.get("description", "").strip()
    size = data.get("size", "1024x1024")

    if not product:
        return jsonify({"error": "Missing 'product' in request body"}), 400

    # 1. Call text planner: audience, goal, 100 associations (concrete objects only)
    audience, goal, associations = call_text_planner(product, description)

    # 2. Choose 3 hybrid pairs
    pairs = choose_hybrid_pairs(associations, count=3)

    ads = []

    for idx, (obj_a, obj_b) in enumerate(pairs, start=1):
        headline = f"{product}: {obj_a.title()} Meets {obj_b.title()}"
        copy = (
            f"{product} reimagined as a bold hybrid between {obj_a} and {obj_b}. "
            f"Designed for {audience}, this visual metaphor supports the goal: {goal}. "
            "Striking, memorable and built to stand out in any ACE campaign."
        )

        image_b64 = generate_hybrid_image(product, audience, goal, obj_a, obj_b, size)

        zip_b64 = build_zip_from_image_and_copy(image_b64, headline, copy, idx)
        zip_filename = f"ace_ad_{idx}.zip"

        ads.append(
            {
                "headline": headline,
                "copy": copy,
                "image_base64": image_b64,
                "zip_base64": zip_b64,
                "zip_filename": zip_filename,
            }
        )

    return jsonify({"ads": ads}), 200


if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
