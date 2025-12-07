import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# In-memory override mode flag (strongest state)
OVERRIDE_ACTIVE = False


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


def build_engine_prompt(product: str, description: str) -> str:
    """Summarize the ENGINE document into a single instruction prompt."""
    return f"""You are the ACE Engine.
Product: {product}
Description: {description}

Follow these rules:
1. Infer a realistic target audience (age, lifestyle, preferences, pains, needs, familiarity level).
2. Define 3 distinct advertising goals (3 separate ads). Example only: Awareness, Value/Benefit, Emotion/Urgency.
3. For each goal, internally consider many concrete, physical objects (no abstract concepts, no feelings).
4. For each ad, choose a pair of objects A (central) and B (associative) and design one strong visual concept:
   - A and B exist in real physical space, with one consistent realistic background.
   - Use either:
     • Shape-Swap (high shape similarity, creating a fused hybrid object), or
     • Placement (low/medium similarity, two separate objects placed in a meaningful relationship).
   - Never break physical reality: no impossible shadows, no unnatural floating, no third object, no mutilation.
5. Your output must be valid JSON with this structure:
{{
  "ads": [
    {{
      "headline": "Short headline, up to 6–8 words",
      "copy": "Exactly 50 words of persuasive ad copy.",
      "image_prompt": "Detailed, concrete visual instructions for the image model that implement the ACE rules above, including A and B, background and lighting."
    }},
    {{
      "headline": "...",
      "copy": "... (50 words)",
      "image_prompt": "..."
    }},
    {{
      "headline": "...",
      "copy": "... (50 words)",
      "image_prompt": "..."
    }}
  ]
}}

Return JSON only, with no extra text at all.
"""


def generate_ads(product: str, description: str, size: str):
    """Call OpenAI once for text+prompts, then 3 times for images."""
    text_model = os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini")
    image_model = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")

    prompt = build_engine_prompt(product, description)

    chat = client.chat.completions.create(
        model=text_model,
        messages=[
            {"role": "system", "content": "You are a precise JSON generator."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )

    raw_text = chat.choices[0].message.content
    try:
        data = json.loads(raw_text)
    except Exception:
        # Try to extract JSON if the model wrapped it in backticks or extra text
        try:
            start = raw_text.index("{")
            end = raw_text.rindex("}") + 1
            data = json.loads(raw_text[start:end])
        except Exception as e:
            raise RuntimeError(f"Failed to parse JSON from model: {e}")

    ads = data.get("ads") or []
    if len(ads) != 3:
        raise RuntimeError("Model did not return exactly 3 ads.")

    results = []
    for ad in ads:
        img_prompt = ad.get("image_prompt") or ""
        if not img_prompt:
            raise RuntimeError("Missing image_prompt for one of the ads.")

        image_resp = client.images.generate(
            model=image_model,
            prompt=img_prompt,
            size=size,
            n=1,
        )
        b64 = image_resp.data[0].b64_json

        results.append(
            {
                "headline": ad.get("headline", ""),
                "copy": ad.get("copy", ""),
                "image_base64": b64,
            }
        )

    return results


@app.route("/generate", methods=["POST"])
def generate():
    """Main generation endpoint with TOKEN + OVERRIDE logic.

    IMPORTANT BEHAVIOR (matches your spec):
    - Frontend is allowed to click GENERATE.
    - Backend is the authority that decides if creation is allowed.
    - If creation fails for any reason in TOKEN mode → the user
      should be allowed to try again (no token burn on failure).
    """
    global OVERRIDE_ACTIVE

    data = request.get_json(force=True, silent=False) or {}
    product = (data.get("product") or "").strip()
    description = (data.get("description") or "").strip()
    size = (data.get("size") or "1024x1024").strip()
    token_flag = bool(data.get("token"))
    override_flag = bool(data.get("override"))

    if not product:
        return jsonify({"error": "Missing 'product' in request body"}), 400

    if size not in ("1024x1024", "1536x1024", "1024x1536"):
        return jsonify({"error": "Unsupported size"}), 400

    # 1) Handle 4242 override activation (does NOT create an ad)
    if product == "4242":
        OVERRIDE_ACTIVE = True
        return jsonify({"status": "override_activated"}), 200

    # 2) Consolidate override state: strongest state in the system
    if override_flag:
        OVERRIDE_ACTIVE = True

    if OVERRIDE_ACTIVE:
        mode = "override"
    elif token_flag:
        # User came with a valid token (after payment)
        mode = "token"
    else:
        # No valid token / override → block creation
        return jsonify({"error": "Generation not allowed. No valid token or override."}), 403

    # 3) Try to generate ads. Any failure returns HTTP 200 with an error,
    # so the frontend can show a message and allow another attempt.
    try:
        ads = generate_ads(product, description, size)
        return jsonify({"mode": mode, "ads": ads}), 200
    except Exception as e:
        # Do NOT burn the token here. The frontend will see `error` and
        # will NOT mark the token as used.
        return jsonify(
            {
                "mode": mode,
                "error": f"Generation failed: {e}",
                "ads": [],
            }
        ), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
