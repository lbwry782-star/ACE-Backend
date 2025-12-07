import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai

app = Flask(__name__)

# CORS — allow your frontend domain
frontend_url = os.environ.get("FRONTEND_URL")
if frontend_url:
    CORS(app, resources={r"/*": {"origins": [frontend_url]}})
else:
    CORS(app)

# OpenAI legacy client (works with current key + models)
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    openai.api_key = None
else:
    openai.api_key = api_key
    openai.timeout = 120  # seconds for slow generations

TEXT_MODEL = os.environ.get("OPENAI_TEXT_MODEL", "gpt-4.1-mini")
IMAGE_MODEL = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1")

ALLOWED_SIZES = {"1024x1024", "1024x1536", "1536x1024"}


def build_copy_50_words(product, description, extra_hint=""):
    """Fallback copy generator: around 50 words, English only."""
    base_parts = []
    if product:
        base_parts.append(product)
    if description:
        base_parts.append(description)
    if extra_hint:
        base_parts.append(extra_hint)
    base = ". ".join([p.strip() for p in base_parts if p.strip()])
    filler = (
        "Discover the benefits today and boost your results with this "
        "automated creative advertising engine designed for powerful "
        "social media campaigns and engaging visual storytelling worldwide."
    )
    words = (base + " " + filler).split()
    if len(words) < 50:
        extra = (filler + " ") * 5
        words = (base + " " + extra).split()
    return " ".join(words[:50])


def run_engine(product, description):
    """TEXT ENGINE V2 — Audience → Targets → Objects → A/B → Headline/Copy/Prompt.

    Returns a dict:
    {
      "audience": {...},
      "ads":[
        {
          "target": "...",
          "objects": {
            "A": "...",
            "B": "...",
            "list": ["...", ... up to 80]
          },
          "visual_mode": "swap" or "placement",
          "headline": "...",
          "copy": "...",
          "image_prompt": "..."
        }, ...
      ]
    }
    If anything fails, returns a simple fallback structure.
    """
    # Basic fallback structure
    fallback = {
        "audience": {
            "age_range": "24-44",
            "lifestyle": "creative, digital-first marketers",
            "needs": [
                "faster ad production",
                "original visual concepts",
                "better campaign performance"
            ],
            "familiarity": "high",
            "tone_preference": "confident"
        },
        "ads": [
            {
                "target": "Awareness",
                "objects": {
                    "A": "Lightbulb",
                    "B": "Keyboard",
                    "list": []
                },
                "visual_mode": "placement",
                "headline": "Ideas That Build Themselves",
                "copy": build_copy_50_words(product, description, "awareness ad"),
                "image_prompt": (
                    "A realistic advertising photograph in a real studio. "
                    "A warm glowing lightbulb placed very close to a modern keyboard "
                    "on a dark desk, shallow depth of field, soft consistent lighting, "
                    "no logos, no written text inside the image, no extra objects."
                )
            },
            {
                "target": "Benefit",
                "objects": {
                    "A": "Engine Gear",
                    "B": "Stopwatch",
                    "list": []
                },
                "visual_mode": "placement",
                "headline": "Faster Than Any Creative Team",
                "copy": build_copy_50_words(product, description, "benefit ad"),
                "image_prompt": (
                    "A realistic advertising photograph. A metallic engine gear and a "
                    "sleek stopwatch placed side by side on a clean industrial desk, "
                    "symbolizing speed and precision. Soft directional light, no logos, "
                    "no text inside the image, no extra objects, commercial photo style."
                )
            },
            {
                "target": "Emotion",
                "objects": {
                    "A": "Hourglass",
                    "B": "Chess King",
                    "list": []
                },
                "visual_mode": "placement",
                "headline": "Before Your Competitor Does",
                "copy": build_copy_50_words(product, description, "urgency ad"),
                "image_prompt": (
                    "A realistic advertising photograph. A glass hourglass and a black "
                    "chess king standing very close together on a wooden table, in a "
                    "spotlight that emphasizes tension and strategy. No floating objects, "
                    "no logos, no text inside the image, no extra items, studio lighting."
                )
            }
        ]
    }

    if openai.api_key is None:
        return fallback

    try:
        engine_prompt = (
            "You are the ACE ENGINE V2. You must output ONLY valid JSON, no prose.\n\n"
            "Goal: From a product and description, build an advertising engine structure:\n"
            "- Infer audience (age_range, lifestyle, needs, familiarity, tone_preference).\n"
            "- Define exactly 3 ads: Awareness, Benefit, Emotion.\n"
            "- For each ad, generate 80 physical objects only (no abstract concepts).\n"
            "- For each ad, choose Object A (primary meaning) and Object B (secondary meaning).\n"
            "- Decide visual_mode: 'swap' for high shape similarity or 'placement' otherwise.\n"
            "- Create an English headline (3-6 words).\n"
            "- Create exactly 50 English words of persuasive marketing copy.\n"
            "- Create an English image_prompt describing a realistic advertising photograph "
            "that follows these rules: real environment, no floating objects, no text inside "
            "the image, no logos, coherent lighting, commercial style.\n\n"
            "Product name: " + product + "\n"
            "Product description: " + description + "\n\n"
            "Return ONLY a JSON object with this structure (no extra text):\n"
            "{\n"
            "  \"audience\": {\n"
            "    \"age_range\": \"...\",\n"
            "    \"lifestyle\": \"...\",\n"
            "    \"needs\": [\"...\", \"...\"],\n"
            "    \"familiarity\": \"low|medium|high\",\n"
            "    \"tone_preference\": \"...\"\n"
            "  },\n"
            "  \"ads\": [\n"
            "    {\n"
            "      \"target\": \"Awareness\",\n"
            "      \"objects\": {\n"
            "        \"A\": \"...\",\n"
            "        \"B\": \"...\",\n"
            "        \"list\": [\"physical object 1\", \"physical object 2\", ... ]\n"
            "      },\n"
            "      \"visual_mode\": \"swap\" or \"placement\",\n"
            "      \"headline\": \"...\",\n"
            "      \"copy\": \"exactly 50 English words\",\n"
            "      \"image_prompt\": \"realistic advertising photo description\"\n"
            "    },\n"
            "    {\n"
            "      \"target\": \"Benefit\",\n"
            "      \"objects\": { ... },\n"
            "      \"visual_mode\": \"swap\" or \"placement\",\n"
            "      \"headline\": \"...\",\n"
            "      \"copy\": \"exactly 50 English words\",\n"
            "      \"image_prompt\": \"...\"\n"
            "    },\n"
            "    {\n"
            "      \"target\": \"Emotion\",\n"
            "      \"objects\": { ... },\n"
            "      \"visual_mode\": \"swap\" or \"placement\",\n"
            "      \"headline\": \"...\",\n"
            "      \"copy\": \"exactly 50 English words\",\n"
            "      \"image_prompt\": \"...\"\n"
            "    }\n"
            "  ]\n"
            "}\n"
        )

        chat_resp = openai.ChatCompletion.create(
            model=TEXT_MODEL,
            messages=[{"role": "user", "content": engine_prompt}],
            temperature=0.4,
        )
        content = chat_resp["choices"][0]["message"]["content"]

        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = content[start : end + 1]
        else:
            json_str = content

        engine_data = json.loads(json_str)

        if "ads" not in engine_data or not isinstance(engine_data["ads"], list):
            return fallback

        while len(engine_data["ads"]) < 3:
            engine_data["ads"].append(fallback["ads"][len(engine_data["ads"])])

        for idx in range(3):
            if idx >= len(fallback["ads"]):
                break
            ad = engine_data["ads"][idx]
            fb_ad = fallback["ads"][idx]

            ad.setdefault("target", fb_ad["target"])
            ad.setdefault("objects", fb_ad["objects"])
            ad.setdefault("visual_mode", fb_ad["visual_mode"])
            ad.setdefault("headline", fb_ad["headline"])
            ad.setdefault("copy", fb_ad["copy"])
            ad.setdefault("image_prompt", fb_ad["image_prompt"])

            words = (ad.get("copy") or "").split()
            if len(words) < 30 or len(words) > 70:
                ad["copy"] = fb_ad["copy"]

        if "audience" not in engine_data:
            engine_data["audience"] = fallback["audience"]

        return engine_data

    except Exception:
        return fallback


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/engine", methods=["POST"])
def engine_endpoint():
    if openai.api_key is None:
        return jsonify({"error": "OPENAI_API_KEY missing on server"}), 500

    data = request.get_json(silent=True) or {}
    product = (data.get("product") or "").strip()
    description = (data.get("description") or "").strip()

    if not product:
        return jsonify({"error": "Missing 'product' in request body"}), 400
    if not description:
        return jsonify({"error": "Missing 'description' in request body"}), 400

    engine_data = run_engine(product, description)
    return jsonify(engine_data), 200


@app.route("/generate", methods=["POST"])
def generate():
    if openai.api_key is None:
        return jsonify({"error": "OPENAI_API_KEY missing on server"}), 500

    data = request.get_json(silent=True) or {}

    token = data.get("token", None)
    if not token:
        return jsonify({"error": "Token missing or false. Generation is not allowed."}), 403

    product = (data.get("product") or "").strip()
    description = (data.get("description") or "").strip()
    size = (data.get("size") or "").strip() or "1024x1024"

    if not product:
        return jsonify({"error": "Missing 'product' in request body"}), 400
    if not description:
        return jsonify({"error": "Missing 'description' in request body"}), 400
    if size not in ALLOWED_SIZES:
        return jsonify({"error": f"Unsupported size '{size}'"}), 400

    try:
        engine_data = run_engine(product, description)
        ads_plan = engine_data.get("ads") or []

        while len(ads_plan) < 3:
            ads_plan.append(ads_plan[0])

        first_prompt = (ads_plan[0].get("image_prompt") or "").strip()
        if not first_prompt:
            first_prompt = (
                f"High-quality realistic advertising photograph for product '{product}'. "
                "Two real objects combined or placed side by side in a clever way, "
                "no logos, no written text inside the image, realistic lighting, "
                "suitable for a professional commercial campaign."
            )

        img_resp = openai.Image.create(
            model=IMAGE_MODEL,
            prompt=first_prompt,
            size=size,
            n=3
        )

        images_data = img_resp.get("data", [])
        if len(images_data) < 3:
            images_data = (images_data * 3)[:3]

        ads_out = []
        for idx in range(3):
            ad_plan = ads_plan[idx]
            img_item = images_data[idx]
            b64_data = img_item.get("b64_json")

            headline = (ad_plan.get("headline") or "").strip()
            copy_text = (ad_plan.get("copy") or "").strip()

            if not headline:
                headline = f"ACE for {product}"[:60]
            if not copy_text:
                copy_text = build_copy_50_words(product, description)

            ad_payload = {
                "headline": headline,
                "copy": copy_text,
            }
            if b64_data:
                ad_payload["image_base64"] = b64_data

            ads_out.append(ad_payload)

        if not ads_out:
            return jsonify({"error": "No ads generated from image model"}), 500

        return jsonify({"ads": ads_out, "size_used": size}), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal generation error", "details": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
