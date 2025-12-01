import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)
CORS(app)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
IMAGE_MODEL = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1")
TEXT_MODEL = os.environ.get("OPENAI_TEXT_MODEL", "gpt-4.1-mini")

client = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


def sanitize_size(size: str) -> str:
    allowed = {"1024x1024", "1024x1792", "1792x1024"}
    return size if size in allowed else "1024x1024"


@app.post("/generate")
def generate():
    """Generate 3 images + 3 text pairs.
    IMPORTANT: This endpoint must never crash the app.
    On any internal error it returns an empty, valid JSON payload (HTTP 200).
    """
    try:
        data = request.get_json(silent=True) or {}
        product = (data.get("product") or "").strip()
        description = (data.get("description") or "").strip()
        size = sanitize_size(str(data.get("size") or "1024x1024"))

        # If required config or product is missing, return empty but valid payload.
        if not product or client is None:
            return jsonify({"images": [], "headlines": [], "copies": []})

        # ---- IMAGE GENERATION ----
        try:
            image_resp = client.images.generate(
                model=IMAGE_MODEL,
                prompt=(
                    "You are an ad-creation engine called ACE. "
                    "Create a photographic advertising image based on the following product. "
                    "Use a single striking object arrangement that metaphorically represents "
                    "the product and its advertising objective. "
                    "Do not place any text inside the image; the headline will be added separately. "
                    f"Product name: {product}. "
                    f"Product description: {description}. "
                    "Style: clean, high-end studio photography, realistic lighting, "
                    "no logos, no brand names."
                ),
                size=size,
                n=3
            )
            images_b64 = [item.b64_json for item in image_resp.data]
            images_data_urls = [f"data:image/png;base64,{b64}" for b64 in images_b64]
        except Exception as e:
            # If image generation fails, return empty visuals but do NOT crash.
            images_data_urls = []

        # ---- TEXT GENERATION ----
        headlines = []
        copies = []
        for _ in range(3):
            try:
                chat = client.chat.completions.create(
                    model=TEXT_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an advertising copywriter. "
                                "Write a short English headline (3–7 words) and a 50-word "
                                "marketing copy for a photographic ad. "
                                "The headline must not mention words like 'hybrid' or 'composite'. "
                                "Return JSON with keys 'headline' and 'copy'."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Product name: {product}. "
                                f"Product description: {description}. "
                                "Focus on clarity, benefit and a subtle metaphor."
                            ),
                        },
                    ],
                    response_format={"type": "json_object"},
                )
                content = chat.choices[0].message.content
                parsed = json.loads(content)
                headlines.append(str(parsed.get("headline", "")).strip())
                copies.append(str(parsed.get("copy", "")).strip())
            except Exception:
                # Fallback safe text if anything fails
                headlines.append("Your new ad")
                copies.append(
                    "Discover a clear and simple way to present your product. "
                    "This ad focuses on one strong visual idea so your message stands out "
                    "without noise or confusion."
                )

        return jsonify(
            {
                "images": images_data_urls,
                "headlines": headlines,
                "copies": copies,
            }
        )
    except Exception as e:
        # Last-resort safety: never return HTTP 500.
        return jsonify({"images": [], "headlines": [], "copies": []})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
