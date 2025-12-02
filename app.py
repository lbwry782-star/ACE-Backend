import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)
CORS(app)

# ----- OpenAI client configuration -----
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
IMAGE_MODEL = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1")
TEXT_MODEL = os.environ.get("OPENAI_TEXT_MODEL", "gpt-4.1-mini")

client = None
if OPENAI_API_KEY:
    # Timeout is kept lower than gunicorn worker timeout to avoid worker TIMEOUT.
    client = OpenAI(api_key=OPENAI_API_KEY, timeout=20.0)


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "mode": "real",
        "image_model": IMAGE_MODEL,
        "text_model": TEXT_MODEL,
        "has_api_key": bool(OPENAI_API_KEY)
    })


def sanitize_size(size: str) -> str:
    allowed = {"1024x1024", "1024x1792", "1792x1024"}
    return size if size in allowed else "1024x1024"


@app.post("/generate")
def generate():
    """Generate 3 real images + 3 text pairs using OpenAI.

    Designed to be robust in production:
    - If the API key is missing or invalid, returns an empty but valid payload (HTTP 200).
    - All OpenAI calls are inside try/except so the worker never crashes.
    - If anything unexpected happens, returns an empty payload with HTTP 200.
    """
    try:
        data = request.get_json(silent=True) or {}
        product = (data.get("product") or "").strip()
        description = (data.get("description") or "").strip()
        size = sanitize_size(str(data.get("size") or "1024x1024"))

        # If there is no product or no client, quietly return an empty payload.
        if not product or client is None:
            return jsonify({"images": [], "headlines": [], "copies": []})

        # ---- IMAGE GENERATION ----
        images_data_urls = []
        try:
            image_resp = client.images.generate(
                model=IMAGE_MODEL,
                prompt=(
                    "You are an ad-creation engine. "
                    "Create a single photographic advertising image based on the following product. "
                    "Use a clear, concrete visual metaphor for the product and its benefit. "
                    "No text inside the image. No logos, no brand names. "
                    f"Product name: {product}. "
                    f"Product description: {description}. "
                    "Style: high-end studio photography, realistic light, clean background."
                ),
                size=size,
                n=3,
            )
            images_b64 = [item.b64_json for item in image_resp.data]
            images_data_urls = [f"data:image/png;base64,{b64}" for b64 in images_b64]
        except Exception as e:
            # Log server-side but never crash or return 500.
            print("OpenAI image error:", repr(e))
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
                                "Do not mention words like 'hybrid', 'composite', 'AI' "
                                "or 'generated'. "
                                "Return JSON with keys 'headline' and 'copy'."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Product name: {product}. "
                                f"Product description: {description}. "
                                "Focus on the main benefit and a clear metaphor."
                            ),
                        },
                    ],
                    response_format={ "type": "json_object" },
                )
                content = chat.choices[0].message.content
                parsed = json.loads(content)
                headlines.append(str(parsed.get("headline", "")).strip())
                copies.append(str(parsed.get("copy", "")).strip())
            except Exception as e:
                print("OpenAI text error:", repr(e))
                headlines.append("Your new ad")
                copies.append(
                    "Discover a clear, simple way to communicate what you offer. "
                    "This ad focuses on one strong visual idea so your product stands out "
                    "without distractions or noise."
                )

        return jsonify({
            "images": images_data_urls,
            "headlines": headlines,
            "copies": copies,
        })
    except Exception as e:
        # Last‑resort safety: never return HTTP 500.
        print("Unexpected /generate error:", repr(e))
        return jsonify({"images": [], "headlines": [], "copies": []})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
