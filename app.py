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
    # Keep this below gunicorn timeout so worker never hangs too long.
    client = OpenAI(api_key=OPENAI_API_KEY, timeout=15.0)


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "mode": "real-fast",
        "image_model": IMAGE_MODEL,
        "text_model": TEXT_MODEL,
        "has_api_key": bool(OPENAI_API_KEY)
    })


def sanitize_size(size: str) -> str:
    allowed = {"1024x1024", "1024x1792", "1792x1024"}
    return size if size in allowed else "1024x1024"


@app.post("/generate")
def generate():
    try:
        data = request.get_json(silent=True) or {}
        product = (data.get("product") or "").strip()
        description = (data.get("description") or "").strip()
        size = sanitize_size(str(data.get("size") or "1024x1024"))

        if not product or client is None:
            return jsonify({"images": [], "headlines": [], "copies": []})

        # ---- IMAGE (single, then duplicated) ----
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
                n=1,
            )
            images_b64 = [item.b64_json for item in image_resp.data]
            if images_b64:
                url = "data:image/png;base64," + images_b64[0]
                images_data_urls = [url, url, url]
        except Exception as e:
            print("OpenAI image error:", repr(e))
            images_data_urls = []

        # ---- TEXT (three different variations) ----
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
        print("Unexpected /generate error:", repr(e))
        return jsonify({"images": [], "headlines": [], "copies": []})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
