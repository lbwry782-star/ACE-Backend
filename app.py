import os
import base64
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
    data = request.get_json(silent=True) or {}
    product = (data.get("product") or "").strip()
    description = (data.get("description") or "").strip()
    size = sanitize_size(str(data.get("size") or "1024x1024"))

    if not product or client is None:
        return jsonify({"images": [], "headlines": [], "copies": []})

    prompt = (
        "You are an ad-creation engine called ACE. "
        "Create a photographic advertising image based on the following product. "
        "Use a single hybrid object made from exactly two real physical objects that "
        "metaphorically represent the product and its advertising objective. "
        "Do not place any text inside the image; the headline will be added separately. "
        f"Product name: {product}. "
        f"Product description: {description}. "
        "Style: clean, high-end studio photography, realistic lighting, no logos, no brand names."
    )

    image_resp = client.images.generate(
        model=IMAGE_MODEL,
        prompt=prompt,
        size=size,
        n=3
    )

    images_b64 = [item.b64_json for item in image_resp.data]
    images_data_urls = [f"data:image/png;base64,{b64}" for b64 in images_b64]

    headlines = []
    copies = []
    for _ in range(3):
        chat = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an advertising copywriter. "
                        "Write a short English headline (3–7 words) and a 50-word marketing copy "
                        "for a photographic ad. The headline must not mention words like 'hybrid' or 'composite'. "
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
        try:
            import json
            parsed = json.loads(content)
            headlines.append(str(parsed.get("headline", "")).strip())
            copies.append(str(parsed.get("copy", "")).strip())
        except Exception:
            headlines.append("Your new ACE ad")
            copies.append(
                "Discover a powerful new way to present your product. "
                "ACE blends clarity, emotion and simple visual impact so your message stands out."
            )

    return jsonify(
        {
            "images": images_data_urls,
            "headlines": headlines,
            "copies": copies,
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
