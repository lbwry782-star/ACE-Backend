import os
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)

frontend_url = os.environ.get("FRONTEND_URL")
if frontend_url:
    CORS(app, resources={r"/*": {"origins": frontend_url}})
else:
    CORS(app)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

IMAGE_MODEL = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1")
TEXT_MODEL = os.environ.get("OPENAI_TEXT_MODEL", "gpt-4.1-mini")


@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200


def map_size(user_size: str) -> str:
    # Accept only the three legal sizes, default to 1024x1024
    allowed = {"1024x1024", "1024x1792", "1792x1024"}
    if user_size in allowed:
        return user_size
    return "1024x1024"


@app.post("/generate")
def generate():
    data = request.get_json(force=True, silent=True) or {}

    product = data.get("product", "").strip()
    description = data.get("description", "").strip()
    size = map_size(data.get("size", ""))

    if not product:
        return jsonify({"error": "Missing product"}), 400

    # Build a concise prompt summarizing the ACE Engine idea at high level
    base_prompt = (
        f"Create a highly realistic advertising image for the product: '{product}'. "
        f"Product description: '{description}'. "
        "Use two physical objects only in the scene, merged or placed together in a meaningful way, "
        "with no third object. Follow photo-realistic lighting and avoid surreal or broken reality. "
        "Leave empty space for a short English headline on top of the image."
    )

    try:
        img_response = client.images.generate(
            model=IMAGE_MODEL,
            prompt=base_prompt,
            size=size,
            n=3
        )
    except Exception as e:
        return jsonify({"error": f"Image generation failed: {e}"}), 500

    # Extract base64 images
    image_b64_list = []
    try:
        for d in img_response.data:
            if hasattr(d, "b64_json") and d.b64_json:
                image_b64_list.append(d.b64_json)
    except Exception:
        pass

    if len(image_b64_list) != 3:
        # Fallback: if something went wrong
        return jsonify({"error": "Did not receive 3 images from model"}), 500

    ads = []

    for idx, img_b64 in enumerate(image_b64_list, start=1):
        # Generate headline + copy
        try:
            msg = [
                {
                    "role": "system",
                    "content": (
                        "You are an advertising copywriter. "
                        "Write a sharp English headline (3-7 words) and a 50-word body copy for a digital ad. "
                        "The ad is based on a realistic image created from two physical objects only, "
                        "with no surreal or broken reality."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Product: {product}\n"
                        f"Description: {description}\n"
                        "Return your answer in JSON with keys 'headline' and 'copy'. "
                        "The 'copy' must be exactly 50 words."
                    ),
                },
            ]

            completion = client.chat.completions.create(
                model=TEXT_MODEL,
                messages=msg,
                temperature=0.7,
            )
            content = completion.choices[0].message.content

        except Exception as e:
            return jsonify({"error": f"Text generation failed: {e}"}), 500

        # Try to parse JSON-like response, fallback to raw text
        headline = f"ACE Ad {idx}"
        copy_text = content.strip()

        try:
            import json
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                headline = parsed.get("headline", headline).strip() or headline
                copy_text = parsed.get("copy", copy_text).strip() or copy_text
        except Exception:
            # not JSON, leave as is
            pass

        ads.append(
            {
                "image_base64": img_b64,
                "headline": headline,
                "copy": copy_text,
            }
        )

    return jsonify({"ads": ads}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
