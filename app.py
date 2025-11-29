import os
import base64
import io
import zipfile
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)

frontend_url = os.environ.get("FRONTEND_URL", "*")
CORS(app, resources={r"/*": {"origins": [frontend_url]}})

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

TEXT_MODEL = os.environ.get("OPENAI_TEXT_MODEL", "gpt-4.1-mini")
IMAGE_MODEL = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


def build_prompt(product, description):
    rules = (
        "You are ACE, an automated advertising engine that generates photographic "
        "hybrid-object ads based on a product.\n\n"
        "For the given product, create 3 distinct ad concepts. Each ad must include:\n"
        "- An English headline (3-7 words).\n"
        "- A 50-word English marketing copy.\n"
        "- An image_prompt describing a photographic hybrid composition using two physical objects (A and B) "
        "AND including the headline text visually inside the image near the hybrid object.\n\n"
        "Association rules:\n"
        "- Imagine exactly 100 concrete, photographable associations (physical objects only) derived from the product, audience, and goal.\n"
        "- From those 100 objects, choose 3 pairs (A,B). Each pair yields one ad concept.\n"
        "- Associations must be real-world objects (no abstract ideas, verbs, or emotions).\n\n"
        "Hybrid rules:\n"
        "- The visual is a hybrid of Object A and Object B, either merged into one form or placed together in a balanced composition.\n"
        "- Photographic style only, no illustration or sketch.\n"
        "- No logos, brands, or watermarks inside the image.\n"
        "- Natural, coherent lighting and background.\n\n"
        "Headline & copy:\n"
        "- Headline: short, punchy, 3-7 English words.\n"
        "- Copy: exactly 50 English words, professional and clear.\n"
        "- Do not mention ACE, AI, or 'hybrid' explicitly in the text.\n"
        "- The headline text must be suitable to appear visually inside the image as a title near the hybrid object.\n"
    )

    prod_block = f"PRODUCT: {product}\nDESCRIPTION: {description or '(no extra description)'}"
    final_prompt = rules + "\n\n" + prod_block + """\n\nOUTPUT FORMAT (STRICT JSON):\n
Return a JSON object with exactly this structure:

{
  "ads": [
    {
      "headline": "English headline 3-7 words",
      "copy": "Exactly 50 English words...",
      "image_prompt": "Detailed English prompt describing the hybrid object, scene, and headline placement inside the image."
    },
    {
      "headline": "...",
      "copy": "...",
      "image_prompt": "..."
    },
    {
      "headline": "...",
      "copy": "...",
      "image_prompt": "..."
    }
  ]
}

- "copy" MUST be exactly 50 words (counted in English words).
- In each image_prompt, describe where the headline text appears inside the image (above, below, or beside the hybrid object) and how it looks.
- Do not include any other keys.
- Do not include comments or explanations outside the JSON.
"""
    return final_prompt


def ensure_50_words(text):
    words = text.strip().split()
    if len(words) > 50:
        words = words[:50]
    elif len(words) < 50:
        last = words[-1] if words else "."
        while len(words) < 50:
            words.append(last)
    return " ".join(words)


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(force=True, silent=True) or {}
    product = (data.get("product") or "").strip()
    description = (data.get("description") or "").strip()
    size = (data.get("size") or "1024x1024").strip()

    if not product:
        return jsonify({"error": "Missing 'product' field"}), 400

    try:
        prompt = build_prompt(product, description)
        completion = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": "You are a precise JSON generator."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content
        parsed = json.loads(content)
        raw_ads = parsed.get("ads", [])
        ads = []

        for idx, item in enumerate(raw_ads[:3]):
            headline = (item.get("headline") or "").strip()
            copy = ensure_50_words(item.get("copy") or "")
            base_img_prompt = (
                item.get("image_prompt")
                or f"A high-quality photographic hybrid-object advertisement for {product}. "
                   f"Include the headline text '{headline}' inside the image near the hybrid object."
            )

            # D = let the engine decide best placement: describe options, not fixed
            full_img_prompt = (
                base_img_prompt
                + f" The image must visually include the headline text: '{headline}' "
                  "inside the composition, placed near the hybrid object (above, below, or to the side), "
                  "clear and readable, professional typography, no extra text."
            )

            img_resp = client.images.generate(
                model=IMAGE_MODEL,
                prompt=full_img_prompt,
                size=size,
            )
            b64_image = img_resp.data[0].b64_json
            image_bytes = base64.b64decode(b64_image)

            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                img_filename = f"ad{idx+1}_image.png"
                txt_filename = f"ad{idx+1}_copy.txt"
                zf.writestr(img_filename, image_bytes)
                zf.writestr(txt_filename, f"{headline}\n\n{copy}")
            zip_bytes = zip_buffer.getvalue()
            zip_b64 = base64.b64encode(zip_bytes).decode("utf-8")

            ads.append(
                {
                    "headline": headline,
                    "copy": copy,
                    "image_base64": b64_image,
                    "zip_base64": zip_b64,
                    "filename": f"ad{idx+1}.zip",
                }
            )

        return jsonify({"ads": ads})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
