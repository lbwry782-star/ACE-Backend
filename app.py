
    import os
    import json
    import base64
    import io
    import logging
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    from openai import OpenAI
    import zipfile

    TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini")
    IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
    FRONTEND_URL = os.getenv("FRONTEND_URL", "")

    client = OpenAI()

    app = Flask(__name__)

    if FRONTEND_URL:
        CORS(app, resources={r"/*": {"origins": [FRONTEND_URL]}}, supports_credentials=True)
    else:
        CORS(app)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("ace-backend")


    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"}), 200


    def _map_size_label(ad_size: str) -> str:
        if not ad_size:
            return "1024x1024"
        label = ad_size.strip().lower()
        mapping = {
            "1024 x 1024 – square": "1024x1024",
            "1024 x 1024 - square": "1024x1024",
            "1024×1024 – square": "1024x1024",
            "1024×1024 - square": "1024x1024",
            "1024 x 1536 – portrait": "1024x1536",
            "1024 x 1536 - portrait": "1024x1536",
            "1536 x 1024 – landscape": "1536x1024",
            "1536 x 1024 - landscape": "1536x1024",
        }
        return mapping.get(label, "1024x1024")


    def _parse_variations(raw_text: str):
        try:
            return json.loads(raw_text)
        except Exception:
            pass

        start = raw_text.find("[")
        end = raw_text.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw_text[start : end + 1])
            except Exception:
                pass

        return None


    @app.route("/generate_ads", methods=["POST"])
    def generate_ads():
        try:
            data = request.get_json(force=True) or {}
        except Exception:
            return jsonify({"error": "Invalid JSON body"}), 400

        product = (data.get("product") or "").strip()
        description = (data.get("description") or "").strip()
        ad_size_label = (data.get("ad_size") or "").strip()

        if not product:
            return jsonify({"error": "Missing 'product' in request body"}), 400

        if len(product.split()) > 15:
            return jsonify({"error": "Product name must be 15 words or fewer"}), 400

        size = _map_size_label(ad_size_label)

        text_prompt = f"""
You are the text engine for the ACE Advertising Engine.

You receive a product name and an optional description and must create EXACTLY 3 ad variations.

Product: {product}
Description: {description or "N/A"}
Ad size: {ad_size_label or "1024 x 1024 – Square"}

For each variation, generate:
- "headline": 3–7 words, catchy, English only.
- "copy": exactly 50 words, persuasive, English only, a single paragraph.
- "image_prompt": 1–2 short sentences (max 60 words) describing a photographic hybrid-object visual that could be used for this ad. No text in the image, no logos, no celebrities, no brands.

Return ONLY a valid JSON array with 3 objects and the keys: "headline", "copy", "image_prompt".
        """.strip()

        try:
            chat_resp = client.chat.completions.create(
                model=TEXT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful advertising copywriter and art director. You always follow the instructions exactly and always return valid JSON when asked.",
                    },
                    {"role": "user", "content": text_prompt},
                ],
                temperature=0.9,
                max_tokens=900,
            )
        except Exception as e:
            logger.exception("Error calling OpenAI chat model")
            return jsonify({"error": f"Text generation failed: {str(e)}"}), 500

        raw_output = chat_resp.choices[0].message.content
        variations = _parse_variations(raw_output)

        if not isinstance(variations, list) or len(variations) == 0:
            logger.error("Failed to parse JSON variations from model output")
            return jsonify({"error": "Failed to parse variations from text model output"}), 500

        if len(variations) < 3:
            while len(variations) < 3:
                variations.append(variations[-1])

        variations = variations[:3]

        image_prompt_base = variations[0].get("image_prompt") or f"Photographic advertisement for {product}. High quality studio lighting, realistic details."

        full_image_prompt = (
            f"{image_prompt_base} "
            f"Advertising photograph, no text, no logos, no watermarks, high quality, professional lighting."
        )

        try:
            img_resp = client.images.generate(
                model=IMAGE_MODEL,
                prompt=full_image_prompt,
                n=3,
                size=size,
                response_format="b64_json",
                quality="high",
            )
        except Exception as e:
            logger.exception("Error calling OpenAI image model")
            return jsonify({"error": f"Image generation failed: {str(e)}"}), 500

        image_data_list = []
        for i, item in enumerate(img_resp.data):
            b64 = item.b64_json if hasattr(item, "b64_json") else item.get("b64_json")
            if not b64:
                logger.error("Missing b64_json in image response item %s", i)
                return jsonify({"error": "Invalid image response from OpenAI"}), 500
            image_data_list.append(b64)

        results = []
        for idx, (variation, img_b64) in enumerate(zip(variations, image_data_list), start=1):
            headline = variation.get("headline") or f"Ad Variation {idx}"
            copy = variation.get("copy") or ""

            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                img_bytes = base64.b64decode(img_b64)
                zf.writestr(f"ace_ad_{idx}.png", img_bytes)
                text_content = f"{headline}\n\n{copy}"
                zf.writestr(f"ace_ad_{idx}.txt", text_content)

            zip_buffer.seek(0)
            zip_b64 = base64.b64encode(zip_buffer.read()).decode("ascii")

            results.append(
                {
                    "headline": headline,
                    "copy": copy,
                    "image_b64": f"data:image/png;base64,{img_b64}",
                    "zip_b64": zip_b64,
                    "zip_filename": f"ace_ad_{idx}.zip",
                    "size": size,
                }
            )

        return jsonify(
            {
                "success": True,
                "product": product,
                "ad_size_label": ad_size_label,
                "size": size,
                "variations": results,
            }
        ), 200


    if __name__ == "__main__":
        port = int(os.environ.get("PORT", "10000"))
        app.run(host="0.0.0.0", port=port)
