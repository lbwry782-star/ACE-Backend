import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)

# CORS: allow only the configured frontend URL if present
frontend_url = os.environ.get("FRONTEND_URL")
if frontend_url:
    CORS(app, resources={r"/*": {"origins": [frontend_url]}})
else:
    CORS(app)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

TEXT_MODEL = os.environ.get("OPENAI_TEXT_MODEL", "gpt-4.1-mini")
IMAGE_MODEL = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1")

ALLOWED_SIZES = {"1024x1024", "1024x1536", "1536x1024"}


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


def build_copy_prompt(product: str, description: str) -> str:
    return (
        "You are the ACE Advertising Engine copywriter.\n"
        "You write short English headlines and exactly 50-word English marketing copy for ads.\n\n"
        f"Product: {product.strip()}\n"
        f"Description: {description.strip()}\n\n"
        "Create 3 different ad variations that could match 3 different hybrid-object images for this product.\n"
        "Rules:\n"
        "- Headlines must be 3–7 English words.\n"
        "- Each marketing copy must be exactly 50 English words.\n"
        "- No hashtags. No emojis. No price claims unless clearly implied.\n"
        "- Focus on benefits, promise, and a soft call-to-action.\n\n"
        "Output format (TEXT ONLY, no JSON):\n"
        "AD 1\n"
        "HEADLINE: <headline for ad 1>\n"
        "COPY: <50-word marketing copy for ad 1>\n"
        "---\n"
        "AD 2\n"
        "HEADLINE: <headline for ad 2>\n"
        "COPY: <50-word marketing copy for ad 2>\n"
        "---\n"
        "AD 3\n"
        "HEADLINE: <headline for ad 3>\n"
        "COPY: <50-word marketing copy for ad 3>\n"
        "Do not add anything else."
    )


def parse_ads(raw_text: str):
    """Parse the raw text into a list of 3 ads: [{headline, copy}, ...]."""
    ads = []
    chunks = [c.strip() for c in raw_text.split("---") if c.strip()]

    for chunk in chunks:
        headline = None
        copy_lines = []
        for line in chunk.splitlines():
            line_s = line.strip()
            upper = line_s.upper()
            if upper.startswith("HEADLINE:"):
                headline = line_s.split(":", 1)[1].strip()
            elif upper.startswith("COPY:"):
                # first line of copy, excluding the prefix
                first_copy = line_s.split(":", 1)[1].strip()
                if first_copy:
                    copy_lines.append(first_copy)
            elif copy_lines:
                copy_lines.append(line_s)

        if headline and copy_lines:
            copy_text = " ".join(copy_lines).strip()
            ads.append({"headline": headline, "copy": copy_text})

        if len(ads) == 3:
            break

    # Fallback – if parsing failed, treat whole text as one big copy
    if not ads:
        ads.append(
            {
                "headline": "ACE Hybrid Ad",
                "copy": raw_text.strip(),
            }
        )

    # Always return 3 ads (repeat last if needed)
    while len(ads) < 3:
        ads.append(ads[-1])

    return ads


def generate_copies(product: str, description: str):
    """Call OpenAI Responses API and extract plain text safely."""
    prompt = build_copy_prompt(product, description)
    response = client.responses.create(
        model=TEXT_MODEL,
        input=prompt,
        max_output_tokens=700,
    )

    text = None

    # Prefer the convenience attribute if available
    try:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            text = output_text
    except Exception:
        text = None

    # Fallback to the first output/content item
    if not text:
        try:
            first_output = response.output[0]
            first_content = first_output.content[0]
            text_block = getattr(first_content, "text", None)
            if hasattr(text_block, "value"):
                text = text_block.value
            elif isinstance(text_block, str):
                text = text_block
        except Exception:
            text = None

    # Last resort: string representation
    if not text:
        text = str(response)

    return parse_ads(text)


def build_image_prompt(product: str, description: str, headline: str) -> str:
    """Prompt gpt-image-1 to create a hybrid-object ad with a very bold headline on the image."""
    return (
        "Create a hyper-realistic photographic advertising image for the following product. "
        "Use a single central hybrid object built from two real-world objects that are conceptually related "
        "to the product's main benefit. Use shape overlap, like a gentle solar eclipse, to merge them. "
        "Place the hybrid object on a minimal dark background with soft light and no clutter.\n\n"
        f"Product: {product}\n"
        f"Short description: {description or 'Use common-sense assumptions about the product.'}\n"
        "The image must contain a single English headline, placed in a strong composition near the hybrid object, "
        "without hiding it. The headline text must be very bold, high-contrast, and clearly readable at first glance. "
        "Do NOT add any long marketing text in the image. Only embed this short English headline once:\n"
        f"\"{headline}\"\n"
        "No logos unless they are part of the headline text. No extra frames or borders."
    )


def generate_images_for_ads(product: str, description: str, ads, size: str):
    results = []
    target_size = size if size in ALLOWED_SIZES else "1024x1024"

    for ad in ads:
        headline = (ad.get("headline") or "ACE Hybrid Ad").strip()
        image_prompt = build_image_prompt(product, description, headline)
        img = client.images.generate(
            model=IMAGE_MODEL,
            prompt=image_prompt,
            size=target_size,
            n=1,
            quality="high",
        )

        image_b64 = None
        if img and getattr(img, "data", None):
            first = img.data[0]
            image_b64 = getattr(first, "b64_json", None)

        results.append(
            {
                "headline": headline,
                "copy": ad.get("copy", ""),
                "image_b64": image_b64,
            }
        )

    return results


@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json(force=True, silent=True) or {}
        product = (data.get("product") or "").strip()
        description = (data.get("description") or "").strip()
        size = (data.get("size") or "1024x1024").strip()

        if not product:
            return jsonify({"success": False, "error": "Missing 'product' in request body."}), 400

        ads = generate_copies(product, description)
        ads_with_images = generate_images_for_ads(product, description, ads, size)

        return jsonify({"success": True, "ads": ads_with_images}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
