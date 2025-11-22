import os
import io
import base64
from datetime import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS

from openai import OpenAI

# --------------------------------------------------------------------
# Basic config
# --------------------------------------------------------------------

ENGINE_VERSION = "ENGINE_V0_8"
DEFAULT_IMAGE_MODEL = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1")
DEFAULT_TEXT_MODEL = os.environ.get("OPENAI_TEXT_MODEL", "gpt-4.1-mini")

FRONTEND_URL = os.environ.get("FRONTEND_URL")

app = Flask(__name__)

# CORS: restrict to the configured frontend if provided, otherwise allow all (for debugging)
if FRONTEND_URL:
    CORS(app, resources={r"/*": {"origins": [FRONTEND_URL]}})
else:
    CORS(app)

client = OpenAI()

# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------


def build_hybrid_brief(product_name: str, description: str) -> str:
    """
    Create a short design brief for the hybrid-object ad.
    We keep this intentionally compact to save tokens.
    """
    desc = description.strip()
    prompt = (
        "You are an advertising art director. "
        "Design a single striking hybrid object for a square digital ad. "
        "The ad must be photographic only, no text, no logos, no brands. "
        "Build a hybrid object from two real-world objects with high shape similarity. "
        "The background should be simple and must not distract from the hybrid object. "
        f"Product: {product_name}. "
        f"Product description: {desc}. "
        "Describe in 2–3 short sentences the hybrid object and its background."
    )
    return prompt


def generate_text_variations(product_name: str, description: str, ad_dimensions: str):
    """
    Ask the text model for three short headlines + 50-word copies.
    Returns a list of dicts: [{headline, copy}, ...].
    """
    system_msg = (
        "You are an advertising copywriter. "
        "Write ad copy in English only. "
        "Each ad must have a short headline (3–7 words) and exactly 50 words of body copy. "
        "No brand names, no pricing, no contact details."
    )

    user_msg = (
        f"Product name: {product_name}\n"
        f"Product description: {description}\n"
        f"Ad size: {ad_dimensions}\n\n"
        "Create 3 different ad concepts for this product. "
        "For each concept, output in JSON with this schema:\n"
        "{ \"ads\": [ {\"headline\": \"...\", \"copy\": \"50-word text\"}, ... ] }"
    )

    completion = client.chat.completions.create(
        model=DEFAULT_TEXT_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )

    content = completion.choices[0].message.content
    data = {}
    try:
        data = json.loads(content)
    except Exception:
        # very defensive: fallback to simple placeholders
        return [
            {
                "headline": f"{product_name} – Ad #{i+1}",
                "copy": (
                    f"This is a placeholder 50-word marketing text for {product_name}. "
                    "Use this ad to attract attention and highlight the unique benefits "
                    "of the product. Replace this copy with a custom text once the system "
                    "is fully configured."
                ),
            }
            for i in range(3)
        ]

    ads = data.get("ads", [])
    results = []
    for i in range(3):
        if i < len(ads) and isinstance(ads[i], dict):
            headline = str(ads[i].get("headline", f"{product_name} – Ad #{i+1}")).strip()
            copy = str(ads[i].get("copy", "")).strip()
        else:
            headline = f"{product_name} – Ad #{i+1}"
            copy = (
                f"This is a fallback 50-word marketing text for {product_name}. "
                "Use this ad to attract attention and highlight the unique benefits "
                "of the product."
            )
        results.append({"headline": headline, "copy": copy})

    return results


def generate_image_bytes(prompt: str, size: str) -> bytes:
    """
    Call OpenAI Images API and return raw JPG bytes for a single image.
    """
    response = client.images.generate(
        model=DEFAULT_IMAGE_MODEL,
        prompt=prompt,
        size=size,
        n=1,
        response_format="b64_json",
    )

    b64 = response.data[0].b64_json
    return base64.b64decode(b64)


def normalise_size(dimensions: str) -> str:
    """
    Map the UI dimension string (e.g. '1080 x 1080') to one of the supported
    OpenAI image sizes. We keep it simple: square -> 1024x1024, vertical -> 1024x1536.
    """
    dims = dimensions.lower().replace("×", "x").replace(" ", "")
    if "1080x1350" in dims or "1080x1920" in dims:
        return "1024x1536"
    elif "1080x566" in dims or "1200x628" in dims or "1640x856" in dims:
        return "1344x768"
    else:
        # default square
        return "1024x1024"


# --------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------


@app.route("/health", methods=["GET"])
def health():
    """
    Simple health check used by Render and by you in the browser.
    Also tells you whether an OpenAI key seems to be configured.
    """
    openai_ok = bool(os.environ.get("OPENAI_API_KEY"))
    return jsonify(
        {
            "engine": ENGINE_VERSION,
            "image_model": DEFAULT_IMAGE_MODEL,
            "openai_configured": openai_ok,
            "status": "ok",
            "time": datetime.utcnow().isoformat(),
        }
    )


@app.route("/generate", methods=["POST"])
def generate():
    """
    Main generation endpoint.
    Expects JSON:
    {
      "product_name": "...",
      "description": "...",
      "ad_dimensions": "1080 x 1080"
    }

    Returns JSON:
    {
      "status": "ok",
      "engine": "...",
      "ads": [
        {
          "id": 1,
          "headline": "...",
          "copy": "...",
          "image_base64": "data:image/jpeg;base64,..."
        },
        ...
      ]
    }
    """
    try:
        payload = request.get_json(force=True, silent=False)
    except Exception:
        return jsonify({"status": "error", "message": "Invalid JSON payload."}), 400

    product_name = (payload or {}).get("product_name", "").strip()
    description = (payload or {}).get("description", "").strip()
    ad_dimensions = (payload or {}).get("ad_dimensions", "").strip() or "1080 x 1080"

    if not product_name or not description:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Both product_name and description are required.",
                }
            ),
            400,
        )

    # --- Generate text concepts first ---
    try:
        text_ads = generate_text_variations(product_name, description, ad_dimensions)
    except Exception as e:
        app.logger.exception("Error while generating text variations")
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Failed to generate ad text. Please try again later.",
                }
            ),
            500,
        )

    # --- Generate images, one by one, to avoid long blocking calls ---
    image_size = normalise_size(ad_dimensions)
    ads_out = []
    brief = build_hybrid_brief(product_name, description)

    for idx, text_ad in enumerate(text_ads, start=1):
        # Slightly vary the prompt per ad so each image is unique
        variation_prompt = (
            brief
            + f" Create variation #{idx} with a clearly distinct hybrid object idea, "
            "but always based on strong shape similarity between the two source objects."
        )
        try:
            img_bytes = generate_image_bytes(variation_prompt, image_size)
            img_b64 = base64.b64encode(img_bytes).decode("ascii")
            data_url = f"data:image/jpeg;base64,{img_b64}"
        except Exception:
            app.logger.exception("Error while generating image %s", idx)
            # Fallback: transparent 1x1 pixel
            data_url = (
                "data:image/gif;base64,R0lGODlhAQABAAAAACw="
                "AAAAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw=="
            )

        ads_out.append(
            {
                "id": idx,
                "headline": text_ad["headline"],
                "copy": text_ad["copy"],
                "image_base64": data_url,
            }
        )

    return jsonify({"status": "ok", "engine": ENGINE_VERSION, "ads": ads_out})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
