
import os
from flask import Flask, jsonify
from flask_cors import CORS

try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception:
    client = None

IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini")

app = Flask(__name__)
frontend_origin = os.getenv("FRONTEND_ORIGIN", "*")
CORS(app, resources={r"/*": {"origins": frontend_origin}})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/debug-image", methods=["GET"])
def debug_image():
    """Call OpenAI image model once and show the exact error or success."""
    result = {
        "IMAGE_MODEL": IMAGE_MODEL,
        "TEXT_MODEL": TEXT_MODEL,
        "client_loaded": client is not None,
        "success": False,
        "error": None,
    }

    if client is None:
        result["error"] = "OpenAI client not loaded (check OPENAI_API_KEY)"
        return jsonify(result)

    try:
        r = client.images.generate(
            model=IMAGE_MODEL,
            prompt="Simple product photo on a plain background",
            size="1024x1024",
            n=1
        )
        # if we got here, it worked
        result["success"] = True
        result["error"] = None
    except Exception as e:
        result["error"] = str(e)

    return jsonify(result)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
