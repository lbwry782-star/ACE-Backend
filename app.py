
import os
from flask import Flask, jsonify
from flask_cors import CORS

try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception:
    client = None

app = Flask(__name__)
CORS(app)

IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini")

@app.route("/model-test", methods=["GET"])
def model_test():
    results = {
        "IMAGE_MODEL_OK": False,
        "TEXT_MODEL_OK": False,
        "IMAGE_ERROR": None,
        "TEXT_ERROR": None,
    }

    if client is None:
        results["IMAGE_ERROR"] = "OpenAI client not loaded"
        results["TEXT_ERROR"] = "OpenAI client not loaded"
        return jsonify(results)

    # test text model
    try:
        r = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )
        results["TEXT_MODEL_OK"] = True
    except Exception as e:
        results["TEXT_ERROR"] = str(e)

    # test image model
    try:
        r = client.images.generate(
            model=IMAGE_MODEL,
            prompt="test",
            size="512x512",
            n=1
        )
        results["IMAGE_MODEL_OK"] = True
    except Exception as e:
        results["IMAGE_ERROR"] = str(e)

    return jsonify(results)

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
