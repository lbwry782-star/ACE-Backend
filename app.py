import os
import io
import zipfile
from datetime import datetime
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

app = Flask(__name__)
# Allow all origins so GitHub Pages frontend can always call us
CORS(app, resources={r"/*": {"origins": "*"}})


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "time": datetime.utcnow().isoformat() + "Z",
        }
    )


@app.post("/generate")
def generate():
    data = request.get_json(silent=True) or {}
    product = (data.get("product") or "Your product").strip()
    description = (data.get("description") or "").strip()
    size_str = (data.get("size") or "1080x1350").strip()

    # We don't actually need the size to build the ZIP in this demo.
    # Parsing is only to keep compatibility with the frontend.
    try:
        width, height = map(int, size_str.lower().split("x"))
    except Exception:
        width, height = 1080, 1350

    mem_file = io.BytesIO()
    with zipfile.ZipFile(mem_file, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Three simple text placeholders for ads
        for i in range(1, 4):
            zf.writestr(
                f"ad_{i}.txt",
                (
                    f"ACE demo – Ad {i}\n"
                    f"Product: {product}\n"
                    f"Description: {description}\n"
                    f"Frame size: {width}x{height}\n"
                ),
            )

        copy_text = (
            f"ACE demo package for product: {product}\n"
            f"Short description: {description}\n\n"
            "This demo ZIP contains three placeholder ads as text files.\n"
            "In the full ACE engine, each ad would be a photographic hybrid-object visual "
            "with a 50-word marketing text, according to the official engine rules and "
            "Terms & Policies.\n"
        )
        zf.writestr("copy.txt", copy_text)

    mem_file.seek(0)
    # Very conservative send_file usage, compatible with old Flask versions
    response = send_file(mem_file, mimetype="application/zip")
    response.headers["Content-Disposition"] = "attachment; filename=ace_ads_package.zip"
    return response


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
