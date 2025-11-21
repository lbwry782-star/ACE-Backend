import os
import io
import zipfile
from datetime import datetime
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

app = Flask(__name__)
# Allow all origins so GitHub Pages frontend can always talk to this backend
CORS(app, resources={r"/*": {"origins": "*"}})


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "time": datetime.utcnow().isoformat() + "Z",
        }
    )


def build_copy_variants(product: str, description: str):
    # Very simple deterministic templates for three English marketing texts.
    base_desc = description if description else "A smart, ready-to-use solution for busy people who want fast, professional results."
    p = product or "your product"

    copy1 = (
        f"Meet {p}, designed for real life. {base_desc} "
        "Save time, cut the guesswork, and enjoy a smooth experience from the first use. "
        "Perfect for anyone who cares about quality but refuses to waste hours learning complicated tools."
    )

    copy2 = (
        f"Turn everyday moments into something special with {p}. {base_desc} "
        "Built for modern creators and small teams, it delivers clarity, speed, and confidence. "
        "Start today and feel the difference of a tool that finally works the way you think."
    )

    copy3 = (
        f"{p} gives you an unfair advantage. {base_desc} "
        "Instead of struggling with scattered tools and messy workflows, you get one focused solution. "
        "Ideal for ambitious people who want consistent, professional results without hiring an expensive agency."
    )

    return copy1, copy2, copy3


@app.post("/generate")
def generate():
    data = request.get_json(silent=True) or {}
    product = (data.get("product") or "Your product").strip()
    description = (data.get("description") or "").strip()
    size_str = (data.get("size") or "1080x1350").strip()

    copy1, copy2, copy3 = build_copy_variants(product, description)

    mem_file = io.BytesIO()
    with zipfile.ZipFile(mem_file, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "ad_1.txt",
            f"Ad 1 – {product}\n\n{copy1}\n",
        )
        zf.writestr(
            "ad_2.txt",
            f"Ad 2 – {product}\n\n{copy2}\n",
        )
        zf.writestr(
            "ad_3.txt",
            f"Ad 3 – {product}\n\n{copy3}\n",
        )

        copy_text = (
            f"ACE demo package for product: {product}\n"
            f"Short description from Builder: {description}\n\n"
            "This demo ZIP contains three simple marketing texts in English.\n"
            "Each file (ad_1.txt, ad_2.txt, ad_3.txt) is a different variation based on the same product.\n"
            "In the full ACE engine, each text would be a polished 50-word script paired with a visual ad.\n"
        )
        zf.writestr("copy.txt", copy_text)

    mem_file.seek(0)
    from flask import Response

    response = send_file(mem_file, mimetype="application/zip")
    response.headers["Content-Disposition"] = "attachment; filename=ace_ads_package.zip"
    return response


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
