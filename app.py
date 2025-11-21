import os
import io
import zipfile
from datetime import datetime
from flask import Flask, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


@app.get("/health")
def health():
    return jsonify({"status": "ok", "time": datetime.utcnow().isoformat() + "Z", "demo": "amir_gottlieb_v1"})


PRODUCT_NAME = "Amir Gottlieb – History Teacher"
PRODUCT_DESC = "A focused history teacher for final examination classes, helping students understand events, dates and big ideas before the exam."

AD_1_TEXT = (
    "Ad 1 – Calm before the exam\n\n"
    "Turn panic into clarity. Amir Gottlieb breaks big history stories into simple, memorable paths so final exams feel structured, not scary. "
    "Short focused sessions, clear explanations, and smart tips that help students walk into the history exam feeling ready, not confused."
)

AD_2_TEXT = (
    "Ad 2 – From dates to meaning\n\n"
    "History is more than names and dates. With Amir Gottlieb, students learn how to connect events, understand causes and consequences, and answer questions with confidence. "
    "Perfect for final exam classes who need someone patient, organized and fully focused on real exam questions."
)

AD_3_TEXT = (
    "Ad 3 – Last-minute rescue\n\n"
    "Final exam around the corner? Amir Gottlieb helps students quickly organize the material: timelines, key concepts and typical questions. "
    "Instead of flipping through endless pages alone, they get a calm guide who highlights exactly what matters most for the history test."
)

COPY_TXT = (
    f"ACE demo package for product: {PRODUCT_NAME}\n"
    f"Short description from Builder: {PRODUCT_DESC}\n\n"
    "This demo ZIP contains three static ad images (JPG) and three matching English marketing texts as simple .txt files.\n"
    "In the full ACE engine, both visuals and copy would be generated automatically for any product according to the official advertising rules.\n"
)


@app.post("/generate")
def generate():
    mem_file = io.BytesIO()
    with zipfile.ZipFile(mem_file, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i in range(1, 4):
            img_path = os.path.join(os.path.dirname(__file__), f"ad_{i}.jpg")
            if os.path.exists(img_path):
                with open(img_path, "rb") as f:
                    zf.writestr(f"ad_{i}.jpg", f.read())

        zf.writestr("ad_1.txt", AD_1_TEXT)
        zf.writestr("ad_2.txt", AD_2_TEXT)
        zf.writestr("ad_3.txt", AD_3_TEXT)
        zf.writestr("copy.txt", COPY_TXT)

    mem_file.seek(0)
    response = send_file(mem_file, mimetype="application/zip")
    response.headers["Content-Disposition"] = "attachment; filename=ace_ads_package.zip"
    return response


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
