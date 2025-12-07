import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)

# CORS — allow your frontend domain
frontend_url = os.environ.get("FRONTEND_URL")
if frontend_url:
    CORS(app, resources={r"/*": {"origins": [frontend_url]}})
else:
    CORS(app)

api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

IMAGE_MODEL = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1")
ALLOWED_SIZES = {"1024x1024", "1024x1536", "1536x1024"}


def build_copy_50_words(product, description, angle):
    """Deterministic 50-word marketing text."""
    base = (
        f"{product} is an automated creative advertising engine for ambitious brands. "
        f"{description} "
        f"This ad focuses on {angle} to help you stand out, attract the right audience and turn attention into action. "
        f"Use it to launch campaigns, test angles and consistently ship fresh visuals across your favorite social platforms."
    )
    words = base.strip().split()
    if len(words) < 50:
        # repeat description if too short
        extra = (description + " ") * 10
        words = (base + " " + extra).strip().split()
    return " ".join(words[:50])


def infer_audience(product, description):
    text = f"{product} {description}".lower()
    lifestyle = "digital-first professionals"
    tone = "confident"
    if any(k in text for k in ["student", "exam", "learning", "school"]):
        lifestyle = "busy students preparing for exams"
        tone = "supportive"
    if any(k in text for k in ["agency", "marketing", "brand", "campaign"]):
        lifestyle = "marketing teams and creative freelancers"
        tone = "bold"
    return {
        "age_range": "22-45",
        "lifestyle": lifestyle,
        "needs": [
            "faster ad production",
            "original visual concepts",
            "better campaign performance",
        ],
        "familiarity": "medium",
        "tone_preference": tone,
    }


def build_objects_for_target(target):
    base_objects = [
        "lightbulb", "keyboard", "camera", "hourglass", "chess king",
        "gear", "stopwatch", "notebook", "coffee cup", "city billboard",
        "studio spotlight", "smartphone", "microphone", "road sign", "bridge",
        "forest path", "window", "elevator button", "clock", "dice",
    ]
    # simple expansion to 80
    objs = []
    i = 0
    while len(objs) < 80:
        name = base_objects[i % len(base_objects)]
        objs.append(f"{target.lower()} object {len(objs)+1} — {name}")
        i += 1
    return objs


def pick_A_B(target):
    if target == "Awareness":
        return "lightbulb", "keyboard"
    if target == "Benefit":
        return "engine gear", "stopwatch"
    if target == "Emotion":
        return "hourglass", "chess king"
    return "object A", "object B"


def visual_mode_for_pair(A, B):
    # simple rule: if words share length and first letter → swap, else placement
    if A[0].lower() == B[0].lower() and len(A.split()[0]) == len(B.split()[0]):
        return "swap"
    return "placement"


def build_image_prompt(product, description, target, A, B, mode):
    base = (
        f"A realistic advertising photograph in a real studio for '{product}'. "
        f"The ad is about {target.lower()} and uses a {A} and a {B} as the main visual elements. "
    )
    if mode == "swap":
        relation = (
            f"The shapes of the {A} and the {B} visually fuse into one believable hybrid object, "
            f"shot from a single clear angle."
        )
    else:
        relation = (
            f"The {A} and the {B} are placed very close together, forming one balanced scene without overlapping shapes."
        )
    tail = (
        " Soft, consistent studio lighting, shallow depth of field, no logos, "
        "no written text inside the image, no extra objects, no unrealistic floating, "
        "photorealistic commercial quality photograph."
    )
    return base + relation + tail


def run_engine(product, description):
    audience = infer_audience(product, description)

    targets = ["Awareness", "Benefit", "Emotion"]
    angles = ["awareness and discovery", "benefits and value", "emotion and urgency"]
    ads = []

    for target, angle in zip(targets, angles):
        objects_list = build_objects_for_target(target)
        A, B = pick_A_B(target)
        mode = visual_mode_for_pair(A, B)
        headline = {
            "Awareness": "See What ACE Can Do",
            "Benefit": "Smarter Ads In Seconds",
            "Emotion": "Win Attention Before They Do",
        }.get(target, f"ACE for {product}"[:40])
        copy_text = build_copy_50_words(product, description, angle)
        prompt = build_image_prompt(product, description, target, A, B, mode)
        ads.append(
            {
                "target": target,
                "objects": {"A": A, "B": B, "list": objects_list},
                "visual_mode": mode,
                "headline": headline,
                "copy": copy_text,
                "image_prompt": prompt,
            }
        )

    return {"audience": audience, "ads": ads}


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/engine", methods=["POST"])
def engine_endpoint():
    data = request.get_json(silent=True) or {}
    product = (data.get("product") or "").strip()
    description = (data.get("description") or "").strip()

    if not product:
        return jsonify({"error": "Missing 'product' in request body"}), 400
    if not description:
        return jsonify({"error": "Missing 'description' in request body"}), 400

    engine_data = run_engine(product, description)
    return jsonify(engine_data), 200


@app.route("/generate", methods=["POST"])
def generate():
    if client is None:
        return jsonify({"error": "OPENAI_API_KEY missing on server"}), 500

    data = request.get_json(silent=True) or {}

    token = data.get("token", None)
    if not token:
        return jsonify({"error": "Token missing or false. Generation is not allowed."}), 403

    product = (data.get("product") or "").strip()
    description = (data.get("description") or "").strip()
    size = (data.get("size") or "").strip() or "1024x1024"

    if not product:
        return jsonify({"error": "Missing 'product' in request body"}), 400
    if not description:
        return jsonify({"error": "Missing 'description' in request body"}), 400
    if size not in ALLOWED_SIZES:
        return jsonify({"error": f"Unsupported size '{size}'"}), 400

    try:
        engine_data = run_engine(product, description)
        ads_plan = engine_data.get("ads") or []
        if len(ads_plan) < 3:
            ads_plan = (ads_plan * 3)[:3]

        first_prompt = ads_plan[0]["image_prompt"]

        img_resp = client.images.generate(
            model=IMAGE_MODEL,
            prompt=first_prompt,
            size=size,
            n=3,
        )

        images = img_resp.data
        if len(images) < 3:
            images = (images * 3)[:3]

        ads_out = []
        for idx in range(3):
            ad_plan = ads_plan[idx]
            img_item = images[idx]
            b64_data = getattr(img_item, "b64_json", None)

            ad_payload = {
                "headline": ad_plan["headline"],
                "copy": ad_plan["copy"],
            }
            if b64_data:
                ad_payload["image_base64"] = b64_data

            ads_out.append(ad_payload)

        return jsonify({"ads": ads_out, "size_used": size}), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal generation error", "details": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
