
import os
import json
import hmac
import hashlib
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)
CORS(app)

# =========================
#  OpenAI Client / Models
# =========================

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
IMAGE_MODEL = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1")
TEXT_MODEL = os.environ.get("OPENAI_TEXT_MODEL", "gpt-4.1-mini")

# =========================
#  Simple In-Memory Token Store (for demo / single-instance)
#  In production you would use DB/Redis instead.
#  Structure:
#  TOKENS = {
#      "cp_value": {
#          "status": "unused" | "used" | "in_progress" | "failed_once"
#      }
#  }
# =========================

TOKENS = {}

def get_token(cp: str):
    if not cp:
        return None
    return TOKENS.get(cp)

def set_token(cp: str, status: str):
    if not cp:
        return
    TOKENS[cp] = {"status": status}

# =========================
#  IPN / Webhook from ICOUNT
#  POST /ipn
#  This endpoint is called by iCount AFTER a successful payment.
#  It should:
#    1) Verify signature (if IPN_SECRET is configured)
#    2) If payment is PAID -> create or update token as "unused"
# =========================

IPN_SECRET = os.environ.get("IPN_SECRET", "").encode("utf-8") if os.environ.get("IPN_SECRET") else None

def verify_ipn_signature(raw_body: bytes, signature: str) -> bool:
    """Optional HMAC verification if IPN_SECRET is set.
    If no secret is set, this function returns True (no verification).
    """
    if not IPN_SECRET:
        # No secret configured -> accept (demo mode)
        return True
    if not signature:
        return False
    digest = hmac.new(IPN_SECRET, raw_body, hashlib.sha256).hexdigest()
    # Use constant-time comparison
    return hmac.compare_digest(digest, signature)

@app.route("/ipn", methods=["POST"])
def ipn():
    # Read raw body for signature verification
    raw_body = request.get_data() or b""
    signature = request.headers.get("X-ICOUNT-SIGNATURE", "")

    if not verify_ipn_signature(raw_body, signature):
        return jsonify({"error": "Invalid IPN signature"}), 400

    # Try to parse JSON first, then form-encoded
    data = {}
    try:
        if request.is_json:
            data = request.get_json(silent=True) or {}
        else:
            data = request.form.to_dict()
    except Exception:
        data = {}

    status = str(data.get("status", "")).lower()
    cp = str(data.get("cp", "")).strip()  # iCount internal reference / cp parameter

    if status == "paid" and cp:
        # Create or reset token for this cp
        set_token(cp, "unused")
        return jsonify({"ok": True, "cp": cp, "token_status": "unused"}), 200

    # For non-paid or missing cp we simply acknowledge
    return jsonify({"ok": True, "ignored": True}), 200

# =========================
#  Token Status Endpoint
#  GET /token-status?cp=...
#  Frontend can call this to know if Generate is allowed.
# =========================

@app.route("/token-status", methods=["GET"])
def token_status():
    cp = request.args.get("cp", "").strip()
    token = get_token(cp)
    if not token:
        return jsonify({
            "allowed": False,
            "status": "no_token",
            "remaining_generations": 0
        }), 200

    status = token.get("status", "no_token")
    if status in ("unused", "failed_once"):
        return jsonify({
            "allowed": True,
            "status": status,
            "remaining_generations": 1
        }), 200
    else:
        # used or in_progress
        return jsonify({
            "allowed": False,
            "status": status,
            "remaining_generations": 0
        }), 200

# =========================
#  ACE ENGINE — Text + Image
# =========================

ACE_ENGINE_RULES = """You are ACE ENGINE v5.0 — an advertising creation engine.
Follow ALL of these rules strictly when creating ads:

H00 — Audience is inferred ONLY from product name and product description.
H01 — Create three different ads per product, each with a different advertising goal.
H02 — For each ad, internally imagine 80 PHYSICAL photographic associations (objects only, no abstract concepts).
H03 — Choose two physical objects: A = core meaning, B = semantic reinforcement.
H04 — First decide the view / projection (camera angle), then choose A and B that match that view.
H05 — Similarity logic between A and B shapes (in that view):
       High similarity  -> HYBRID (A and B fused into one scene)
       Medium similarity -> SIDE-BY-SIDE (A and B shown separately in one scene)
       Low similarity -> REJECT the pair and choose new A and B.
H06 — In HYBRID composition, the background scene always belongs to A
       and B is blended into A's form, without breaking realism.
H07 — HEADLINE: 3–7 words only, MUST contain the product name exactly once.
H08 — 50-word copy is for Social / Landing Page ONLY. It must NEVER appear as text on the image.
H09 — Visual style: strictly PHOTOREALISTIC photography only.
       NO illustration, NO icon, NO vector, NO 3D, NO "AI style".
H10 — Poster contains only VISUAL + HEADLINE. No long text inside the image.

TE01 — NO marketing text is allowed inside the image itself.
       The only text allowed on the poster is a single short headline.
"""


def build_text_prompt(product: str, description: str, ad_index: int) -> str:
    return f"""{ACE_ENGINE_RULES}

PRODUCT NAME: {product}
PRODUCT DESCRIPTION: {description}

You are now creating AD #{ad_index} out of 3.
For this specific ad:
1) Infer a clear advertising goal that is different from the other two ads.
2) Infer the target audience (age, lifestyle, pains, needs, familiarity with the category).
3) Choose two PHYSICAL objects A and B that express the goal, following the similarity rules.
4) Decide if the final scene is HYBRID or SIDE-BY-SIDE, but do NOT use these words in the image description.
5) Design one photorealistic scene that would be used as a poster background.

Return a JSON object with EXACTLY this structure and field names:

{{
  "headline": "string, 3–7 words, MUST contain the product name exactly once",
  "copy_50_words": "exactly 50 words of marketing copy, for social/landing only, not for the image",
  "image_prompt": "a detailed, single-scene description for a PHOTOREALISTIC photo. 
                   Describe how the two objects appear and interact according to the rules.
                   DO NOT mention 'text', 'headline', 'poster', 'layout', 'A', 'B', 'hybrid' or 'side-by-side'
                   and DO NOT ask for any writing inside the image."
}}

Important:
- The copy_50_words field MUST contain exactly 50 words.
- The headline MUST be 3–7 words and include the product name.
- The image_prompt MUST obey all visual rules above.
"""


def call_ace_text(product: str, description: str, ad_index: int):
    prompt = build_text_prompt(product, description, ad_index)
    chat = client.chat.completions.create(
        model=TEXT_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a precise JSON-only ACE ENGINE text generator."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.9,
    )
    content = chat.choices[0].message.content
    data = json.loads(content)
    headline = data.get("headline", "").strip()
    copy_50 = data.get("copy_50_words", "").strip()
    image_prompt = data.get("image_prompt", "").strip()
    return headline, copy_50, image_prompt


def call_ace_image(image_prompt: str, size: str):
    img = client.images.generate(
        model=IMAGE_MODEL,
        prompt=(
            image_prompt
            + "\nPhotorealistic photography, no text, no UI, no frames, no extra graphic elements. "
              "No logos, no watermarks. Only the described physical objects and a realistic background."
        ),
        size=size,
        n=1,
    )
    return img.data[0].b64_json


def generate_single_ad(product: str, description: str, size: str, index: int):
    headline, copy_50, image_prompt = call_ace_text(product, description, index)
    image_b64 = call_ace_image(image_prompt, size)
    return {
        "headline": headline,
        "copy": copy_50,
        "image_base64": image_b64,
    }

# =========================
#  Health
# =========================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

# =========================
#  Authorization Helper
# =========================

DEV_OVERRIDE_SECRET = os.environ.get("ACE_DEV_SECRET", "")

def determine_mode(cp: str, override_secret: str):
    """Determine if the call is override, token, or blocked.
    Returns: 'override' | 'token' | ''
    """
    # Override mode (internal dev only, not exposed in UI)
    if DEV_OVERRIDE_SECRET and override_secret and override_secret == DEV_OVERRIDE_SECRET:
        return "override"

    # Token mode based on cp / iCount reference
    token = get_token(cp)
    if token and token.get("status") in ("unused", "failed_once"):
        return "token"

    return ""

# =========================
#  Generate Endpoint
# =========================

@app.route("/generate", methods=["POST"])
def generate():
    try:
        payload = request.get_json(force=True, silent=False)
    except Exception:
        return jsonify({"error": "Invalid JSON body"}), 400

    product = (payload or {}).get("product", "").strip()
    description = (payload or {}).get("description", "").strip()
    size = (payload or {}).get("size", "1024x1024").strip()
    cp = (payload or {}).get("cp", "").strip() or request.args.get("cp", "").strip()
    override_secret = (payload or {}).get("dev_secret", "").strip()

    if not product or not description:
        return jsonify({"error": "Missing product or description"}), 400

    if size not in {"1024x1024", "1024x1792", "1792x1024"}:
        return jsonify({"error": "Unsupported size. Use OpenAI generic sizes only."}), 400

    mode = determine_mode(cp, override_secret)
    if not mode:
        return jsonify({"error": "Generation not allowed. No valid token or override."}), 403

    # If token mode, mark as in_progress to avoid double-click race
    if mode == "token":
        set_token(cp, "in_progress")

    try:
        ads = []
        for i in range(1, 4):
            ad = generate_single_ad(product, description, size, i)
            ads.append(ad)

        # Mark token as used on successful generation
        if mode == "token":
            set_token(cp, "used")

        return jsonify({"mode": mode, "ads": ads}), 200

    except Exception as e:
        # On failure in TOKEN mode we allow one more attempt:
        if mode == "token":
            previous = get_token(cp)
            prev_status = (previous or {}).get("status")
            # if it was in_progress -> set to failed_once (one more allowed)
            if prev_status in (None, "in_progress", "unused"):
                set_token(cp, "failed_once")
            else:
                # already failed_before? then mark used to prevent loops
                set_token(cp, "used")

        return jsonify({"error": "Generation failed", "details": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
