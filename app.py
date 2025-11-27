import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)

frontend_url = os.environ.get("FRONTEND_URL")
if frontend_url:
    CORS(app, resources={r"/*": {"origins": [frontend_url]}})
else:
    CORS(app)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

TEXT_MODEL = os.environ.get("OPENAI_TEXT_MODEL", "gpt-4.1-mini")
IMAGE_MODEL = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1")

ALLOWED_SIZES = {"1024x1024", "1024x1536", "1536x1024"}


@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


def build_copy_prompt(product: str, description: str) -> str:
    return (
        "You are the ACE advertising engine. Write 3 English ads.\n"
        "For each ad you must output:\n"
        "- HEADLINE: 3-7 English words\n"
        "- COPY: exactly 50 English words\n\n"
        f"Product: {product}\n"
        f"Description: {description}\n\n"
        "Output format (plain text, no JSON):\n"
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

    if not ads:
        ads.append(
            {
                "headline": "ACE Hybrid Ad",
                "copy": raw_text.strip(),
            }
        )

    while len(ads) < 3:
        ads.append(ads[-1])

    return ads


def generate_copies(product: str, description: str):
    prompt = build_copy_prompt(product, description)
    response = client.responses.create(
        model=TEXT_MODEL,
        input=prompt,
        max_output_tokens=700,
    )

    text = None

    try:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            text = output_text
    except Exception:
        text = None

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

    if not text:
        text = str(response)

    return parse_ads(text)


def build_image_prompt(product: str, description: str, headline: str) -> str:
    return (
        "Create a hyper-realistic photographic advertising image.\n\n"
        "CONCEPT STAGE (ALREADY DONE BY TEXT ENGINE):\n"
        "- Assume that 50 conceptual associations related to the product and its goal have already been selected.\n"
        "- From these, you now work ONLY with visual 2D shape categories.\n\n"
        "2D SHAPE CATEGORIES:\n"
        "- Circle / ellipse\n"
        "- Square / rectangle\n"
        "- Triangle / trapezoid\n"
        "- Elongated organic figure (person, animal, bottle, vehicle etc.)\n"
        "- Ring / frame / hollow shape\n"
        "- Complex shape (cluster) if needed.\n\n"
        "SHAPE-FIRST PAIR SELECTION (OBJECT A and OBJECT B):\n"
        "- Select TWO real-world objects, A and B, from the conceptual pool ONLY by 2D silhouette similarity, not by meaning.\n"
        "- A and B must belong to the SAME 2D shape category.\n"
        "- Match:\n"
        "  • height/width ratio\n"
        "  • overall contour\n"
        "  • dominant axis (vertical / horizontal / diagonal)\n"
        "  • internal structure (center, ring, fill etc.).\n"
        "- They should be interchangeable in the same frame window: swapping A with B keeps almost the same silhouette, and differences are mostly in texture and material.\n\n"
        "FULL OBJECTS AND NATURAL BACKGROUND:\n"
        "- Never crop A or B at the frame edges.\n"
        "- Each object must appear as a complete object, fully inside the image boundaries.\n"
        "- Object A has its natural environment (classroom, kitchen, road, office, etc.).\n"
        "- Build the hybrid scene as follows:\n"
        "  1) Use the natural background of object A (its environment).\n"
        "  2) Remove A as a visible object.\n"
        "  3) Place object B as a complete, intact object inside A's background.\n"
        "- Viewers can still understand that this is 'A's world' by the context of the background.\n"
        "- No half-objects: an object is either fully present or not present at all.\n\n"
        "FULL SWAP RULE (NO HALF-AND-HALF BODIES):\n"
        "- Do NOT create a half-A half-B fused body.\n"
        "- Do NOT slice objects in the middle.\n"
        "- Instead, use a full swap logic: background of A + full B placed into that background.\n"
        "- You may softly adjust lighting and shadows around B so it fits the scene, but do not distort its silhouette.\n\n"
        "COMPOSITION AND MARGINS:\n"
        "- Place B as the main central object (or slightly off-center) inside the frame.\n"
        "- Keep at least 10% margin from every image edge to the main object, so nothing important is cut.\n"
        "- Minimal, clean photographic composition: no extra props or clutter beyond what is needed in A's natural environment.\n\n"
        "HEADLINE PLACEMENT (TEXT ON IMAGE):\n"
        f"- Embed this headline once: \"{headline}\"\n"
        "- Place the headline NEXT TO the hybrid object, outside its silhouette.\n"
        "- The headline must never overlap, cover, touch or cross the object.\n"
        "- Use bold, high-contrast typography so it is easy to read at first glance.\n\n"
        "PRODUCT CONTEXT:\n"
        f"- Product: {product}\n"
        f"- Description: {description or 'Use common-sense assumptions.'}\n\n"
        "Generate one final image only."
    )


def generate_images_for_ads(product: str, description: str, ads, size: str):
    results = []
    target_size = size if size in ALLOWED_SIZES else "1024x1024"

    for ad in ads:
        headline = (ad.get("headline") or "ACE Ad").strip()
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
