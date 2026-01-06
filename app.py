import os
import base64
import io
import zipfile
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)
CORS(app)  # Enable CORS for all origins (Phase 1)

# Valid ad sizes
VALID_AD_SIZES = {"1024x1024", "1024x1536", "1536x1024"}

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TEXT_MODEL = os.getenv("TEXT_MODEL", "gpt-4.1-mini")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gpt-image-1")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

# Initialize OpenAI client
if not OPENAI_API_KEY or not OPENAI_API_KEY.strip():
    client = None
    print("WARNING: OPENAI_API_KEY is not set or is empty")
else:
    # Only set base_url if explicitly provided and non-empty
    client_kwargs = {"api_key": OPENAI_API_KEY}
    if OPENAI_BASE_URL and OPENAI_BASE_URL.strip():
        client_kwargs["base_url"] = OPENAI_BASE_URL
    client = OpenAI(**client_kwargs)
    print("INFO: OPENAI_API_KEY is present (client initialized)")


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"ok": True}), 200


def generate_headline_and_text(product_name, product_description, attempt):
    """Generate headline and exactly 50-word marketing text using OpenAI."""
    if not client:
        raise ValueError("OpenAI API key not configured")
    
    # Generate headline: 3-7 words, includes product_name, original (not copied from description)
    headline_prompt = f"""Generate a creative advertising headline for a product. Requirements:
- Must be exactly 3-7 words
- Must include the product name: "{product_name}"
- Must be original and NOT a quote or variation of the product description
- Must be a compelling promise or benefit statement
- Product description (for context only, do not copy): {product_description}

Return ONLY the headline, nothing else."""

    headline_response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {"role": "system", "content": "You are a creative advertising copywriter."},
            {"role": "user", "content": headline_prompt}
        ],
        temperature=0.9,
        max_tokens=50
    )
    
    headline = headline_response.choices[0].message.content.strip()
    
    # Ensure headline is 3-7 words
    headline_words = headline.split()
    if len(headline_words) < 3:
        headline = f"{product_name} transforms everything"
    elif len(headline_words) > 7:
        headline = " ".join(headline_words[:7])
    
    # Generate marketing text: exactly 50 words (headline excluded)
    marketing_prompt = f"""Generate marketing text for an advertisement. Requirements:
- Must be EXACTLY 50 words (count carefully)
- Must be based on the product: {product_name}
- Product description: {product_description}
- Headline (already used, do not repeat): {headline}
- Must be compelling, professional, and persuasive
- Do not include the headline in the word count

Return ONLY the marketing text, exactly 50 words."""

    marketing_response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {"role": "system", "content": "You are a professional marketing copywriter. Always return exactly the requested word count."},
            {"role": "user", "content": marketing_prompt}
        ],
        temperature=0.8,
        max_tokens=200
    )
    
    marketing_text = marketing_response.choices[0].message.content.strip()
    
    # Ensure exactly 50 words
    words = marketing_text.split()
    if len(words) > 50:
        marketing_text = " ".join(words[:50])
    elif len(words) < 50:
        # Try to get more words from a follow-up request
        additional_prompt = f"""The previous marketing text was {len(words)} words. Add exactly {50 - len(words)} more words to complete it. The text so far: {marketing_text}"""
        additional_response = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": "You are a professional marketing copywriter. Always return exactly the requested word count."},
                {"role": "user", "content": additional_prompt}
            ],
            temperature=0.8,
            max_tokens=100
        )
        additional_words = additional_response.choices[0].message.content.strip().split()
        all_words = words + additional_words
        marketing_text = " ".join(all_words[:50])
    
    return headline, marketing_text


def generate_image(product_name, product_description, headline, ad_size, attempt):
    """Generate image using OpenAI DALL-E."""
    if not client:
        raise ValueError("OpenAI API key not configured")
    
    # Map ad_size to OpenAI format
    # OpenAI DALL-E supports: 1024x1024, 1024x1536, 1536x1024
    # We map our sizes to the closest OpenAI-supported sizes
    size_map = {
        "1024x1024": "1024x1024",
        "1024x1536": "1024x1536",  # Portrait: use closest supported size
        "1536x1024": "1536x1024"   # Landscape: use closest supported size
    }
    openai_size = size_map.get(ad_size, "1024x1024")
    
    # Generate image prompt based on the spec requirements
    # For Phase 2, we'll create a simple prompt that includes the headline visually
    image_prompt = f"""
YOU ARE A PROFESSIONAL ADVERTISING PHOTOGRAPHER.
YOU MUST FOLLOW ALL RULES BELOW. NO EXCEPTIONS.

CORE ACE ENGINE RULES (MANDATORY):
1. The image MUST depict TWO DISTINCT PHYSICAL OBJECTS (A and B).
2. Objects A and B MUST be REAL, TANGIBLE, PHYSICAL OBJECTS — not concepts or symbols.
3. Objects A and B MUST have REAL SHAPE SIMILARITY (projection-based similarity).
4. If shape similarity is strong, YOU MUST create a TRUE HYBRID OBJECT
   (physical fusion or overlap, not side-by-side).
5. If shape similarity is insufficient, place the objects SIDE BY SIDE.
6. NEVER show only one object.
7. NEVER use abstract, metaphorical, symbolic, or conceptual visuals.

BACKGROUND RULE:
- The background MUST be the CLASSIC NATURAL BACKGROUND of the dominant object (C).
- NEVER use studio backgrounds, black backgrounds, gradients, or abstract scenes.

STYLE RULES:
- Ultra-realistic photography ONLY.
- Looks like a real camera photograph.
- Natural lighting, correct perspective, real materials.
- NO illustration, NO 3D render, NO CGI, NO AI-art look.

HEADLINE RULES (INSIDE IMAGE):
- A short headline of 3–7 words MUST appear INSIDE the image.
- The headline MUST include the PRODUCT NAME.
- The headline is NOT a quote and NOT the product description.
- The headline must be clearly readable and visually dominant.

COMPOSITION:
- Professional commercial advertising composition.
- Clear subject focus.
- Correct depth, lighting, and proportions.

PRODUCT NAME:
{product_name}

PRODUCT DESCRIPTION:
{product_description}

HEADLINE TO DISPLAY:
{headline}

TASK:
Generate ONE final advertising image that fully follows ALL the rules above.
Do not explain. Do not describe. Do not add text outside the image.
"""

    # Debug log: print image prompt, ad_size, and attempt
    print("=== IMAGE PROMPT SENT TO OPENAI ===")
    print(image_prompt)
    print("=== END IMAGE PROMPT ===")
    print(f"Selected ad_size: {ad_size}")
    print(f"Attempt: {attempt}")

    try:
        response = client.images.generate(
            model=IMAGE_MODEL,
            prompt=image_prompt,
            size=openai_size,
            quality="auto",
            n=1
        )
        
        # Debug: log which keys exist in the response
        response_keys = list(response.data[0].__dict__.keys()) if hasattr(response.data[0], '__dict__') else []
        print(f"OpenAI image response keys: {response_keys}")
        
        # Use b64_json from OpenAI response (no URL fetch needed)
        image_base64 = response.data[0].b64_json
        
        return f"data:image/jpeg;base64,{image_base64}"
    except Exception as e:
        # Handle external errors cleanly
        raise ValueError(f"Image generation failed: {str(e)}")


def create_zip_in_memory(image_base64, marketing_text, attempt):
    """Create a ZIP file in memory containing the image and text file."""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Decode image from base64 and add to ZIP
        image_data = base64.b64decode(image_base64.split(',')[1] if ',' in image_base64 else image_base64)
        zip_file.writestr(f"ad_{attempt}.jpg", image_data)
        
        # Add marketing text file
        zip_file.writestr(f"ad_{attempt}.txt", marketing_text)
    
    zip_buffer.seek(0)
    zip_base64 = base64.b64encode(zip_buffer.getvalue()).decode('utf-8')
    
    return zip_base64


@app.route('/generate', methods=['POST'])
def generate():
    """Generate a single ad with real OpenAI generation (Phase 2)."""
    # Get JSON body
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400
    
    # Validate required fields
    product_name = data.get("product_name")
    product_description = data.get("product_description")
    ad_size = data.get("ad_size")
    attempt = data.get("attempt")
    
    # Check all fields are present
    if not product_name:
        return jsonify({"error": "product_name is required and must be a non-empty string"}), 400
    
    if not product_description:
        return jsonify({"error": "product_description is required and must be a non-empty string"}), 400
    
    if not ad_size:
        return jsonify({"error": "ad_size is required and must be a non-empty string"}), 400
    
    if attempt is None:
        return jsonify({"error": "attempt is required and must be 1, 2, or 3"}), 400
    
    # Check all fields are strings (except attempt)
    if not isinstance(product_name, str) or not product_name.strip():
        return jsonify({"error": "product_name must be a non-empty string"}), 400
    
    if not isinstance(product_description, str) or not product_description.strip():
        return jsonify({"error": "product_description must be a non-empty string"}), 400
    
    if not isinstance(ad_size, str) or not ad_size.strip():
        return jsonify({"error": "ad_size must be a non-empty string"}), 400
    
    # Validate attempt
    if not isinstance(attempt, int) or attempt not in [1, 2, 3]:
        return jsonify({"error": "attempt must be exactly 1, 2, or 3"}), 400
    
    # Validate ad_size is one of the allowed values
    if ad_size not in VALID_AD_SIZES:
        return jsonify({
            "error": f"ad_size must be exactly one of: {', '.join(sorted(VALID_AD_SIZES))}"
        }), 400
    
    # Check OpenAI API key - fail fast with clear error
    if not OPENAI_API_KEY or not OPENAI_API_KEY.strip() or not client:
        return jsonify({"error": "Server misconfigured: OPENAI_API_KEY is not set"}), 500
    
    try:
        # Generate headline and marketing text
        headline, marketing_text = generate_headline_and_text(product_name, product_description, attempt)
        
        # Generate image
        image_data_url = generate_image(product_name, product_description, headline, ad_size, attempt)
        
        # Extract base64 from data URL for ZIP creation
        image_base64 = image_data_url.split(',')[1] if ',' in image_data_url else image_data_url
        
        # Create ZIP in memory
        zip_base64 = create_zip_in_memory(image_base64, marketing_text, attempt)
        
        # Return single ad object
        ad = {
            "ad_id": attempt,
            "headline": headline,
            "marketing_text_50_words": marketing_text,
            "image_data_url": image_data_url,
            "zip_base64": zip_base64,
            "zip_filename": f"ad_{attempt}.zip"
        }
        
        return jsonify(ad), 200
    
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        # Handle external errors cleanly without stack traces
        return jsonify({"error": "Generation failed. Please try again."}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=False)

