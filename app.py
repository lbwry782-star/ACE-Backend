import os
import json
import base64
import io
import zipfile
import time
import random
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from openai import RateLimitError, APIError, APIConnectionError, APITimeoutError

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


class RetryableError(Exception):
    """Exception raised when OpenAI call fails after max retries."""
    def __init__(self, message, code=429):
        self.message = message
        self.code = code
        super().__init__(self.message)


def retry_openai_call(call_func, max_retries=4, operation_name="OpenAI call"):
    """
    Retry wrapper for OpenAI calls with exponential backoff + jitter.
    Handles transient errors: 429, 500-599, timeouts.
    Returns the result on success, raises RetryableError on failure after max retries.
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return call_func()
        except (RateLimitError, APIConnectionError, APITimeoutError) as e:
            last_exception = e
            error_code = getattr(e, 'status_code', None) or getattr(e, 'code', None) or 429
            
            # Calculate wait time: exponential backoff + jitter
            base_wait = 2 ** attempt  # 1, 2, 4, 8 seconds
            jitter = random.uniform(0, 1)  # 0-1 second random jitter
            wait_time = base_wait + jitter
            
            if attempt < max_retries - 1:
                print(f"OPENAI_RETRY attempt={attempt + 1} wait={wait_time:.2f}s reason={type(e).__name__} code={error_code} operation={operation_name}")
                time.sleep(wait_time)
            else:
                print(f"OPENAI_RETRY attempt={attempt + 1} wait={wait_time:.2f}s reason={type(e).__name__} code={error_code} operation={operation_name} - MAX RETRIES REACHED")
        except APIError as e:
            # Check if it's a 5xx server error
            status_code = getattr(e, 'status_code', None)
            if status_code and 500 <= status_code < 600:
                last_exception = e
                base_wait = 2 ** attempt
                jitter = random.uniform(0, 1)
                wait_time = base_wait + jitter
                
                if attempt < max_retries - 1:
                    print(f"OPENAI_RETRY attempt={attempt + 1} wait={wait_time:.2f}s reason=APIError code={status_code} operation={operation_name}")
                    time.sleep(wait_time)
                else:
                    print(f"OPENAI_RETRY attempt={attempt + 1} wait={wait_time:.2f}s reason=APIError code={status_code} operation={operation_name} - MAX RETRIES REACHED")
            else:
                # Non-retryable API error (4xx except 429)
                raise
        except Exception as e:
            # Non-retryable error, re-raise immediately
            raise
    
    # If we get here, all retries failed
    error_code = getattr(last_exception, 'status_code', None) or getattr(last_exception, 'code', None) or 429
    raise RetryableError(f"OpenAI call failed after {max_retries} retries", error_code)


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

    headline_response = retry_openai_call(
        lambda: client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": "You are a creative advertising copywriter."},
                {"role": "user", "content": headline_prompt}
            ],
            temperature=0.9,
            max_tokens=50
        ),
        operation_name="headline_generation"
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

    marketing_response = retry_openai_call(
        lambda: client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": "You are a professional marketing copywriter. Always return exactly the requested word count."},
                {"role": "user", "content": marketing_prompt}
            ],
            temperature=0.8,
            max_tokens=200
        ),
        operation_name="marketing_text_generation"
    )
    
    marketing_text = marketing_response.choices[0].message.content.strip()
    
    # Ensure exactly 50 words
    words = marketing_text.split()
    if len(words) > 50:
        marketing_text = " ".join(words[:50])
    elif len(words) < 50:
        # Try to get more words from a follow-up request
        additional_prompt = f"""The previous marketing text was {len(words)} words. Add exactly {50 - len(words)} more words to complete it. The text so far: {marketing_text}"""
        additional_response = retry_openai_call(
            lambda: client.chat.completions.create(
                model=TEXT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a professional marketing copywriter. Always return exactly the requested word count."},
                    {"role": "user", "content": additional_prompt}
                ],
                temperature=0.8,
                max_tokens=100
            ),
            operation_name="marketing_text_additional"
        )
        additional_words = additional_response.choices[0].message.content.strip().split()
        all_words = words + additional_words
        marketing_text = " ".join(all_words[:50])
    
    return headline, marketing_text


def pick_two_objects(product_name, product_description, headline):
    """Select two physical objects (A and B), layout, and background using text model."""
    if not client:
        raise ValueError("OpenAI API key not configured")
    
    selection_prompt = f"""You are an ACE engine object selector. Select two physical objects for an advertisement.

Product Name: {product_name}
Product Description: {product_description}
Headline: {headline}

Rules:
1. Generate a list of 80 physical, real, associative objects based on the product and headline.
2. Objects must be simple, everyday, familiar physical objects — NOT ideas, symbols, abstract concepts, or illustrations.
3. Objects should be non-functional / not functionally linked.
4. Do NOT pick objects containing text/logos/letters/numbers/external graphics (unless inherent like playing cards, dice dots, engraved compass letters).

Selection:
- A = object with central meaning to the ad goal
- B = object used for conceptual emphasis (but pairing is still only by shape similarity)
- layout = "HYBRID" if A and B have strong shape similarity (can reach almost geometric overlap), otherwise "SIDE_BY_SIDE"
- background_classic_of_C = classic natural background for the dominant object (A's projection C)

Return ONLY valid JSON with these exact keys:
{{
  "A": "object name",
  "B": "object name",
  "layout": "HYBRID" or "SIDE_BY_SIDE",
  "background_classic_of_C": "description of classic natural background"
}}

Do not include any explanation or other text."""
    
    try:
        response = retry_openai_call(
            lambda: client.chat.completions.create(
                model=TEXT_MODEL,
                messages=[
                    {"role": "system", "content": "You are an ACE engine object selector. Always return valid JSON only."},
                    {"role": "user", "content": selection_prompt}
                ],
                temperature=0.8,
                response_format={"type": "json_object"}
            ),
            operation_name="object_selection"
        )
        
        result = json.loads(response.choices[0].message.content.strip())
        
        # Validate fields exist
        A = result.get("A", "")
        B = result.get("B", "")
        layout = result.get("layout", "SIDE_BY_SIDE")
        background_classic_of_C = result.get("background_classic_of_C", "")
        
        # Validate layout
        if layout not in ["HYBRID", "SIDE_BY_SIDE"]:
            layout = "SIDE_BY_SIDE"
        
        # Fallback if missing fields
        if not A or not B:
            A = "product object"
            B = "complementary object"
        
        return {
            "A": A,
            "B": B,
            "layout": layout,
            "background_classic_of_C": background_classic_of_C
        }
    except Exception as e:
        # Fallback on error
        print(f"Warning: Object selection failed: {str(e)}, using fallback")
        return {
            "A": "product object",
            "B": "complementary object",
            "layout": "SIDE_BY_SIDE",
            "background_classic_of_C": "natural background"
        }


def generate_image(product_name, product_description, headline, ad_size, attempt):
    """Generate image using OpenAI DALL-E with strict two-object enforcement."""
    if not client:
        raise ValueError("OpenAI API key not configured")
    
    # Map ad_size to OpenAI format
    size_map = {
        "1024x1024": "1024x1024",
        "1024x1536": "1024x1536",
        "1536x1024": "1536x1024"
    }
    openai_size = size_map.get(ad_size, "1024x1024")
    
    # Step 1: Pick two objects, layout, and background
    objects = pick_two_objects(product_name, product_description, headline)
    A = objects["A"]
    B = objects["B"]
    layout = objects["layout"]
    background_classic_of_C = objects["background_classic_of_C"]
    
    # Debug log: print selected A, B, layout, and openai_size (NOT full prompt, NOT secrets)
    print(f"Selected A: {A}")
    print(f"Selected B: {B}")
    print(f"Selected layout: {layout}")
    print(f"Selected ad_size: {ad_size} (OpenAI size: {openai_size})")
    print(f"Attempt: {attempt}")
    
    # Step 2: Build strict image prompt with explicit A/B/layout/background
    if layout == "HYBRID":
        layout_instruction = f"""Create a TRUE HYBRID: Object B's projection (D) is perfectly embedded into Object A's projection (C).
Present the HYBRID at an angle that maximizes both projections' visibility while keeping full photographic realism.
The objects must be physically fused or overlapped, NOT side-by-side."""
    else:  # SIDE_BY_SIDE
        layout_instruction = f"""Place Object A (C projection) and Object B (D projection) SIDE BY SIDE at the same angle.
Highlight maximal similar area between the projections.
Place them close together, emphasizing their shape similarity."""
    
    image_prompt = f"""YOU ARE A PROFESSIONAL ADVERTISING PHOTOGRAPHER.
YOU MUST FOLLOW ALL RULES BELOW. NO EXCEPTIONS.

MANDATORY OBJECTS (BOTH MUST APPEAR):
- Object A: {A}
- Object B: {B}
- YOU MUST SHOW BOTH OBJECT A AND OBJECT B IN THE IMAGE.
- NEVER show only one object. NEVER show only Object A. NEVER show only Object B.
- BOTH objects must be clearly visible and recognizable.

LAYOUT INSTRUCTION:
{layout_instruction}

BACKGROUND RULE:
- The background MUST be: {background_classic_of_C}
- This is the classic natural background of Object A (the dominant object C).
- NEVER use studio backgrounds, black backgrounds, gradients, or abstract scenes.

STYLE RULES:
- Ultra-realistic photography ONLY.
- Looks like a real camera photograph.
- Natural lighting, correct perspective, real materials.
- NO illustration, NO 3D render, NO CGI, NO AI-art look.

HEADLINE RULES (INSIDE IMAGE):
- The headline "{headline}" MUST appear INSIDE the image.
- The headline must be clearly readable and visually integrated into the composition.
- Place it above/below/next-to the objects (never on them).

CRITICAL ENFORCEMENT:
- YOU MUST SHOW BOTH OBJECT A ({A}) AND OBJECT B ({B}).
- The image MUST contain BOTH objects. Single-object images are FORBIDDEN.
- BOTH objects must be clearly visible and recognizable in the final image.

PRODUCT NAME: {product_name}
PRODUCT DESCRIPTION: {product_description}

TASK:
Generate ONE final advertising image that fully follows ALL the rules above.
The image MUST show BOTH Object A and Object B. Do not explain. Do not describe."""

    # Debug log: print image prompt, ad_size, and attempt
    print("=== IMAGE PROMPT SENT TO OPENAI ===")
    print(image_prompt)
    print("=== END IMAGE PROMPT ===")

    try:
        response = retry_openai_call(
            lambda: client.images.generate(
                model=IMAGE_MODEL,
                prompt=image_prompt,
                size=openai_size,
                quality="auto",
                n=1
            ),
            operation_name="image_generation"
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
    
    except RetryableError as e:
        # Transient OpenAI error after max retries - return 503
        return jsonify({"error": "RETRYABLE_ERROR", "code": e.code}), 503
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        # Handle external errors cleanly without stack traces
        return jsonify({"error": "Generation failed. Please try again."}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=False)

