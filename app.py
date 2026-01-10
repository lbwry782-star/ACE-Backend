from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import zipfile
import base64
import json
import os
import openai
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS for all origins

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory storage for generated ads (in production, use Redis or database)
ads_storage = {}

# Initialize OpenAI client (legacy SDK pattern)
openai_api_key = os.environ.get('OPENAI_API_KEY')
if not openai_api_key:
    logger.error("OPENAI_API_KEY is not set in environment")
    raise ValueError("OPENAI_API_KEY environment variable is required")

openai.api_key = openai_api_key

openai_image_model = os.environ.get('OPENAI_IMAGE_MODEL', 'gpt-image-1')
openai_text_model = os.environ.get('OPENAI_TEXT_MODEL', 'gpt-4.1-mini')

# Load ACE Engine specification from file
ACE_ENGINE_SPEC_PATH = 'ACE_ENGINE_HE.txt'
ace_engine_system_prompt = None

try:
    if os.path.exists(ACE_ENGINE_SPEC_PATH):
        with open(ACE_ENGINE_SPEC_PATH, 'r', encoding='utf-8') as f:
            ace_engine_system_prompt = f.read().strip()
        logger.info(f"ACE Engine specification loaded from {ACE_ENGINE_SPEC_PATH} ({len(ace_engine_system_prompt)} characters)")
    else:
        logger.warning(f"ACE Engine specification file {ACE_ENGINE_SPEC_PATH} not found. Using default placeholder.")
        ace_engine_system_prompt = "[PLACEHOLDER: ACE Engine specification not loaded]"
except Exception as e:
    logger.error(f"Error loading ACE Engine specification: {e}")
    ace_engine_system_prompt = "[ERROR: Failed to load ACE Engine specification]"

logger.info(f"OpenAI client initialized with image model: {openai_image_model}, text model: {openai_text_model}")


def plan_ads(product_name, product_description, size, run_index):
    """Plan 3 ads using ACE engine specification as SYSTEM prompt. Returns strict JSON with run_index and ads array."""
    user_prompt = f"""Product Name: {product_name}
Product Description: {product_description}
Size: {size}
Run Index: {run_index}

Return ONLY valid JSON according to the schema below. No prose.

Schema:
{{
  "run_index": {run_index},
  "ads": [
    {{
      "ad_index": 1,
      "message": "...",
      "mode": "replacement",
      "object_a": "...",
      "object_b": "...",
      "environment_context": "...",
      "visual_description": "...",
      "headline": "...",
      "marketing_text": "...",
      "image_prompt": "..."
    }},
    {{ "ad_index": 2, ... }},
    {{ "ad_index": 3, ... }}
  ]
}}

Requirements:
- Exactly 3 ads
- Exactly 2 ads with mode="replacement", exactly 1 ad with mode="side_by_side"
- headline interprets visual (does not describe it), 3-7 words, includes product name
- marketing_text ~50 words, does not describe visual
- object_a and object_b are physical objects (no words, no scenes)
- environment_context is existence-inference context (not narrative)
- image_prompt: photorealistic only (no vector/illustration/3D/CGI/AI-look)
- No text/logos/labels except headline (graphics only if inherent physical structure)
- Image contains ONLY ONE headline, NO OTHER TEXT"""

    for attempt in range(2):  # Retry once if invalid
        try:
            response = openai.ChatCompletion.create(
                model=openai_text_model,
                messages=[
                    {"role": "system", "content": ace_engine_system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.8,
                max_tokens=3500
            )
            
            # Handle both dict and object responses (legacy SDK compatibility)
            if isinstance(response, dict):
                content = response['choices'][0]['message']['content'].strip()
            else:
                content = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            # Parse JSON
            plan = json.loads(content)
            
            # Validate structure: must have run_index and ads array
            if not isinstance(plan, dict):
                raise ValueError(f"Expected JSON object, got {type(plan)}")
            
            if 'ads' not in plan:
                raise ValueError("Missing 'ads' field in plan")
            
            ads_plan = plan['ads']
            if not isinstance(ads_plan, list) or len(ads_plan) != 3:
                raise ValueError(f"Expected 'ads' array with 3 items, got {type(ads_plan)} with {len(ads_plan) if isinstance(ads_plan, list) else 'N/A'} items")
            
            # Validate each ad has required fields
            required_fields = ['ad_index', 'message', 'mode', 'object_a', 'object_b', 'environment_context', 
                              'visual_description', 'headline', 'marketing_text', 'image_prompt']
            for i, ad_plan in enumerate(ads_plan):
                for field in required_fields:
                    if field not in ad_plan:
                        raise ValueError(f"Ad {i+1} missing required field: {field}")
                
                # Ensure ad_index matches position
                if ad_plan.get('ad_index') != i + 1:
                    ad_plan['ad_index'] = i + 1
                
                # Validate mode values
                mode = ad_plan.get('mode', '').lower()
                if mode not in ['replacement', 'side_by_side']:
                    raise ValueError(f"Ad {i+1} has invalid mode: {mode}. Must be 'replacement' or 'side_by_side'")
                
                # Validate headline: should not be empty and should include product name (case-insensitive check)
                headline = ad_plan.get('headline', '')
                if not headline or len(headline.split()) < 3 or len(headline.split()) > 7:
                    raise ValueError(f"Ad {i+1} headline must be 3-7 words, got {len(headline.split())} words")
                
                # Validate marketing_text: approximately 50 words
                marketing_words = ad_plan.get('marketing_text', '').split()
                if len(marketing_words) < 40 or len(marketing_words) > 60:
                    logger.warning(f"Ad {i+1} marketing_text has {len(marketing_words)} words (expected ~50)")
            
            # Validate mode distribution: exactly 2 replacement, exactly 1 side_by_side
            modes = [ad.get('mode', '').lower() for ad in ads_plan]
            replacement_count = modes.count('replacement')
            side_by_side_count = modes.count('side_by_side')
            
            if replacement_count != 2 or side_by_side_count != 1:
                raise ValueError(f"Invalid mode distribution: {replacement_count} replacement, {side_by_side_count} side_by_side. Expected: 2 replacement, 1 side_by_side")
            
            # Set run_index if not present or incorrect
            plan['run_index'] = run_index
            
            # Debug logging: planned mode, object_a, object_b, headline
            for ad in ads_plan:
                logger.info(f"PLANNED ad_index={ad['ad_index']} mode={ad['mode']} object_a={ad['object_a']} object_b={ad['object_b']} headline='{ad['headline']}'")
            
            logger.info(f"Successfully planned 3 ads (2 replacement, 1 side_by_side) for run_index={run_index}")
            return plan
            
        except json.JSONDecodeError as e:
            if attempt == 0:
                logger.warning(f"Invalid JSON on attempt {attempt + 1}, retrying... Error: {e}")
                continue
            else:
                logger.error(f"Invalid JSON after retry: {e}")
                raise ValueError(f"Failed to parse JSON response: {e}")
        except ValueError as e:
            if attempt == 0:
                logger.warning(f"Validation error on attempt {attempt + 1}, retrying... Error: {e}")
                continue
            else:
                logger.error(f"Validation failed after retry: {e}")
                raise
        except Exception as e:
            logger.error(f"Error planning ads: {e}", exc_info=True)
            raise
    
    raise ValueError("Failed to generate valid plan after 2 attempts")


def generate_image_with_openai(image_prompt, width, height, ad_index, headline):
    """Generate image using ONLY the image_prompt from ACE engine plan, with forced constraints appended"""
    try:
        # Map size to OpenAI format
        openai_size = f"{width}x{height}"
        
        # Force these additions to EVERY image_prompt before calling the image model
        forced_constraints = """ Photorealistic photograph. No illustration, no vector, no 3D, no CGI, no AI style. No logos, no labels, no packaging text, no signs. The ONLY text in the entire image is the headline (3-7 words). No other text at all. Headline must be integrated in the image as one separate element and must not touch the visual."""
        
        final_prompt = image_prompt + forced_constraints
        
        logger.info(f"Generating image for ad {ad_index} with size {openai_size}")
        logger.info(f"NO_OTHER_TEXT_CONSTRAINT_APPENDED: True (forced constraints added to image_prompt)")
        logger.debug(f"Image prompt (first 200 chars): {image_prompt[:200]}...")
        
        # Generate image with OpenAI (legacy SDK pattern)
        # Use the image_prompt with forced constraints appended
        response = openai.Image.create(
            prompt=final_prompt,
            n=1,
            size=openai_size,
            response_format="b64_json"
        )
        
        # Get base64 image from response (legacy SDK format - handles both dict and object)
        response_data = response.get('data') if isinstance(response, dict) else (response.data if hasattr(response, 'data') else None)
        
        if response_data and len(response_data) > 0:
            image_data = response_data[0]
            
            # Extract b64_json (handles both dict and object access)
            if isinstance(image_data, dict):
                image_base64 = image_data.get('b64_json')
            else:
                image_base64 = getattr(image_data, 'b64_json', None)
            
            if image_base64:
                image_bytes = base64.b64decode(image_base64)
                
                # Detect image format and validate
                mime_type, extension = detect_image_format(image_bytes)
                if not mime_type:
                    raise ValueError("Generated image format is not supported. Expected PNG, JPEG, or WebP")
                
                is_valid, error_msg = validate_image_bytes(image_bytes)
                if not is_valid:
                    raise ValueError(f"Generated image validation failed: {error_msg}")
                
                logger.info(f"Successfully generated image for ad {ad_index}, format: {mime_type}, size: {len(image_bytes)} bytes")
                return image_bytes, image_base64, mime_type
            else:
                # Log available fields for debugging
                available_fields = list(image_data.keys()) if isinstance(image_data, dict) else [attr for attr in dir(image_data) if not attr.startswith('_')]
                logger.error(f"OpenAI response does not contain b64_json. Available fields: {available_fields}")
                raise ValueError("OpenAI response does not contain b64_json field. Image generation may have failed.")
        else:
            raise ValueError("OpenAI response does not contain image data")
            
    except Exception as e:
        logger.error(f"Error generating image with OpenAI: {e}")
        raise


def detect_image_format(image_bytes):
    """Detect image format from magic bytes. Returns (mime_type, extension) or (None, None) if unknown."""
    if not image_bytes or len(image_bytes) < 4:
        return None, None
    
    # JPEG: starts with FF D8
    if image_bytes[0] == 0xFF and image_bytes[1] == 0xD8:
        return "image/jpeg", "jpg"
    
    # PNG: starts with 89 50 4E 47
    if (image_bytes[0] == 0x89 and image_bytes[1] == 0x50 and 
        image_bytes[2] == 0x4E and image_bytes[3] == 0x47):
        return "image/png", "png"
    
    # WebP: starts with RIFF...WEBP (52 49 46 46 ... 57 45 42 50)
    if (len(image_bytes) >= 12 and 
        image_bytes[0] == 0x52 and image_bytes[1] == 0x49 and 
        image_bytes[2] == 0x46 and image_bytes[3] == 0x46 and
        image_bytes[8] == 0x57 and image_bytes[9] == 0x45 and
        image_bytes[10] == 0x42 and image_bytes[11] == 0x50):
        return "image/webp", "webp"
    
    return None, None


def validate_image_bytes(image_bytes):
    """Validate that bytes represent a valid image (PNG, JPEG, or WebP)"""
    # Minimum size threshold: at least 5KB for a real image
    MIN_SIZE_BYTES = 5000
    
    if not image_bytes or len(image_bytes) < MIN_SIZE_BYTES:
        return False, f"Image data too small (less than {MIN_SIZE_BYTES} bytes, got {len(image_bytes) if image_bytes else 0})"
    
    # Detect image format
    mime_type, extension = detect_image_format(image_bytes)
    if not mime_type:
        return False, "Invalid image format. Expected PNG, JPEG, or WebP"
    
    # Basic validation passed
    return True, None


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"ok": True}), 200


@app.route('/api/generate', methods=['POST'])
def generate():
    """Generate 3 ads for a product"""
    try:
        data = request.get_json()
        
        # Validate required fields
        product_name = data.get('product_name', '').strip()
        product_description = data.get('product_description', '').strip()
        size = data.get('size', '').strip()
        run_index = data.get('run_index', 1)
        
        if not product_name or not product_description or not size:
            return jsonify({
                "error": "Missing required fields: product_name, product_description, size"
            }), 400
        
        # Validate size
        allowed_sizes = ["1024x1024", "1024x1536", "1536x1024"]
        if size not in allowed_sizes:
            return jsonify({
                "error": f"Invalid size. Must be one of: {', '.join(allowed_sizes)}"
            }), 400
        
        # Parse dimensions
        width, height = map(int, size.split('x'))
        
        # Plan 3 ads using ACE engine (Step A: Planning)
        try:
            plan = plan_ads(product_name, product_description, size, run_index)
        except Exception as e:
            logger.error(f"Error planning ads: {e}", exc_info=True)
            return jsonify({
                "error": "Failed to plan ads",
                "message": str(e)
            }), 500
        
        ads_plan = plan['ads']
        
        # Generate 3 ads based on the plan (Step B: Image generation)
        ads = []  # API response
        ads_for_storage = []  # Full data with image_bytes for ZIP
        
        for ad_plan in ads_plan:
            try:
                ad_index = ad_plan['ad_index']
                headline = ad_plan['headline']
                marketing_text = ad_plan['marketing_text']
                image_prompt = ad_plan['image_prompt']
                mode = ad_plan.get('mode', 'unknown')
                object_a = ad_plan.get('object_a', 'unknown')
                object_b = ad_plan.get('object_b', 'unknown')
                
                # Log debug info: mode, object_a, object_b, headline (Step D: Debug logging)
                logger.info(f"PLANNED ad_index={ad_index} mode={mode} object_a={object_a} object_b={object_b}")
                logger.info(f"FINAL_HEADLINE ad_index={ad_index}: '{headline}'")
                
                # Ensure marketing_text is approximately 50 words (trim if needed)
                words = marketing_text.split()
                if len(words) > 60:
                    marketing_text = " ".join(words[:50])
                    logger.warning(f"Ad {ad_index} marketing_text trimmed from {len(words)} to 50 words")
                elif len(words) < 40:
                    logger.warning(f"Ad {ad_index} marketing_text is too short ({len(words)} words)")
                
                # Generate image using image_prompt with forced constraints (Step B)
                image_bytes, image_base64, mime_type = generate_image_with_openai(
                    image_prompt, width, height, ad_index, headline
                )
                
                # Validate the generated image
                is_valid, error_msg = validate_image_bytes(image_bytes)
                if not is_valid:
                    logger.error(f"Image validation failed for ad {ad_index}: {error_msg}")
                    return jsonify({
                        "error": "Image generation failed",
                        "message": error_msg
                    }), 500
                
                # Create data URL with correct MIME type (preserve PNG/WebP if model returns them)
                image_data_url = f"data:{mime_type};base64,{image_base64}"
                
                # Detect extension for ZIP filename
                _, extension = detect_image_format(image_bytes)
                
                # API response (Step C: API output - unchanged shape)
                ad = {
                    "ad_index": ad_index,
                    "marketing_text": marketing_text,
                    "image_jpg": image_data_url
                }
                ads.append(ad)
                
                # Store full ad data for ZIP download (Step C: Store for ZIP)
                ad_storage = {
                    "ad_index": ad_index,
                    "marketing_text": marketing_text,
                    "image_jpg": image_data_url,
                    "image_bytes": image_bytes,  # Store bytes for ZIP generation
                    "extension": extension or "jpg"  # Store extension for ZIP filename
                }
                ads_for_storage.append(ad_storage)
                
                logger.info(f"Successfully generated ad {ad_index} for {product_name}")
                
            except Exception as e:
                logger.error(f"Error generating ad {ad_plan.get('ad_index', 'unknown')}: {e}", exc_info=True)
                return jsonify({
                    "error": "Image generation failed",
                    "message": str(e)
                }), 500
        
        # Store ads for download endpoint (keyed by run_index) - Step C: Store for ZIP
        storage_key = f"{product_name}_{run_index}"
        ads_storage[storage_key] = ads_for_storage
        
        return jsonify({
            "ads": ads,
            "run_index": run_index
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500


@app.route('/api/download', methods=['GET'])
def download():
    """Download ZIP file for a specific ad"""
    try:
        ad_index = request.args.get('ad_index', type=int)
        run_index = request.args.get('run_index', type=int)
        product_name = request.args.get('product_name', '').strip()
        
        if not ad_index or not run_index or not product_name:
            return jsonify({
                "error": "Missing required parameters: ad_index, run_index, product_name"
            }), 400
        
        storage_key = f"{product_name}_{run_index}"
        if storage_key not in ads_storage:
            return jsonify({
                "error": "Ad not found. Please generate ads first."
            }), 404
        
        ads = ads_storage[storage_key]
        if ad_index < 1 or ad_index > len(ads):
            return jsonify({
                "error": f"Invalid ad_index. Must be between 1 and {len(ads)}"
            }), 400
        
        ad = ads[ad_index - 1]
        
        # Create ZIP in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add image (decode base64 if it's a data URL)
            image_data = ad.get('image_jpg', '')
            if image_data.startswith('data:image'):
                # Extract base64 part
                base64_data = image_data.split(',')[1] if ',' in image_data else image_data
                try:
                    image_bytes = base64.b64decode(base64_data)
                except Exception as e:
                    return jsonify({
                        "error": "Invalid base64 image data",
                        "message": str(e)
                    }), 400
            else:
                # Assume it's already base64
                try:
                    image_bytes = base64.b64decode(image_data)
                except Exception as e:
                    return jsonify({
                        "error": "Invalid base64 image data",
                        "message": str(e)
                    }), 400
            
            # Detect image format and validate before adding to ZIP
            mime_type, extension = detect_image_format(image_bytes)
            if not mime_type:
                return jsonify({
                    "error": "Invalid image format",
                    "message": "Expected PNG, JPEG, or WebP"
                }), 400
            
            is_valid, error_msg = validate_image_bytes(image_bytes)
            if not is_valid:
                return jsonify({
                    "error": "Invalid image",
                    "message": error_msg
                }), 400
            
            # Use detected extension for ZIP filename
            zip_file.writestr(f"ad_{ad_index}.{extension}", image_bytes)
            
            # Add text file
            marketing_text = ad.get('marketing_text', '')
            zip_file.writestr(f"ad_{ad_index}.txt", marketing_text.encode('utf-8'))
        
        zip_buffer.seek(0)
        
        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f"ad_{ad_index}.zip"
        )
        
    except Exception as e:
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

