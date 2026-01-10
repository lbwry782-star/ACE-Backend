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

logger.info(f"OpenAI client initialized with image model: {openai_image_model}, text model: {openai_text_model}")


def generate_headline_and_text(product_name, product_description, ad_index):
    """Generate headline and marketing text using OpenAI"""
    try:
        prompt = f"""Generate an advertising headline and marketing text for a product.

Product Name: {product_name}
Product Description: {product_description}

Requirements:
- Headline: 3-7 words, must include the product name "{product_name}", original and compelling
- Marketing text: Exactly 50 words (headline excluded), persuasive and informative

Return JSON format:
{{
    "headline": "the headline text here",
    "marketing_text": "exactly 50 words of marketing text here"
}}"""

        response = openai.ChatCompletion.create(
            model=openai_text_model,
            messages=[
                {"role": "system", "content": "You are an expert copywriter. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=300
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
        
        result = json.loads(content)
        headline = result.get("headline", f"{product_name}: Innovation You Need")
        marketing_text = result.get("marketing_text", f"Discover {product_name}: {product_description[:200]}")
        
        # Ensure marketing text is exactly 50 words
        words = marketing_text.split()
        if len(words) > 50:
            marketing_text = " ".join(words[:50])
        elif len(words) < 50:
            # Pad with product description if needed
            desc_words = product_description.split()
            needed = 50 - len(words)
            marketing_text = marketing_text + " " + " ".join(desc_words[:needed])
            marketing_text = " ".join(marketing_text.split()[:50])
        
        return headline, marketing_text
        
    except Exception as e:
        logger.error(f"Error generating headline/text: {e}")
        # Fallback
        headline = f"{product_name}: Innovation You Need"
        marketing_text = f"Discover {product_name}: {product_description[:200]}"
        words = marketing_text.split()[:50]
        marketing_text = " ".join(words)
        return headline, marketing_text


def generate_image_with_openai(headline, product_name, product_description, width, height, ad_index):
    """Generate a realistic photographic image using OpenAI with headline included"""
    try:
        # Map size to OpenAI format
        openai_size = f"{width}x{height}"
        
        # Build image prompt that includes the headline
        image_prompt = f"""Create a realistic, professional photographic advertisement image.

Product: {product_name}
Description: {product_description}

Image Requirements:
- Realistic photographic style (not illustration, not cartoon, not graphic design)
- Professional product photography quality
- The headline text "{headline}" must appear clearly inside the image, integrated naturally into the composition
- Headline should be placed on a background area, not overlapping the main product/subject
- Keep all text and important elements at least 8-12% away from all edges (safe margins)
- High quality, sharp, well-lit, professional composition
- Size: {width}x{height} pixels

The image should be a complete, realistic photograph suitable for advertising."""

        logger.info(f"Generating image for ad {ad_index} with size {openai_size}")
        
        # Generate image with OpenAI (legacy SDK pattern)
        # Legacy SDK Image.create: prompt, n, size, response_format (if supported)
        # Note: legacy SDK may not support 'model' parameter in Image.create
        response = openai.Image.create(
            prompt=image_prompt,
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
        
        # Generate 3 ads using OpenAI
        ads = []
        for ad_index in range(1, 4):
            try:
                # Generate headline and marketing text
                headline, marketing_text = generate_headline_and_text(product_name, product_description, ad_index)
                
                # Generate image with OpenAI (includes headline in image)
                image_bytes, image_base64, mime_type = generate_image_with_openai(
                    headline, product_name, product_description, width, height, ad_index
                )
                
                # Validate the generated image
                is_valid, error_msg = validate_image_bytes(image_bytes)
                if not is_valid:
                    logger.error(f"Image validation failed for ad {ad_index}: {error_msg}")
                    return jsonify({
                        "error": "Image generation failed",
                        "message": error_msg
                    }), 500
                
                # Create data URL with correct MIME type
                image_data_url = f"data:{mime_type};base64,{image_base64}"
                
                ad = {
                    "ad_index": ad_index,
                    "marketing_text": marketing_text,
                    "image_jpg": image_data_url
                }
                ads.append(ad)
                
                logger.info(f"Successfully generated ad {ad_index} for {product_name}")
                
            except Exception as e:
                logger.error(f"Error generating ad {ad_index}: {e}", exc_info=True)
                return jsonify({
                    "error": "Image generation failed",
                    "message": str(e)
                }), 500
        
        # Store ads for download endpoint (keyed by run_index)
        storage_key = f"{product_name}_{run_index}"
        ads_storage[storage_key] = ads
        
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

