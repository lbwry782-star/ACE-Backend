from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import zipfile
import base64
import json
import os
from openai import OpenAI
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS for all origins

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory storage for generated ads (in production, use Redis or database)
ads_storage = {}

# Initialize OpenAI client
openai_api_key = os.environ.get('OPENAI_API_KEY')
if not openai_api_key:
    logger.error("OPENAI_API_KEY is not set in environment")
    raise ValueError("OPENAI_API_KEY environment variable is required")

openai_image_model = os.environ.get('OPENAI_IMAGE_MODEL', 'gpt-image-1')
openai_text_model = os.environ.get('OPENAI_TEXT_MODEL', 'gpt-4.1-mini')

client = OpenAI(api_key=openai_api_key)
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

        response = client.chat.completions.create(
            model=openai_text_model,
            messages=[
                {"role": "system", "content": "You are an expert copywriter. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=300
        )
        
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
        
        # Generate image with OpenAI (gpt-image-1 returns base64 by default)
        response = client.images.generate(
            model=openai_image_model,
            prompt=image_prompt,
            size=openai_size,
            quality="auto"
        )
        
        # Get base64 image from response
        if hasattr(response, 'data') and len(response.data) > 0:
            image_data = response.data[0]
            # Check for b64_json field (base64 encoded image)
            if hasattr(image_data, 'b64_json') and image_data.b64_json:
                image_base64 = image_data.b64_json
                image_bytes = base64.b64decode(image_base64)
                
                # Validate the image
                is_valid, error_msg = validate_jpeg_bytes(image_bytes)
                if not is_valid:
                    raise ValueError(f"Generated image validation failed: {error_msg}")
                
                logger.info(f"Successfully generated image for ad {ad_index}, size: {len(image_bytes)} bytes")
                return image_bytes, image_base64
            else:
                # Log available fields for debugging
                available_fields = [attr for attr in dir(image_data) if not attr.startswith('_')]
                logger.error(f"OpenAI response does not contain b64_json. Available fields: {available_fields}")
                raise ValueError("OpenAI response does not contain b64_json field. Image generation may have failed.")
        else:
            raise ValueError("OpenAI response does not contain image data")
            
    except Exception as e:
        logger.error(f"Error generating image with OpenAI: {e}")
        raise


def validate_jpeg_bytes(image_bytes):
    """Validate that bytes represent a valid JPEG image"""
    # Minimum size threshold: at least 5KB for a real image
    MIN_SIZE_BYTES = 5000
    
    if not image_bytes or len(image_bytes) < MIN_SIZE_BYTES:
        return False, f"Image data too small (less than {MIN_SIZE_BYTES} bytes, got {len(image_bytes) if image_bytes else 0})"
    
    # Check JPEG magic bytes
    if len(image_bytes) < 2 or not (image_bytes[0] == 0xFF and image_bytes[1] == 0xD8):
        return False, "Invalid JPEG magic bytes (not starting with FF D8)"
    
    # Check JPEG end marker
    if len(image_bytes) < 2 or not (image_bytes[-2] == 0xFF and image_bytes[-1] == 0xD9):
        return False, "Invalid JPEG end marker (not ending with FF D9)"
    
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
                image_bytes, image_base64 = generate_image_with_openai(
                    headline, product_name, product_description, width, height, ad_index
                )
                
                # Validate the generated image
                is_valid, error_msg = validate_jpeg_bytes(image_bytes)
                if not is_valid:
                    logger.error(f"Image validation failed for ad {ad_index}: {error_msg}")
                    return jsonify({
                        "error": "Image generation failed",
                        "message": error_msg
                    }), 500
                
                # Create data URL
                image_data_url = f"data:image/jpeg;base64,{image_base64}"
                
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
            
            # Validate JPEG before adding to ZIP
            is_valid, error_msg = validate_jpeg_bytes(image_bytes)
            if not is_valid:
                return jsonify({
                    "error": "Invalid JPEG image",
                    "message": error_msg
                }), 400
            
            zip_file.writestr(f"ad_{ad_index}.jpg", image_bytes)
            
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

