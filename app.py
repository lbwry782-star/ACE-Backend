from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import zipfile
import base64
import json
import os
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)
CORS(app)  # Enable CORS for all origins

# In-memory storage for generated ads (in production, use Redis or database)
ads_storage = {}


def generate_placeholder_image(width, height, ad_index, product_name):
    """Generate a placeholder JPEG image with valid format"""
    # Create a simple image with gradient background
    img = Image.new('RGB', (width, height), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    # Add some text/content
    try:
        # Try to use a default font, fallback to basic if not available
        font = ImageFont.load_default()
    except:
        font = None
    
    text = f"Ad {ad_index}\n{product_name[:30]}"
    bbox = draw.textbbox((0, 0), text, font=font) if font else (0, 0, 100, 50)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    position = ((width - text_width) // 2, (height - text_height) // 2)
    draw.text(position, text, fill=(50, 50, 50), font=font)
    
    # Convert to JPEG bytes
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='JPEG', quality=85)
    img_bytes = img_buffer.getvalue()
    img_buffer.close()
    
    return img_bytes


def validate_jpeg_bytes(image_bytes):
    """Validate that bytes represent a valid JPEG image"""
    if not image_bytes or len(image_bytes) < 100:
        return False, "Image data too small (less than 100 bytes)"
    
    # Check JPEG magic bytes
    if not (image_bytes[0] == 0xFF and image_bytes[1] == 0xD8):
        return False, "Invalid JPEG magic bytes (not starting with FF D8)"
    
    # Check JPEG end marker
    if not (image_bytes[-2] == 0xFF and image_bytes[-1] == 0xD9):
        return False, "Invalid JPEG end marker (not ending with FF D9)"
    
    # Try to open with PIL to verify it's a valid image
    try:
        img_buffer = io.BytesIO(image_bytes)
        img = Image.open(img_buffer)
        img.verify()
        img_buffer.close()
        return True, None
    except Exception as e:
        return False, f"PIL validation failed: {str(e)}"


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
        
        # Generate 3 ads (mock implementation for Phase 1)
        ads = []
        for ad_index in range(1, 4):
            # Generate a valid JPEG image
            try:
                image_bytes = generate_placeholder_image(width, height, ad_index, product_name)
                
                # Validate the generated image
                is_valid, error_msg = validate_jpeg_bytes(image_bytes)
                if not is_valid:
                    return jsonify({
                        "error": "Image generation failed",
                        "message": error_msg
                    }), 500
                
                # Encode to base64
                image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                image_data_url = f"data:image/jpeg;base64,{image_base64}"
                
                ad = {
                    "ad_index": ad_index,
                    "marketing_text": f"Discover {product_name}: {product_description[:50]}...",
                    "image_jpg": image_data_url
                }
                ads.append(ad)
                
            except Exception as e:
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

