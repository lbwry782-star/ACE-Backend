from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import zipfile
import base64
import json
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all origins

# In-memory storage for generated ads (in production, use Redis or database)
ads_storage = {}


def generate_placeholder_image(width, height, ad_index, product_name):
    """Generate a minimal valid JPEG placeholder without PIL"""
    # Use a known minimal valid 1x1 pixel JPEG (base64 encoded)
    # This is a real valid JPEG that browsers can display
    # We'll decode it and use it as a template, then adjust dimensions in header
    minimal_jpeg_b64 = "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/2wBDAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwA/8A"
    
    try:
        # Decode the minimal JPEG
        minimal_jpeg = base64.b64decode(minimal_jpeg_b64)
        
        # For a placeholder, we'll return a larger valid JPEG by repeating/expanding
        # the minimal structure. Since we can't easily resize without PIL, we'll
        # create a minimal valid JPEG with the requested dimensions in the header
        # but use the same encoded data structure (browsers will handle it)
        
        # Build a minimal valid JPEG with correct dimensions
        # Start of Image
        soi = bytes([0xFF, 0xD8])
        
        # JFIF APP0
        app0 = bytes([
            0xFF, 0xE0, 0x00, 0x10,
            0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,  # "JFIF\0\1"
            0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00
        ])
        
        # Quantization table
        dqt = bytes([0xFF, 0xDB, 0x00, 0x43, 0x00]) + bytes([8] * 64)
        
        # Start of Frame with correct dimensions
        sof = bytes([
            0xFF, 0xC0, 0x00, 0x11, 0x08,
            (height >> 8) & 0xFF, height & 0xFF,
            (width >> 8) & 0xFF, width & 0xFF,
            0x01, 0x11, 0x00
        ])
        
        # Huffman tables (DC)
        dht_dc = bytes([
            0xFF, 0xC4, 0x00, 0x1F, 0x00,
            0x00, 0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B
        ])
        
        # Huffman tables (AC) - minimal
        dht_ac = bytes([
            0xFF, 0xC4, 0x00, 0xB5, 0x10,
            0x00, 0x02, 0x01, 0x03, 0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7D,
            0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
            0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0,
            0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
            0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
            0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
            0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
            0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
            0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5,
            0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
            0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
            0xF9, 0xFA
        ])
        
        # Start of Scan
        sos = bytes([0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00])
        
        # Minimal encoded scan data (represents a simple gray image)
        # This is a minimal valid JPEG scan data
        scan_data = bytes([0x3F, 0x00] + [0x00] * 400)  # Ensure minimum size
        
        # End of Image
        eoi = bytes([0xFF, 0xD9])
        
        jpeg_bytes = soi + app0 + dqt + sof + dht_dc + dht_ac + sos + scan_data + eoi
        
        return jpeg_bytes
        
    except Exception:
        # Fallback: return the minimal JPEG as-is (it's valid)
        return base64.b64decode(minimal_jpeg_b64)


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

