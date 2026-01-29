from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import uuid
import logging
import io
import zipfile
from engine.ace_engine import generate_ad

app = Flask(__name__)

# Configure CORS for specific origins only
CORS(app, origins=[
    "https://ace-advertising.agency",
    "http://localhost:5173"
])

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Allowed image sizes
ALLOWED_SIZES = ["1024x1024", "1536x1024", "1024x1536"]


@app.route('/api/generate', methods=['POST'])
def generate():
    """
    Generate a single ad and return as ZIP file.
    
    Request JSON:
    {
        "productName": string,
        "productDescription": string,
        "imageSize": "1024x1024" | "1536x1024" | "1024x1536"
    }
    
    Success: Returns ZIP file (application/zip)
    Error: Returns JSON with error message (HTTP 400 or 500)
    """
    request_id = str(uuid.uuid4())
    
    try:
        # Get and validate request data
        data = request.get_json()
        if not data:
            logger.warning(f"Request {request_id}: Missing JSON body")
            return jsonify({
                'status': 'error',
                'message': 'Invalid request: JSON body required'
            }), 400
        
        # Extract and validate fields
        product_name = data.get('productName', '').strip()
        product_description = data.get('productDescription', '').strip()
        image_size = data.get('imageSize', '').strip()
        
        # Log request (input lengths only, no content)
        logger.info(f"Request {request_id}: productName length={len(product_name)}, productDescription length={len(product_description)}, imageSize={image_size}")
        
        # Validation: productName
        if not product_name:
            logger.warning(f"Request {request_id}: Validation failed - productName is empty")
            return jsonify({
                'status': 'error',
                'message': 'productName is required and must be non-empty'
            }), 400
        
        # Validation: productDescription
        if not product_description:
            logger.warning(f"Request {request_id}: Validation failed - productDescription is empty")
            return jsonify({
                'status': 'error',
                'message': 'productDescription is required and must be non-empty'
            }), 400
        
        # Validation: imageSize
        if not image_size:
            logger.warning(f"Request {request_id}: Validation failed - imageSize is empty")
            return jsonify({
                'status': 'error',
                'message': 'imageSize is required'
            }), 400
        
        if image_size not in ALLOWED_SIZES:
            logger.warning(f"Request {request_id}: Validation failed - imageSize={image_size} not in allowed sizes")
            return jsonify({
                'status': 'error',
                'message': f'imageSize must be one of: {", ".join(ALLOWED_SIZES)}'
            }), 400
        
        # Call engine (isolated from route logic)
        try:
            engine_result = generate_ad(product_name, product_description, image_size)
        except NotImplementedError:
            logger.error(f"Request {request_id}: Engine not yet implemented")
            return jsonify({
                'status': 'error',
                'message': 'Engine implementation pending'
            }), 500
        except ValueError as e:
            logger.error(f"Request {request_id}: Engine validation error: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
        except Exception as e:
            logger.error(f"Request {request_id}: Engine error: {str(e)}", exc_info=True)
            return jsonify({
                'status': 'error',
                'message': 'Ad generation failed'
            }), 500
        
        # Validate engine result structure
        if not isinstance(engine_result, dict):
            logger.error(f"Request {request_id}: Engine returned invalid result type")
            return jsonify({
                'status': 'error',
                'message': 'Engine returned invalid result'
            }), 500
        
        image_bytes_jpg = engine_result.get('image_bytes_jpg')
        marketing_text = engine_result.get('marketing_text')
        
        if not image_bytes_jpg or not isinstance(image_bytes_jpg, bytes):
            logger.error(f"Request {request_id}: Engine result missing or invalid image_bytes_jpg")
            return jsonify({
                'status': 'error',
                'message': 'Engine returned invalid image data'
            }), 500
        
        if not marketing_text or not isinstance(marketing_text, str):
            logger.error(f"Request {request_id}: Engine result missing or invalid marketing_text")
            return jsonify({
                'status': 'error',
                'message': 'Engine returned invalid marketing text'
            }), 500
        
        # Create ZIP in memory
        zip_buffer = io.BytesIO()
        try:
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Add image.jpg
                zip_file.writestr('image.jpg', image_bytes_jpg)
                
                # Add text.txt
                zip_file.writestr('text.txt', marketing_text.encode('utf-8'))
            
            zip_buffer.seek(0)
            
            logger.info(f"Request {request_id}: Successfully generated ZIP (image size: {len(image_bytes_jpg)} bytes, text length: {len(marketing_text)} chars)")
            
            # Return ZIP file directly (not JSON)
            return send_file(
                zip_buffer,
                mimetype='application/zip',
                as_attachment=True,
                download_name='ad.zip'
            )
            
        except Exception as e:
            logger.error(f"Request {request_id}: Error creating ZIP: {str(e)}", exc_info=True)
            return jsonify({
                'status': 'error',
                'message': 'Failed to create ZIP file'
            }), 500
        
    except Exception as e:
        logger.error(f"Request {request_id}: Unexpected error: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': 'Internal server error'
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok'}), 200


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

