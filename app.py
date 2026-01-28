from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid
import time
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS for localhost Vite

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/api/generate', methods=['POST'])
def generate():
    try:
        # Get request data
        data = request.get_json()
        
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        # Log request
        logger.info(f"Request ID: {request_id}")
        logger.info(f"Received data - productName: {data.get('productName', 'N/A')}, productDescription length: {len(data.get('productDescription', ''))}")
        
        # Validate required fields
        product_name = data.get('productName', '').strip()
        product_description = data.get('productDescription', '').strip()
        
        if not product_name:
            return jsonify({
                'status': 'error',
                'message': 'שם המוצר הוא שדה חובה'
            }), 400
        
        if not product_description:
            return jsonify({
                'status': 'error',
                'message': 'תיאור המוצר הוא שדה חובה'
            }), 400
        
        # Simulate work (1.5 seconds)
        time.sleep(1.5)
        
        # Return mock result
        result = {
            'requestId': request_id,
            'status': 'success',
            'result': {
                'title': f'פרסומת עבור {product_name}',
                'summary': f'נוצרה פרסומת מותאמת אישית עבור המוצר {product_name}. הפרסומת מבוססת על התיאור: {product_description[:50]}...',
                'files': []
            }
        }
        
        logger.info(f"Request {request_id} completed successfully")
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'שגיאה בעיבוד הבקשה: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

