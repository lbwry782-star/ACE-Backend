from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
import uuid
import logging
import io
import zipfile
import json
import base64
from datetime import datetime, timezone
# Lazy import: engine only loaded when needed (not for /health endpoint)
# from engine.ace_engine import generate_ad, quota_state_from_dict, quota_state_to_dict

app = Flask(__name__)

# Configure CORS for specific origins only, expose X-ACE-Batch-State header
# supports_credentials=True allows cookies to be sent with requests
CORS(app, origins=[
    "https://ace-advertising.agency",
    "http://localhost:5173"
], expose_headers=["X-ACE-Batch-State"], supports_credentials=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Payment status tracking (in-memory storage)
# Maps payment_session -> {"paid": bool, "docnum": str, "doctype": str, "at": datetime}
paid_sessions = {}

# Quota tracking (in-memory storage)
# Maps payment_session -> {"max": 3, "consumed": int}
quota_by_session = {}

# Cookie binding tracking (in-memory storage)
# Maps cookie_id -> payment_session (for cookie-based authorization)
bound_session_by_cookie = {}
# Maps payment_session -> cookie_id (reverse lookup)
cookie_by_payment_session = {}

# Lazy import function for engine (avoids heavy initialization on startup)
def get_engine_functions():
    """Lazy import of engine functions - only called when needed"""
    from engine.ace_engine import generate_ad, quota_state_from_dict, quota_state_to_dict
    return generate_ad, quota_state_from_dict, quota_state_to_dict

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
        "imageSize": "1024x1024" | "1536x1024" | "1024x1536",
        "adIndex": int (optional, default 0),
        "sessionSeed": string (optional, for deterministic window selection),
        "batchState": {  // optional
            "material_analogy_used": boolean,
            "structural_morphology_used": boolean,
            "structural_exception_used": boolean
        },
        "payment_session": string (required)
    }
    
    Success: Returns ZIP file (application/zip) with X-ACE-Batch-State header
    Error: Returns JSON with error message and batchState (HTTP 400 or 500)
    """
    request_id = str(uuid.uuid4())
    current_batch_state = None  # Track batch state for error responses
    
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
        ad_index = data.get('adIndex', 0)  # Get ad_index from request, default 0
        session_seed = data.get('sessionSeed')  # Optional session seed for deterministic window selection
        batch_state_dict = data.get('batchState')  # Optional
        payment_session = data.get('payment_session')  # Required for quota check
        
        # Payment and quota validation (before engine import)
        if not payment_session:
            logger.warning(f"Request {request_id}: Missing payment_session")
            return jsonify({
                'error': 'payment_required'
            }), 403
        
        # Check if payment_session is marked as paid
        payment_info = paid_sessions.get(payment_session)
        if not payment_info or not payment_info.get('paid', False):
            logger.warning(f"Request {request_id}: Payment session not paid: {payment_session}")
            return jsonify({
                'error': 'payment_required'
            }), 403
        
        # Check quota
        quota_info = quota_by_session.get(payment_session)
        if not quota_info:
            logger.warning(f"Request {request_id}: Quota not found for payment_session: {payment_session}")
            return jsonify({
                'error': 'payment_required'
            }), 403
        
        consumed = quota_info.get('consumed', 0)
        max_quota = quota_info.get('max', 3)
        
        if consumed >= max_quota:
            logger.warning(f"Request {request_id}: Quota exceeded for payment_session: {payment_session} (consumed={consumed}/{max_quota})")
            return jsonify({
                'error': 'quota_exceeded'
            }), 403
        
        # Cookie-based authorization check
        cookie_id = request.cookies.get('ace_pay_claim')
        if not cookie_id:
            logger.warning(f"Request {request_id}: Missing cookie ace_pay_claim")
            return jsonify({
                'error': 'not_authorized'
            }), 403
        
        # Check if cookie_id is bound to a payment_session
        bound_payment_session = bound_session_by_cookie.get(cookie_id)
        if not bound_payment_session:
            logger.warning(f"Request {request_id}: Cookie {cookie_id} not found in bound_session_by_cookie")
            return jsonify({
                'error': 'not_authorized'
            }), 403
        
        # Check if cookie is bound to the correct payment_session
        if bound_payment_session != payment_session:
            logger.warning(f"Request {request_id}: Cookie {cookie_id} bound to {bound_payment_session}, but request has {payment_session}")
            return jsonify({
                'error': 'not_authorized'
            }), 403
        
        # Consume one quota
        quota_info['consumed'] = consumed + 1
        logger.info(f"QUOTA_CONSUME payment_session={payment_session} consumed={consumed + 1}/{max_quota}")
        
        # Load quota_state from X-ACE-Batch-State header if present, otherwise from request JSON
        batch_state_header = request.headers.get('X-ACE-Batch-State')
        if batch_state_header:
            try:
                batch_state_dict = json.loads(batch_state_header)
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Request {request_id}: Invalid X-ACE-Batch-State header, using request batchState")
        
        # Lazy import engine functions (only when actually needed, not for /health)
        generate_ad, quota_state_from_dict, quota_state_to_dict = get_engine_functions()
        
        # Convert batchState dict to BatchQuotaState (defaults to all False if missing)
        quota_state = quota_state_from_dict(batch_state_dict)
        current_batch_state = quota_state_to_dict(quota_state)
        
        # Log request (input lengths only, no content)
        logger.info(f"Request {request_id}: productName length={len(product_name)}, productDescription length={len(product_description)}, imageSize={image_size}, adIndex={ad_index}, batchState={current_batch_state}")
        
        # Validation: productName
        if not product_name:
            logger.warning(f"Request {request_id}: Validation failed - productName is empty")
            return jsonify({
                'status': 'error',
                'message': 'productName is required and must be non-empty',
                'batchState': current_batch_state
            }), 400
        
        # Validation: productDescription
        if not product_description:
            logger.warning(f"Request {request_id}: Validation failed - productDescription is empty")
            return jsonify({
                'status': 'error',
                'message': 'productDescription is required and must be non-empty',
                'batchState': current_batch_state
            }), 400
        
        # Validation: imageSize
        if not image_size:
            logger.warning(f"Request {request_id}: Validation failed - imageSize is empty")
            return jsonify({
                'status': 'error',
                'message': 'imageSize is required',
                'batchState': current_batch_state
            }), 400
        
        if image_size not in ALLOWED_SIZES:
            logger.warning(f"Request {request_id}: Validation failed - imageSize={image_size} not in allowed sizes")
            return jsonify({
                'status': 'error',
                'message': f'imageSize must be one of: {", ".join(ALLOWED_SIZES)}',
                'batchState': current_batch_state
            }), 400
        
        # Call engine (isolated from route logic) with quota state, ad_index, and session_seed
        try:
            engine_result = generate_ad(
                product_name,
                product_description,
                image_size,
                quota_state=quota_state,
                ad_index=ad_index,
                session_seed=session_seed
            )
        except NotImplementedError:
            logger.error(f"Request {request_id}: Engine not yet implemented")
            return jsonify({
                'status': 'error',
                'message': 'Engine implementation pending',
                'batchState': current_batch_state
            }), 500
        except ValueError as e:
            logger.error(f"Request {request_id}: Engine validation error: {str(e)}")
            # Get updated batch state from quota_state (may have been modified)
            updated_batch_state = quota_state_to_dict(quota_state)
            return jsonify({
                'status': 'error',
                'message': str(e),
                'batchState': updated_batch_state
            }), 500
        except Exception as e:
            logger.error(f"Request {request_id}: Engine error: {str(e)}", exc_info=True)
            # Get updated batch state from quota_state (may have been modified)
            updated_batch_state = quota_state_to_dict(quota_state)
            return jsonify({
                'status': 'error',
                'message': 'Ad generation failed',
                'batchState': updated_batch_state
            }), 500
        
        # Validate engine result structure
        if not isinstance(engine_result, dict):
            logger.error(f"Request {request_id}: Engine returned invalid result type")
            return jsonify({
                'status': 'error',
                'message': 'Engine returned invalid result',
                'batchState': current_batch_state
            }), 500
        
        image_bytes_jpg = engine_result.get('image_bytes_jpg')
        marketing_text = engine_result.get('marketing_text')
        batch_state = engine_result.get('batch_state')
        
        if not image_bytes_jpg or not isinstance(image_bytes_jpg, bytes):
            logger.error(f"Request {request_id}: Engine result missing or invalid image_bytes_jpg")
            # Get updated batch state from quota_state (may have been modified)
            updated_batch_state = quota_state_to_dict(quota_state)
            return jsonify({
                'status': 'error',
                'message': 'Engine returned invalid image data',
                'batchState': updated_batch_state
            }), 500
        
        if not marketing_text or not isinstance(marketing_text, str):
            logger.error(f"Request {request_id}: Engine result missing or invalid marketing_text")
            # Get updated batch state from quota_state (may have been modified)
            updated_batch_state = quota_state_to_dict(quota_state)
            return jsonify({
                'status': 'error',
                'message': 'Engine returned invalid marketing text',
                'batchState': updated_batch_state
            }), 500
        
        # Get final batch state from quota_state (updated in-place during generation)
        final_batch_state = quota_state_to_dict(quota_state)
        
        # Create ZIP in memory with single ad
        zip_buffer = io.BytesIO()
        try:
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Add image.jpg
                zip_file.writestr('image.jpg', image_bytes_jpg)
                
                # Add text.txt
                zip_file.writestr('text.txt', marketing_text.encode('utf-8'))
            
            zip_buffer.seek(0)
            
            logger.info(f"Request {request_id}: Successfully generated ZIP (image size: {len(image_bytes_jpg)} bytes, text length: {len(marketing_text)} chars)")
            
            # Return ZIP file with X-ACE-Batch-State header
            response = send_file(
                zip_buffer,
                mimetype='application/zip',
                as_attachment=True,
                download_name='ad.zip'
            )
            
            # Add batch state to response header (updated after generation)
            response.headers['X-ACE-Batch-State'] = json.dumps(final_batch_state)
            
            return response
            
        except Exception as e:
            logger.error(f"Request {request_id}: Error creating ZIP: {str(e)}", exc_info=True)
            # Get final batch state (quota_state was updated in-place during generation)
            final_batch_state = quota_state_to_dict(quota_state)
            return jsonify({
                'status': 'error',
                'message': 'Failed to create ZIP file',
                'batchState': final_batch_state
            }), 500
        
    except Exception as e:
        logger.error(f"Request {request_id}: Unexpected error: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': 'Internal server error',
            'batchState': current_batch_state if current_batch_state else {
                'material_analogy_used': False,
                'structural_morphology_used': False,
                'structural_exception_used': False
            }
        }), 500


@app.route('/api/preview', methods=['POST'])
def preview():
    """
    Generate a single ad and return as JSON for preview.
    
    Request JSON:
    {
        "productName": string,
        "productDescription": string,
        "imageSize": "1024x1024" | "1536x1024" | "1024x1536",
        "adIndex": int (optional, default 0),
        "sessionSeed": string (optional, for deterministic window selection),
        "batchState": {  // optional
            "material_analogy_used": boolean,
            "structural_morphology_used": boolean,
            "structural_exception_used": boolean
        }
    }
    
    Success: Returns JSON with imageBase64, marketingText, batchState and X-ACE-Batch-State header
    Error: Returns JSON with error message and batchState (HTTP 400 or 500)
    """
    request_id = str(uuid.uuid4())
    current_batch_state = None  # Track batch state for error responses
    
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
        ad_index = data.get('adIndex', 0)  # Get ad_index from request, default 0
        session_seed = data.get('sessionSeed')  # Optional session seed for deterministic window selection
        batch_state_dict = data.get('batchState')  # Optional
        
        # Load quota_state from X-ACE-Batch-State header if present, otherwise from request JSON
        batch_state_header = request.headers.get('X-ACE-Batch-State')
        if batch_state_header:
            try:
                batch_state_dict = json.loads(batch_state_header)
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Request {request_id}: Invalid X-ACE-Batch-State header, using request batchState")
        
        # Lazy import engine functions (only when actually needed, not for /health)
        generate_ad, quota_state_from_dict, quota_state_to_dict = get_engine_functions()
        
        # Convert batchState dict to BatchQuotaState (defaults to all False if missing)
        quota_state = quota_state_from_dict(batch_state_dict)
        current_batch_state = quota_state_to_dict(quota_state)
        
        # Log request (input lengths only, no content)
        logger.info(f"Request {request_id}: productName length={len(product_name)}, productDescription length={len(product_description)}, imageSize={image_size}, adIndex={ad_index}, batchState={current_batch_state}")
        
        # Validation: productName
        if not product_name:
            logger.warning(f"Request {request_id}: Validation failed - productName is empty")
            return jsonify({
                'status': 'error',
                'message': 'productName is required and must be non-empty',
                'batchState': current_batch_state
            }), 400
        
        # Validation: productDescription
        if not product_description:
            logger.warning(f"Request {request_id}: Validation failed - productDescription is empty")
            return jsonify({
                'status': 'error',
                'message': 'productDescription is required and must be non-empty',
                'batchState': current_batch_state
            }), 400
        
        # Validation: imageSize
        if not image_size:
            logger.warning(f"Request {request_id}: Validation failed - imageSize is empty")
            return jsonify({
                'status': 'error',
                'message': 'imageSize is required',
                'batchState': current_batch_state
            }), 400
        
        if image_size not in ALLOWED_SIZES:
            logger.warning(f"Request {request_id}: Validation failed - imageSize={image_size} not in allowed sizes")
            return jsonify({
                'status': 'error',
                'message': f'imageSize must be one of: {", ".join(ALLOWED_SIZES)}',
                'batchState': current_batch_state
            }), 400
        
        # Call engine (isolated from route logic) with quota state, ad_index, and session_seed
        try:
            engine_result = generate_ad(
                product_name,
                product_description,
                image_size,
                quota_state=quota_state,
                ad_index=ad_index,
                session_seed=session_seed
            )
        except NotImplementedError:
            logger.error(f"Request {request_id}: Engine not yet implemented")
            return jsonify({
                'status': 'error',
                'message': 'Engine implementation pending',
                'batchState': current_batch_state
            }), 500
        except ValueError as e:
            logger.error(f"Request {request_id}: Engine validation error: {str(e)}")
            # Get updated batch state from quota_state (may have been modified)
            updated_batch_state = quota_state_to_dict(quota_state)
            return jsonify({
                'status': 'error',
                'message': str(e),
                'batchState': updated_batch_state
            }), 500
        except Exception as e:
            logger.error(f"Request {request_id}: Engine error: {str(e)}", exc_info=True)
            # Get updated batch state from quota_state (may have been modified)
            updated_batch_state = quota_state_to_dict(quota_state)
            return jsonify({
                'status': 'error',
                'message': 'Ad generation failed',
                'batchState': updated_batch_state
            }), 500
        
        # Validate engine result structure
        if not isinstance(engine_result, dict):
            logger.error(f"Request {request_id}: Engine returned invalid result type")
            return jsonify({
                'status': 'error',
                'message': 'Engine returned invalid result',
                'batchState': current_batch_state
            }), 500
        
        image_bytes_jpg = engine_result.get('image_bytes_jpg')
        marketing_text = engine_result.get('marketing_text')
        batch_state = engine_result.get('batch_state')
        
        if not image_bytes_jpg or not isinstance(image_bytes_jpg, bytes):
            logger.error(f"Request {request_id}: Engine result missing or invalid image_bytes_jpg")
            # Get updated batch state from quota_state (may have been modified)
            updated_batch_state = quota_state_to_dict(quota_state)
            return jsonify({
                'status': 'error',
                'message': 'Engine returned invalid image data',
                'batchState': updated_batch_state
            }), 500
        
        if not marketing_text or not isinstance(marketing_text, str):
            logger.error(f"Request {request_id}: Engine result missing or invalid marketing_text")
            # Get updated batch state from quota_state (may have been modified)
            updated_batch_state = quota_state_to_dict(quota_state)
            return jsonify({
                'status': 'error',
                'message': 'Engine returned invalid marketing text',
                'batchState': updated_batch_state
            }), 500
        
        # Get final batch state from quota_state (updated in-place during generation)
        final_batch_state = quota_state_to_dict(quota_state)
        
        # Convert image bytes to base64 (without prefix)
        image_base64 = base64.b64encode(image_bytes_jpg).decode('utf-8')
        
        logger.info(f"Request {request_id}: Successfully generated preview (image size: {len(image_bytes_jpg)} bytes, text length: {len(marketing_text)} chars)")
        
        # Return JSON response with X-ACE-Batch-State header
        response = jsonify({
            'imageBase64': image_base64,
            'marketingText': marketing_text,
            'batchState': final_batch_state
        })
        
        # Add batch state to response header (updated after generation)
        response.headers['X-ACE-Batch-State'] = json.dumps(final_batch_state)
        
        return response
        
    except Exception as e:
        logger.error(f"Request {request_id}: Unexpected error: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': 'Internal server error',
            'batchState': current_batch_state if current_batch_state else {
                'material_analogy_used': False,
                'structural_morphology_used': False,
                'structural_exception_used': False
            }
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint - minimal, no heavy imports, returns plain text.
    This endpoint must NOT trigger any ACE/OpenAI initialization.
    """
    return 'ok', 200


@app.route('/api/claim-payment', methods=['POST'])
def claim_payment():
    """
    Claim a payment_session by binding it to a browser cookie.
    
    Request JSON:
    {
        "payment_session": string (required)
    }
    
    Returns:
        Success: 200 {"ok": true} with Set-Cookie header
        Error: 400 if payment_session missing, 403 if not paid or already claimed
    """
    data = request.get_json()
    if not data:
        return jsonify({
            'status': 'error',
            'message': 'JSON body required'
        }), 400
    
    payment_session = data.get('payment_session')
    if not payment_session:
        return jsonify({
            'status': 'error',
            'message': 'payment_session is required'
        }), 400
    
    # Check if payment_session is paid
    payment_info = paid_sessions.get(payment_session)
    if not payment_info or not payment_info.get('paid', False):
        return jsonify({
            'error': 'payment_required'
        }), 403
    
    # Check if payment_session is already claimed by another cookie
    existing_cookie_id = cookie_by_payment_session.get(payment_session)
    if existing_cookie_id:
        return jsonify({
            'error': 'already_claimed'
        }), 403
    
    # Generate new cookie_id
    cookie_id = str(uuid.uuid4())
    
    # Store mapping in both directions
    bound_session_by_cookie[cookie_id] = payment_session
    cookie_by_payment_session[payment_session] = cookie_id
    
    logger.info(f"PAYMENT_CLAIMED payment_session={payment_session} cookie_id={cookie_id}")
    
    # Create response with cookie
    response = jsonify({"ok": True})
    response.set_cookie(
        'ace_pay_claim',
        cookie_id,
        httponly=True,
        secure=True,
        samesite='Lax',
        path='/'
    )
    
    return response, 200


@app.route('/api/ipn/<token>', methods=['GET', 'POST'])
def icount_ipn(token):
    """
    iCount IPN endpoint - receives webhook notifications from iCount payment system.
    
    Args:
        token: The IPN token identifier
    
    Returns:
        "OK", 200
    """
    logger.info(f"ICOUNT_IPN_RECEIVED token={token} method={request.method} args={request.args.to_dict()} form={request.form.to_dict()}")
    
    # Extract payment_session from form data
    form_data = request.form.to_dict()
    payment_session = form_data.get('payment_session')
    
    if payment_session:
        # Extract docnum and doctype from form
        docnum = form_data.get('docnum', '')
        doctype = form_data.get('doctype', '')
        timestamp = datetime.now(timezone.utc)
        
        # Mark payment as paid
        paid_sessions[payment_session] = {
            "paid": True,
            "docnum": docnum,
            "doctype": doctype,
            "at": timestamp
        }
        logger.info(f"PAYMENT_MARKED_PAID payment_session={payment_session} docnum={docnum} doctype={doctype}")
        
        # Initialize quota for new payment_session
        if payment_session not in quota_by_session:
            quota_by_session[payment_session] = {"max": 3, "consumed": 0}
            logger.info(f"QUOTA_INIT payment_session={payment_session} max=3 consumed=0")
    
    return "OK", 200


@app.route('/api/payment-status', methods=['GET'])
def payment_status():
    """
    Check payment status for a given payment_session.
    
    Query parameters:
        payment_session: The payment session ID to check
    
    Returns:
        JSON: {"payment_session": "<id>", "paid": true/false}
        Error: 400 if payment_session parameter is missing
    """
    payment_session = request.args.get('payment_session')
    
    if not payment_session:
        return jsonify({
            'status': 'error',
            'message': 'payment_session parameter is required'
        }), 400
    
    # Check if payment_session exists in paid_sessions
    payment_info = paid_sessions.get(payment_session)
    paid = payment_info.get('paid', False) if payment_info else False
    
    return jsonify({
        "payment_session": payment_session,
        "paid": paid
    }), 200


@app.route('/api/quota-status', methods=['GET'])
def quota_status():
    """
    Check quota status for a given payment_session.
    
    Query parameters:
        payment_session: The payment session ID to check
    
    Returns:
        JSON: {"payment_session": "...", "paid": true/false, "max": 0 or 3, "consumed": n, "remaining": n}
        Error: 400 if payment_session parameter is missing
    """
    payment_session = request.args.get('payment_session')
    
    if not payment_session:
        return jsonify({
            'status': 'error',
            'message': 'payment_session parameter is required'
        }), 400
    
    # Check payment status (same logic as /api/payment-status)
    payment_info = paid_sessions.get(payment_session)
    paid = payment_info.get('paid', False) if payment_info else False
    
    # If not paid, return zeros
    if not paid:
        return jsonify({
            "payment_session": payment_session,
            "paid": False,
            "max": 0,
            "consumed": 0,
            "remaining": 0
        }), 200
    
    # If paid, check quota (lazy init if needed)
    quota_info = quota_by_session.get(payment_session)
    if not quota_info:
        # Lazy initialization for paid session without quota entry
        quota_by_session[payment_session] = {"max": 3, "consumed": 0}
        logger.info(f"QUOTA_LAZY_INIT payment_session={payment_session} max=3 consumed=0")
        quota_info = quota_by_session[payment_session]
    
    max_quota = quota_info.get('max', 3)
    consumed = quota_info.get('consumed', 0)
    remaining = max_quota - consumed
    
    return jsonify({
        "payment_session": payment_session,
        "paid": True,
        "max": max_quota,
        "consumed": consumed,
        "remaining": remaining
    }), 200


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"HEALTH_READY /health returns 200 immediately")
    app.run(host='0.0.0.0', port=port, debug=False)

