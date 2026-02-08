from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
import uuid
import logging
import io
import zipfile
import json
import base64
import os
from datetime import datetime, timezone
from typing import Dict, Optional
# Lazy import: engine only loaded when needed (not for /health endpoint)
# from engine.ace_engine import generate_ad, quota_state_from_dict, quota_state_to_dict

app = Flask(__name__)

# Configure CORS for specific origins only, expose X-ACE-Batch-State header
CORS(app, origins=[
    "https://ace-advertising.agency",
    "http://localhost:5173"
], expose_headers=["X-ACE-Batch-State"])

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Lazy import function for engine (avoids heavy initialization on startup)
def get_engine_functions():
    """Lazy import of engine functions - only called when needed"""
    from engine.ace_engine import generate_ad, quota_state_from_dict, quota_state_to_dict
    return generate_ad, quota_state_from_dict, quota_state_to_dict

# Allowed image sizes
ALLOWED_SIZES = ["1024x1024", "1536x1024", "1024x1536"]

# ============================================================================
# Payment Entitlement System (Future Foundation)
# ============================================================================

# In-memory store for entitlements (isolated, organized)
_entitlements_store: Dict[str, Dict] = {}

def create_entitlement() -> Dict:
    """
    Create a new entitlement with default values.
    
    Returns:
        Dict with sid, status, credits_remaining, created_at, updated_at
    """
    sid = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    
    entitlement = {
        'sid': sid,
        'status': 'pending',
        'credits_remaining': 0,
        'created_at': now,
        'updated_at': now
    }
    
    _entitlements_store[sid] = entitlement
    logger.info(f"ENTITLEMENT_CREATED sid={sid}")
    
    return entitlement

def get_entitlement(sid: str) -> Optional[Dict]:
    """
    Get entitlement by sid.
    
    Args:
        sid: Entitlement ID
        
    Returns:
        Entitlement dict if found, None otherwise
    """
    return _entitlements_store.get(sid)

def find_pending_entitlement(preferred_sid: Optional[str] = None) -> Optional[Dict]:
    """
    Find a pending entitlement.
    
    Priority:
    1. If preferred_sid is provided and exists with status='pending', return it
    2. Otherwise, return the most recently created pending entitlement
    
    Args:
        preferred_sid: Optional preferred sid to look for first
        
    Returns:
        Entitlement dict if found, None otherwise
    """
    # Try preferred sid first if provided
    if preferred_sid:
        entitlement = _entitlements_store.get(preferred_sid)
        if entitlement and entitlement.get('status') == 'pending':
            return entitlement
    
    # Find most recently created pending entitlement
    pending_entitlements = [
        ent for ent in _entitlements_store.values()
        if ent.get('status') == 'pending'
    ]
    
    if not pending_entitlements:
        return None
    
    # Sort by created_at (most recent first)
    pending_entitlements.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    return pending_entitlements[0]

def activate_entitlement(sid: str) -> bool:
    """
    Activate an entitlement (set to paid with 3 credits).
    
    Args:
        sid: Entitlement ID
        
    Returns:
        True if activated, False if not found
    """
    entitlement = _entitlements_store.get(sid)
    if not entitlement:
        return False
    
    entitlement['status'] = 'paid'
    entitlement['credits_remaining'] = 3
    entitlement['updated_at'] = datetime.now(timezone.utc).isoformat()
    
    return True

def consume_credit(sid: str) -> Optional[int]:
    """
    Consume one credit from an entitlement.
    
    Args:
        sid: Entitlement ID
        
    Returns:
        Remaining credits after consumption, or None if not found
    """
    entitlement = _entitlements_store.get(sid)
    if not entitlement:
        return None
    
    current_credits = entitlement.get('credits_remaining', 0)
    if current_credits <= 0:
        return current_credits
    
    # Decrease credits
    new_credits = current_credits - 1
    entitlement['credits_remaining'] = new_credits
    entitlement['updated_at'] = datetime.now(timezone.utc).isoformat()
    
    # If credits reach 0, mark as consumed
    if new_credits == 0:
        entitlement['status'] = 'consumed'
    
    return new_credits


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
        }
    }
    
    Success: Returns ZIP file (application/zip) with X-ACE-Batch-State header
    Error: Returns JSON with error message and batchState (HTTP 400 or 500)
    """
    request_id = str(uuid.uuid4())
    current_batch_state = None  # Track batch state for error responses
    
    # ============================================================================
    # Entitlement Check (Enforcement)
    # ============================================================================
    # This check enforces entitlement requirements before generation.
    
    # Try to read sid from request (query / header / body)
    sid = None
    entitlement_check = "missing_sid"
    
    # Try query parameter first
    sid = request.args.get('sid')
    
    # Try header if not in query
    if not sid:
        sid = request.headers.get('X-Entitlement-Sid')
    
    # Try body if not in query or header
    if not sid:
        data_preview = request.get_json(silent=True)
        if data_preview and isinstance(data_preview, dict):
            sid = data_preview.get('sid')
    
    # Enforce entitlement check
    if not sid:
        entitlement_check = "missing_sid"
        logger.info(f"GENERATE_BLOCKED sid=none reason={entitlement_check}")
        return jsonify({
            'error': 'Payment required'
        }), 403
    
    # Try to get entitlement
    entitlement = get_entitlement(sid)
    
    if not entitlement:
        entitlement_check = "not_found"
        logger.info(f"GENERATE_BLOCKED sid={sid} reason={entitlement_check}")
        return jsonify({
            'error': 'Payment required'
        }), 403
    
    # Check status
    if entitlement.get('status') != 'paid':
        entitlement_check = "not_paid"
        logger.info(f"GENERATE_BLOCKED sid={sid} reason={entitlement_check}")
        return jsonify({
            'error': 'Payment required'
        }), 403
    
    # Check credits
    if entitlement.get('credits_remaining', 0) <= 0:
        entitlement_check = "no_credits"
        logger.info(f"GENERATE_BLOCKED sid={sid} reason={entitlement_check}")
        return jsonify({
            'error': 'Session consumed'
        }), 403
    
    # All checks passed
    entitlement_check = "allowed"
    
    # ============================================================================
    # End of Entitlement Check (generation continues only if allowed)
    # ============================================================================
    
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
            
            # Consume credit only after successful generation
            if sid and entitlement_check == "allowed":
                remaining = consume_credit(sid)
                if remaining is not None:
                    logger.info(f"GENERATE_CREDIT_CONSUMED sid={sid} remaining={remaining}")
            
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


@app.route('/api/entitlement/create', methods=['POST'])
def create_entitlement_endpoint():
    """
    Create a new entitlement.
    
    Returns:
        JSON: {
            "sid": string (UUID),
            "status": "pending",
            "credits_remaining": 0
        }
    """
    entitlement = create_entitlement()
    
    return jsonify({
        'sid': entitlement['sid'],
        'status': entitlement['status'],
        'credits_remaining': entitlement['credits_remaining']
    }), 200


@app.route('/api/entitlement/<sid>', methods=['GET'])
def get_entitlement_endpoint(sid):
    """
    Get entitlement by sid.
    
    Args:
        sid: Entitlement ID
        
    Returns:
        JSON: {
            "sid": string,
            "status": "pending" | "paid" | "consumed",
            "credits_remaining": number,
            "created_at": string (ISO timestamp),
            "updated_at": string (ISO timestamp)
        }
        Error: 404 if entitlement not found
    """
    entitlement = get_entitlement(sid)
    
    if not entitlement:
        return jsonify({
            'status': 'error',
            'message': 'Entitlement not found'
        }), 404
    
    return jsonify(entitlement), 200


# ============================================================================
# IPN Endpoint (Payment Notification from iCount)
# ============================================================================

@app.route('/api/ipn/<token>', methods=['POST'])
def ipn_handler(token):
    """
    IPN endpoint for iCount payment notifications.
    
    This endpoint receives payment notifications and activates entitlements.
    The token in the URL must match ICOUNT_IPN_TOKEN from environment.
    
    Args:
        token: IPN token from URL path
        
    Behavior:
    - Validates token matches ENV variable
    - Finds pending entitlement (by sid in request or latest)
    - Activates entitlement (status=paid, credits_remaining=3)
    
    Returns:
        "OK", 200
    """
    # Validate token from ENV (secret, not logged)
    expected_token = os.environ.get('ICOUNT_IPN_TOKEN')
    if not expected_token:
        logger.warning("IPN endpoint called but ICOUNT_IPN_TOKEN not configured")
        return "OK", 200  # Don't reveal token mismatch to caller
    
    if token != expected_token:
        logger.warning("IPN endpoint called with invalid token")
        return "OK", 200  # Don't reveal token mismatch to caller
    
    # Try to get sid from request (query params, form data, or JSON body)
    sid_from_request = None
    
    # Try query parameter
    sid_from_request = request.args.get('sid')
    
    # Try form data
    if not sid_from_request:
        sid_from_request = request.form.get('sid')
    
    # Try JSON body (if present)
    if not sid_from_request:
        try:
            json_data = request.get_json(silent=True)
            if json_data and isinstance(json_data, dict):
                sid_from_request = json_data.get('sid')
        except Exception:
            pass
    
    # Find pending entitlement (prefer sid from request, otherwise latest)
    entitlement = find_pending_entitlement(sid_from_request)
    
    if not entitlement:
        logger.info("IPN_NO_PENDING_ENTITLEMENT")
        return "OK", 200
    
    # Activate entitlement
    sid = entitlement['sid']
    if activate_entitlement(sid):
        logger.info(f"IPN_ACCEPTED sid={sid}")
    else:
        logger.warning(f"IPN_ACCEPTED failed to activate sid={sid}")
    
    return "OK", 200


@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint - minimal, no heavy imports, returns plain text.
    This endpoint must NOT trigger any ACE/OpenAI initialization.
    """
    return 'ok', 200




if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"HEALTH_READY /health returns 200 immediately")
    app.run(host='0.0.0.0', port=port, debug=False)

