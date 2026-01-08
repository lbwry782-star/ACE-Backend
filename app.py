import os
import json
import base64
import io
import zipfile
import time
import random
import logging
import uuid
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
from openai import OpenAI
from openai import RateLimitError, APIError, APIConnectionError, APITimeoutError

app = Flask(__name__)
CORS(app)  # Enable CORS for all origins (Phase 1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Valid ad sizes
VALID_AD_SIZES = {"1024x1024", "1024x1536", "1536x1024"}

# Engine version
ENGINE_VERSION = "V2"

# Session state storage
# Key: session_id (string) -> value: dict with:
#   - product_name: str (locked on attempt 1)
#   - product_description: str (locked on attempt 1)
#   - selected_size: str (locked on attempt 1)
#   - attempt: int (1, 2, or 3)
#   - used_pairs: list of canonical pair keys (order-insensitive)
#   - created_at: float (timestamp for TTL cleanup)
session_storage = {}

# TTL for abandoned sessions (20 minutes in seconds)
SESSION_TTL_SECONDS = 20 * 60

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TEXT_MODEL = os.getenv("TEXT_MODEL", "gpt-4.1-mini")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gpt-image-1")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

# Initialize OpenAI client
if not OPENAI_API_KEY or not OPENAI_API_KEY.strip():
    client = None
    logger.warning("OPENAI_API_KEY is not set or is empty")
else:
    # Only set base_url if explicitly provided and non-empty
    client_kwargs = {"api_key": OPENAI_API_KEY}
    if OPENAI_BASE_URL and OPENAI_BASE_URL.strip():
        client_kwargs["base_url"] = OPENAI_BASE_URL
    client = OpenAI(**client_kwargs)
    logger.info("OPENAI_API_KEY is present (client initialized)")


def generate_request_id():
    """Generate a short unique request ID."""
    return str(uuid.uuid4())[:8]


def create_pair_key(A, B):
    """Create a canonical pair key (order-insensitive) for tracking used pairs."""
    # Normalize and sort to make order-insensitive
    normalized_A = A.strip().lower()
    normalized_B = B.strip().lower()
    # Sort to ensure same pair regardless of order
    pair = sorted([normalized_A, normalized_B])
    return "|".join(pair)


def cleanup_expired_sessions():
    """Remove sessions that have exceeded TTL."""
    current_time = time.time()
    expired_sessions = []
    for session_id, session_data in session_storage.items():
        if current_time - session_data.get("created_at", 0) > SESSION_TTL_SECONDS:
            expired_sessions.append(session_id)
    for session_id in expired_sessions:
        del session_storage[session_id]
        logger.info(f"SESSION_EXPIRED session_id={session_id} (TTL cleanup)")


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
                logger.info(f"OPENAI_RETRY attempt={attempt + 1} wait={wait_time:.2f}s reason={type(e).__name__} code={error_code} operation={operation_name}")
                time.sleep(wait_time)
            else:
                logger.warning(f"OPENAI_RETRY attempt={attempt + 1} wait={wait_time:.2f}s reason={type(e).__name__} code={error_code} operation={operation_name} - MAX RETRIES REACHED")
        except APIError as e:
            # Check if it's a retryable error: 429 (rate limit) or 5xx server error
            status_code = getattr(e, 'status_code', None)
            if status_code == 429 or (status_code and 500 <= status_code < 600):
                last_exception = e
                base_wait = 2 ** attempt
                jitter = random.uniform(0, 1)
                wait_time = base_wait + jitter
                
                if attempt < max_retries - 1:
                    logger.info(f"OPENAI_RETRY attempt={attempt + 1} wait={wait_time:.2f}s reason=APIError code={status_code} operation={operation_name}")
                    time.sleep(wait_time)
                else:
                    logger.warning(f"OPENAI_RETRY attempt={attempt + 1} wait={wait_time:.2f}s reason=APIError code={status_code} operation={operation_name} - MAX RETRIES REACHED")
            else:
                # Non-retryable API error (4xx except 429)
                raise
        except Exception as e:
            # Non-retryable error, re-raise immediately
            raise
    
    # If we get here, all retries failed
    error_code = getattr(last_exception, 'status_code', None) or getattr(last_exception, 'code', None) or 429
    logger.error(f"OPENAI_RETRY_FAILED returning RETRYABLE_ERROR after {max_retries} retries code={error_code} operation={operation_name}")
    raise RetryableError(f"OpenAI call failed after {max_retries} retries", error_code)


@app.errorhandler(HTTPException)
def handle_http_exception(e):
    """Handle HTTP exceptions (404, etc.) without logging as errors."""
    request_id = generate_request_id()
    
    # For 404, log as INFO only (not error)
    if e.code == 404:
        logger.info(f"HTTP_404 path={request.path} request_id={request_id}")
    else:
        logger.warning(f"HTTP_{e.code} path={request.path} request_id={request_id} message={str(e)}")
    
    # Return JSON with the actual status code
    return jsonify({
        "error": e.name,
        "message": e.description,
        "code": e.code,
        "request_id": request_id
    }), e.code


@app.errorhandler(Exception)
def handle_exception(e):
    """Global exception handler for non-HTTP exceptions to prevent 502 errors and ensure JSON responses."""
    request_id = generate_request_id()
    
    # Log full traceback
    logger.exception(f"Unhandled exception (request_id={request_id}): {type(e).__name__}: {str(e)}")
    
    # Return JSON error response
    return jsonify({
        "error": "INTERNAL_ERROR",
        "message": "Server error",
        "request_id": request_id
    }), 500


@app.route('/', methods=['GET'])
def root():
    """Root endpoint."""
    return jsonify({"status": "ok"}), 200


@app.route('/favicon.ico', methods=['GET'])
def favicon():
    """Favicon endpoint - return 204 (no content)."""
    return '', 204


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"ok": True}), 200


def generate_ad_goal(product_name, product_description, attempt):
    """Generate a distinct advertising goal for the given attempt (1, 2, or 3)."""
    if not client:
        raise ValueError("OpenAI API key not configured")
    
    goal_prompt = f"""You are an advertising strategist. Generate a distinct advertising goal for attempt {attempt} of 3.

Product Name: {product_name}
Product Description: {product_description}

Requirements:
- This is attempt {attempt} of 3 total attempts
- Each attempt must have a DIFFERENT advertising goal
- Infer the target audience from product name + description only (age, lifestyle, needs, knowledge level, pains)
- Generate a specific, distinct advertising goal for this attempt
- The goal should be different from goals for attempts 1, 2, and 3

Return ONLY a concise description of the advertising goal (1-2 sentences), nothing else."""

    try:
        response = retry_openai_call(
            lambda: client.chat.completions.create(
                model=TEXT_MODEL,
                messages=[
                    {"role": "system", "content": "You are an advertising strategist. Generate distinct advertising goals."},
                    {"role": "user", "content": goal_prompt}
                ],
                temperature=0.9,
                max_tokens=150
            ),
            operation_name="ad_goal_generation"
        )
        
        ad_goal = response.choices[0].message.content.strip()
        logger.info(f"AD_GOAL attempt={attempt}: {ad_goal}")
        return ad_goal
    except RetryableError:
        # Propagate transient errors - do NOT consume attempt
        raise
    except Exception as e:
        # Fallback for non-transient errors only
        ad_goal = f"Advertising goal for attempt {attempt}: highlight different aspect of {product_name}"
        logger.info(f"AD_GOAL attempt={attempt}: {ad_goal} (fallback)")
        return ad_goal


def generate_headline_and_text(product_name, product_description, attempt, ad_goal):
    """Generate headline and exactly 50-word marketing text using OpenAI."""
    if not client:
        raise ValueError("OpenAI API key not configured")
    
    # Generate headline: 3-7 words, includes product_name, original (not copied from description)
    headline_prompt = f"""Generate a creative advertising headline for a product. Requirements:
- Must be exactly 3-7 words
- Must include the product name: "{product_name}"
- Must be original and NOT a quote or variation of the product description
- Must be a compelling promise or benefit statement
- Must align with the advertising goal: {ad_goal}
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
- Must align with the advertising goal: {ad_goal}
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


# NOTE: Old pick_two_objects function removed - Engine V2 only uses pick_two_objects_v2
# The old function used V1 terminology ("HYBRID", "overlap_assessment") which is not compliant with V2.
# V2 uses "PERFECT_HYBRID" and "overlap_class" (FULL_OVERLAP/SIMILAR_ONLY/NO_SIMILARITY).


def derive_audience_v2(product_name, product_description):
    """Derive target audience from product name and description (Engine V2 - 01H)."""
    if not client:
        raise ValueError("OpenAI API key not configured")
    
    audience_prompt = f"""You are an advertising strategist. Derive the target audience from the product information.

Product Name: {product_name}
Product Description: {product_description}

Requirements:
- Infer the target audience characteristics from product name + description only
- Consider: age, lifestyle, needs, knowledge level, pains, preferences
- Return a concise description of the target audience (2-3 sentences)

Return ONLY the audience description, nothing else."""
    
    try:
        response = retry_openai_call(
            lambda: client.chat.completions.create(
                model=TEXT_MODEL,
                messages=[
                    {"role": "system", "content": "You are an advertising strategist. Derive target audiences from products."},
                    {"role": "user", "content": audience_prompt}
                ],
                temperature=0.7,
                max_tokens=150
            ),
            operation_name="audience_derivation_v2"
        )
        
        audience = response.choices[0].message.content.strip()
        logger.info(f"AUDIENCE_V2: {audience}")
        return audience
    except Exception as e:
        logger.warning(f"AUDIENCE_V2 failed: {str(e)}, using fallback")
        return f"Target audience for {product_name}"


def generate_associations_v2(product_name, product_description, ad_goal, audience, n=100):
    """Generate exactly n associations for the ad goal (Engine V2).
    
    Args:
        product_name: Product name
        product_description: Product description
        ad_goal: Advertising goal
        audience: Target audience
        n: Number of associations to generate (default 100, can be increased on retry)
    
    Returns:
        List of exactly n associations
    """
    if not client:
        raise ValueError("OpenAI API key not configured")
    
    associations_prompt = f"""You are an ACE engine association generator. Generate EXACTLY {n} associations for an advertising goal.

Product Name: {product_name}
Product Description: {product_description}
Advertising Goal: {ad_goal}
Target Audience: {audience}

Requirements:
- Generate EXACTLY {n} associations (no more, no less)
- Associations should be related to the product, ad goal, and audience
- Return as a numbered list (1. association, 2. association, ...)
- Each association should be a short phrase (1-5 words)

Return ONLY the numbered list of {n} associations, nothing else."""
    
    try:
        response = retry_openai_call(
            lambda: client.chat.completions.create(
                model=TEXT_MODEL,
                messages=[
                    {"role": "system", "content": f"You are an ACE engine association generator. Always return exactly {n} associations."},
                    {"role": "user", "content": associations_prompt}
                ],
                temperature=0.9,
                max_tokens=2000
            ),
            operation_name="associations_generation_v2"
        )
        
        associations_text = response.choices[0].message.content.strip()
        # Parse numbered list
        associations = []
        for line in associations_text.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # Remove numbering/bullet
                clean_line = line.split('.', 1)[-1].strip()
                clean_line = clean_line.lstrip('- ').strip()
                if clean_line:
                    associations.append(clean_line)
        
        # Ensure we have exactly n (pad or truncate if needed)
        if len(associations) < n:
            logger.warning(f"ASSOCIATIONS_V2: Only got {len(associations)} associations, padding to {n}")
            while len(associations) < n:
                associations.append(f"association_{len(associations) + 1}")
        elif len(associations) > n:
            associations = associations[:n]
        
        logger.info(f"ASSOCIATIONS_V2: Generated {len(associations)} associations (requested {n})")
        return associations
    except Exception as e:
        logger.warning(f"ASSOCIATIONS_V2 failed: {str(e)}, using fallback")
        return [f"association_{i+1}" for i in range(n)]


def filter_to_physical_objects_v2(associations, strict_mode=False):
    """Filter associations to physical objects that are photographable with classic backgrounds (Engine V2).
    
    Args:
        associations: List of associations to filter
        strict_mode: If True, use stricter filtering instructions to get at least 60 physical objects
    
    Returns:
        List of physical objects
    """
    if not client:
        raise ValueError("OpenAI API key not configured")
    
    strict_instruction = ""
    if strict_mode:
        strict_instruction = "\n\nCRITICAL: You MUST return at least 60 physical objects. Be more lenient in filtering - include borderline cases that are still physical objects."
    
    filter_prompt = f"""You are an ACE engine object filter. Filter associations to physical objects only.

Associations ({len(associations)} total):
{chr(10).join(f"{i+1}. {assoc}" for i, assoc in enumerate(associations))}

Requirements:
- Filter to ONLY physical, real, photographable objects
- Objects must be simple, everyday, familiar physical objects
- NOT ideas, symbols, abstract concepts, or illustrations
- Objects must have a classic natural background (e.g., book on desk, tree in forest, cup on table)
- Do NOT include objects with text/logos/letters/numbers/external graphics (unless inherent like playing cards, dice dots, engraved compass letters)
- Return only the physical objects that pass the filter
{strict_instruction}

Return as a numbered list of physical objects only (no explanations)."""
    
    try:
        response = retry_openai_call(
            lambda: client.chat.completions.create(
                model=TEXT_MODEL,
                messages=[
                    {"role": "system", "content": "You are an ACE engine object filter. Return only physical objects."},
                    {"role": "user", "content": filter_prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            ),
            operation_name="object_filtering_v2"
        )
        
        filtered_text = response.choices[0].message.content.strip()
        physical_objects = []
        for line in filtered_text.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                clean_line = line.split('.', 1)[-1].strip()
                clean_line = clean_line.lstrip('- ').strip()
                if clean_line:
                    physical_objects.append(clean_line)
        
        logger.info(f"PHYSICAL_OBJECTS_V2: Filtered to {len(physical_objects)} objects (strict_mode={strict_mode})")
        return physical_objects
    except Exception as e:
        logger.warning(f"PHYSICAL_OBJECTS_V2 failed: {str(e)}, using first 20 associations as fallback")
        return associations[:20] if len(associations) >= 20 else associations


def pick_two_objects_v2(product_name, product_description, headline, ad_goal, used_pairs_list=None, attempt=None, request_id=None, num_associations=100):
    """Select two physical objects using V2 engine logic: 100 associations -> filter to physical -> select A/B with FULL_OVERLAP/SIMILAR_ONLY/NO_SIMILARITY.
    
    Args:
        product_name: Product name
        product_description: Product description
        headline: Generated headline
        ad_goal: Distinct advertising goal for this attempt
        used_pairs_list: List of canonical pair keys (A|B) that have been used (forbidden)
        attempt: Attempt number (1, 2, or 3)
        request_id: Request ID for logging
        num_associations: Number of associations to generate (default 100, can be increased by +10 on retry)
    
    Returns:
        dict with keys: A, B, C_projection_description, D_projection_description, overlap_class, 
                        layout, background_classic_of_C, c_is_dominant
    """
    if not client:
        raise ValueError("OpenAI API key not configured")
    
    if used_pairs_list is None:
        used_pairs_list = []
    
    # Step 1: Derive audience (01H)
    audience = derive_audience_v2(product_name, product_description)
    
    # Step 2: Generate exactly num_associations associations
    associations = generate_associations_v2(product_name, product_description, ad_goal, audience, num_associations)
    
    # Step 3: Filter to physical objects (with robustness: retry if < 40)
    physical_objects = filter_to_physical_objects_v2(associations, strict_mode=False)
    
    # Robustness: If filtered < 40, re-run with stricter instruction
    if len(physical_objects) < 40:
        logger.warning(f"PHYSICAL_OBJECTS_V2: Only {len(physical_objects)} objects (< 40), retrying with strict mode")
        physical_objects = filter_to_physical_objects_v2(associations, strict_mode=True)
        
        # If still < 40 after strict mode, regenerate associations and filter again (up to 2 cycles)
        if len(physical_objects) < 40:
            logger.warning(f"PHYSICAL_OBJECTS_V2: Still only {len(physical_objects)} objects after strict mode, regenerating associations")
            # Regenerate associations and filter again (one cycle)
            associations = generate_associations_v2(product_name, product_description, ad_goal, audience, num_associations)
            physical_objects = filter_to_physical_objects_v2(associations, strict_mode=True)
            
            # If still < 40, one more cycle
            if len(physical_objects) < 40:
                logger.warning(f"PHYSICAL_OBJECTS_V2: Still only {len(physical_objects)} objects after regeneration, one more cycle")
                associations = generate_associations_v2(product_name, product_description, ad_goal, audience, num_associations)
                physical_objects = filter_to_physical_objects_v2(associations, strict_mode=True)
    
    if len(physical_objects) < 2:
        logger.warning(f"PHYSICAL_OBJECTS_V2: Only {len(physical_objects)} objects, using fallback")
        physical_objects = ["product object", "complementary object"]
    
    # Build forbidden pairs list text
    forbidden_pairs_text = ""
    if used_pairs_list:
        readable_pairs = []
        for pair_key in used_pairs_list:
            parts = pair_key.split("|")
            if len(parts) == 2:
                readable_pairs.append(f"({parts[0]}, {parts[1]})")
        if readable_pairs:
            forbidden_pairs_text = f"\n\nCRITICAL: The following object pairs have already been used in previous attempts in this session and MUST NOT be selected (in any order): {', '.join(readable_pairs)}\nYou MUST select a DIFFERENT pair (A,B) that is NOT in this list."
    
    # Step 4: Select A and B from physical objects with overlap assessment
    selection_prompt = f"""You are an ACE engine V2 object selector. Select two physical objects from the provided list with strict geometric overlap assessment.

Product Name: {product_name}
Product Description: {product_description}
Headline: {headline}
Advertising Goal: {ad_goal}
Target Audience: {audience}

Physical Objects List ({len(physical_objects)} objects):
{chr(10).join(f"{i+1}. {obj}" for i, obj in enumerate(physical_objects))}
{forbidden_pairs_text}

Selection Rules:
- A = object with central meaning to the ad goal
- B = object used for conceptual emphasis (pairing by shape similarity)
- Both A and B must be selected from the physical objects list above
- Do NOT select objects with text/logos/letters/numbers/external graphics (unless inherent)

Projection Selection (CRITICAL):
- C_projection_description = how to view Object A to maximize its silhouette area (camera angle, perspective, orientation)
- D_projection_description = how to view Object B to maximize its silhouette area (camera angle, perspective, orientation)
- Both projections must be clean silhouettes with the largest visible dominant area
- Choose the view where the silhouette occupies the largest area, clean silhouette

Geometric Overlap Assessment (CRITICAL - V2):
Compare the simplified shapes E (from C) and F (from D) after permitted adjustments (scale/angle/proportion without distortion).
- "FULL_OVERLAP": E and F can achieve FULL geometric overlap - the projection silhouettes are nearly identical after permitted adjustments. This is clear and immediate to an average human eye. The final image would show ONE projection only (single object).
- "SIMILAR_ONLY": E and F are similar but NOT full overlap - they share shape similarity but cannot achieve full geometric overlap even with adjustments. The final image would show TWO separate projections.
- "NO_SIMILARITY": E and F have no clear shape similarity - an average human eye would not see them as similar. CRITICAL: NO_SIMILARITY pairs are INVALID and must NOT be returned as final. If you assess NO_SIMILARITY, you must select a different pair (A,B) from the list.

Layout Decision (V2):
- layout = "PERFECT_HYBRID" ONLY if overlap_class == "FULL_OVERLAP"
- layout = "SIDE_BY_SIDE" ONLY if overlap_class == "SIMILAR_ONLY"
- NEVER return overlap_class == "NO_SIMILARITY" as a final result - it is INVALID. If you assess NO_SIMILARITY, you must select a different pair.
- PERFECT_HYBRID means: the final image shows ONE projection only (single object). The other contributes only background/environment/context and must not appear as a separate object.
- SIDE_BY_SIDE means: the final image shows TWO separate objects/projections, no fusion/overlap/hybridization.

Background:
- background_classic_of_C = classic natural background for the dominant object (A's projection C)
- c_is_dominant = true (C always controls lighting/texture/composition/background)

Return ONLY valid JSON with these exact keys:
{{
  "A": "object name from the list",
  "B": "object name from the list",
  "C_projection_description": "how to view A to maximize silhouette area (largest silhouette, clean)",
  "D_projection_description": "how to view B to maximize silhouette area (largest silhouette, clean)",
  "overlap_class": "FULL_OVERLAP" or "SIMILAR_ONLY" or "NO_SIMILARITY",
  "layout": "PERFECT_HYBRID" (only if overlap_class is FULL_OVERLAP) or "SIDE_BY_SIDE",
  "background_classic_of_C": "description of classic natural background",
  "c_is_dominant": true
}}

Do not include any explanation or other text."""
    
    max_internal_retries = 6
    
    for retry_attempt in range(max_internal_retries):
        try:
            response = retry_openai_call(
                lambda: client.chat.completions.create(
                    model=TEXT_MODEL,
                    messages=[
                        {"role": "system", "content": "You are an ACE engine V2 object selector. Always return valid JSON only."},
                        {"role": "user", "content": selection_prompt}
                    ],
                    temperature=0.8,
                    response_format={"type": "json_object"}
                ),
                operation_name="object_selection_v2"
            )
            
            result = json.loads(response.choices[0].message.content.strip())
            
            # Validate fields exist
            A = result.get("A", "").strip()
            B = result.get("B", "").strip()
            C_projection_description = result.get("C_projection_description", "").strip()
            D_projection_description = result.get("D_projection_description", "").strip()
            overlap_class = result.get("overlap_class", "").strip()
            layout = result.get("layout", "SIDE_BY_SIDE")
            background_classic_of_C = result.get("background_classic_of_C", "")
            c_is_dominant = result.get("c_is_dominant", True)
            
            # Validate required fields
            if not A or not B:
                if retry_attempt < max_internal_retries - 1:
                    logger.warning(f"Object selection V2: Missing A or B, retrying... attempt={retry_attempt + 1}")
                    continue
                else:
                    raise ValueError("Object selection V2: Missing required fields A or B after max retries")
            
            if not C_projection_description or not D_projection_description:
                if retry_attempt < max_internal_retries - 1:
                    logger.warning(f"Object selection V2: Missing projection descriptions, retrying... attempt={retry_attempt + 1}")
                    continue
                else:
                    raise ValueError("Object selection V2: Missing projection descriptions after max retries")
            
            # Check if pair (A,B) is in forbidden pairs list
            pair_key = create_pair_key(A, B)
            if used_pairs_list and pair_key in used_pairs_list:
                req_id_str = f" request_id={request_id}" if request_id else ""
                logger.info(f"REJECTED_PAIR_V2{req_id_str} key={pair_key} (A={A}, B={B})")
                if retry_attempt < max_internal_retries - 1:
                    logger.info(f"FORBIDDEN_PAIR_HIT_V2 retrying selector... attempt={retry_attempt + 1} forbidden_pair={pair_key}")
                    continue  # Retry
                else:
                    raise ValueError(f"Object selection V2: Forbidden pair selected after max retries: {pair_key}")
            
            # Validate overlap_class - must be one of the allowed values
            if overlap_class not in ["FULL_OVERLAP", "SIMILAR_ONLY", "NO_SIMILARITY"]:
                if retry_attempt < max_internal_retries - 1:
                    logger.warning(f"Object selection V2: Invalid overlap_class '{overlap_class}', retrying... attempt={retry_attempt + 1}")
                    continue
                else:
                    raise ValueError(f"Object selection V2: Invalid overlap_class '{overlap_class}' after max retries")
            
            # Enforce layout based on overlap_class (V2 - HARD CONSTRAINT)
            if overlap_class == "FULL_OVERLAP":
                layout = "PERFECT_HYBRID"
            elif overlap_class == "SIMILAR_ONLY":
                layout = "SIDE_BY_SIDE"
            elif overlap_class == "NO_SIMILARITY":
                # NO_SIMILARITY is FORBIDDEN - must reject and repick (never return)
                if retry_attempt < max_internal_retries - 1:
                    logger.info(f"NO_SIMILARITY_V2 INVALID - retrying selector... attempt={retry_attempt + 1} A={A} B={B}")
                    continue  # Retry to get a valid pair
                else:
                    # Max retries reached - raise error to trigger association regeneration
                    logger.warning(f"NO_SIMILARITY_V2 max retries reached - will trigger association regeneration")
                    raise ValueError("NO_SIMILARITY pair selected after max retries - need to regenerate associations")
            
            # Validate layout matches overlap_class (HARD CONSTRAINT)
            if layout == "PERFECT_HYBRID" and overlap_class != "FULL_OVERLAP":
                if retry_attempt < max_internal_retries - 1:
                    logger.warning(f"Object selection V2: Layout mismatch (PERFECT_HYBRID without FULL_OVERLAP), retrying... attempt={retry_attempt + 1}")
                    continue
                else:
                    raise ValueError("Object selection V2: Layout mismatch (PERFECT_HYBRID without FULL_OVERLAP) after max retries")
            
            if layout == "SIDE_BY_SIDE" and overlap_class != "SIMILAR_ONLY":
                if retry_attempt < max_internal_retries - 1:
                    logger.warning(f"Object selection V2: Layout mismatch (SIDE_BY_SIDE without SIMILAR_ONLY), retrying... attempt={retry_attempt + 1}")
                    continue
                else:
                    raise ValueError("Object selection V2: Layout mismatch (SIDE_BY_SIDE without SIMILAR_ONLY) after max retries")
            
            # Return successful selection (only FULL_OVERLAP or SIMILAR_ONLY)
            return {
                "A": A,
                "B": B,
                "C_projection_description": C_projection_description,
                "D_projection_description": D_projection_description,
                "overlap_class": overlap_class,
                "layout": layout,
                "background_classic_of_C": background_classic_of_C,
                "c_is_dominant": c_is_dominant
            }
        except ValueError as e:
            # Re-raise ValueError (NO_SIMILARITY, validation errors) to trigger regeneration
            if retry_attempt < max_internal_retries - 1:
                logger.warning(f"Object selection V2 failed (retry {retry_attempt + 1}): {str(e)}")
                continue
            else:
                # Max retries reached - propagate error to trigger association regeneration
                raise
        except Exception as e:
            if retry_attempt < max_internal_retries - 1:
                logger.warning(f"Object selection V2 failed (retry {retry_attempt + 1}): {str(e)}")
                continue
            else:
                # Max retries reached - raise error to trigger association regeneration
                raise ValueError(f"Object selection V2 failed after {max_internal_retries} retries: {str(e)}")
    
    # Should not reach here, but if we do, raise error
    raise ValueError("Object selection V2: Max retries reached without valid result")


def build_image_prompt_v2(product_name, product_description, headline, A, B, C_projection_description, D_projection_description, layout, background_classic_of_C, ad_size):
    """Build image prompt using V2 engine rules (Engine V2)."""
    
    # OBJECT IDENTITY (CRITICAL - must be at the top)
    object_identity = f"""OBJECT IDENTITY (CRITICAL - MANDATORY):
- Object A = {A}
- Object B = {B}
- These are the two physical objects that MUST appear in the image according to the layout rules below."""
    
    # Camera angle instructions based on projection descriptions
    camera_instruction = f"""CAMERA ANGLE INSTRUCTIONS (CRITICAL):
- Object A (C projection): {C_projection_description}
- Object B (D projection): {D_projection_description}
- The camera angle and perspective MUST match these projection descriptions exactly.
- Both objects must be viewed from the angles that maximize their silhouette areas (largest silhouette, clean)."""
    
    if layout == "PERFECT_HYBRID":
        layout_instruction = f"""Create a PERFECT_HYBRID (Engine V2 - MANDATORY):
- The final image MUST show exactly ONE object/projection only.
- Object B ({B})'s projection (D) must REPLACE Object A ({A})'s projection (C) in the same footprint through full geometric overlap.
- This is REPLACEMENT, NOT stacking, NOT blending, NOT "{A} with {B} on top", NOT "{B} placed on {A}".
- D REPLACES a structural element of C - they become ONE unified projection where D replaces part of C within C's classic structure.
- The other object (the one not shown as the main projection) contributes ONLY background/environment/context and must NOT appear as a separate object.
- The final silhouette must read as ONE single physical object - no visible seams of two separate items.
- No visible boundaries, edges, or seams that suggest two separate items.
- The composition must read as a single, cohesive physical entity.

CRITICAL REPLACEMENT RULE (MANDATORY - NOT STACKING):
- D must REPLACE C in the same footprint through geometric embedding.
- D replaces an equivalent structural element of C (e.g., if C is a shelf of books, D replaces ONE BOOK with a laptop).
- This is REPLACEMENT, NOT positioning, NOT stacking, NOT blending.
- FORBIDDEN: "on top of", "resting on", "placed on", "sitting on", "lying on", "inside of", "next to", "with {B} on top", "{A} with {B}", or any stacking/positioning arrangement.
- FORBIDDEN: D as a separate object positioned above, beside, or within C.
- FORBIDDEN: Any visual arrangement that shows {B} placed on {A} or {A} with {B} on top.
- REQUIRED: D must be geometrically embedded INTO C's structure, replacing it in the same footprint.

CRITICAL: NOT A SINGLE NORMAL OBJECT (MANDATORY - STRICT ENFORCEMENT):
- This is INVALID if it looks like a single normal object without clear hybrid characteristics.
- It MUST clearly be a replacement hybrid of Object A ({A}) and Object B ({B}).
- The viewer must be able to recognize that this is a hybrid where {B} replaces a structural element of {A}.
- The single visible object MUST include unmistakable signature features from BOTH Object A ({A}) AND Object B ({B}).
- The hybrid must be visually recognizable as a fusion where {B}'s characteristics replace part of {A}'s structure.
- If it looks like a single normal object with no hybrid characteristics, it is INVALID.
- This is INVALID if it looks like just Object A ({A}) alone or just Object B ({B}) alone. It must clearly be a replacement hybrid of BOTH.
- The final image must show clear evidence that {B} has replaced a structural element of {A} - the viewer should see both objects' characteristics in the single visible projection.
- If you cannot make it clearly show both objects' characteristics in one unified replacement hybrid, do SIDE_BY_SIDE instead.

MUTUALLY-EXCLUSIVE RULE FOR PERFECT_HYBRID (CRITICAL):
- This layout is PERFECT_HYBRID - it MUST be ONE single projection ONLY.
- REQUIRED: complete fusion into one unified object through replacement - no visible separation.
- REQUIRED: The single visible object must show unmistakable signature features from BOTH {A} and {B}.
- FORBIDDEN: separate objects, "next to", "placed on", "resting on", "side by side", or any arrangement that shows two distinct items.
- FORBIDDEN: any side-by-side appearance, two distinct objects, or visual separation.
- FORBIDDEN: showing Object A ({A}) and Object B ({B}) separately in any way.
- FORBIDDEN: showing only one object without clear hybrid characteristics from both.
- The image MUST be purely PERFECT_HYBRID (one fused object through replacement) - NO side-by-side elements whatsoever.
- If the image shows ANY side-by-side appearance or two separate objects, it is INVALID.
- If the image shows a single normal object without clear hybrid characteristics, it is INVALID."""
    else:  # SIDE_BY_SIDE
        layout_instruction = f"""Place Object A ({A}) (C projection) and Object B ({B}) (D projection) SIDE_BY_SIDE at parallel angles.
- Highlight maximal similar area between the projections.
- Place them close together, emphasizing their shape similarity.
- Both objects must be viewed from angles matching their projection descriptions (parallel angles for side-by-side).

MUTUALLY-EXCLUSIVE RULE FOR SIDE_BY_SIDE (CRITICAL):
- This layout is SIDE_BY_SIDE - it MUST show TWO separate objects/projections clearly separated.
- REQUIRED: Both Object A ({A}) and Object B ({B}) must be visible as separate physical objects in the same frame.
- REQUIRED: Two distinct, separate objects placed side by side.
- REQUIRED: Clear visual separation between Object A ({A}) and Object B ({B}).
- REQUIRED: Both objects must be clearly identifiable as separate entities.
- FORBIDDEN: showing only one object, fusion, overlap, merging, hybridization, or any arrangement that makes them look like one object.
- FORBIDDEN: geometric embedding, projection replacement, or structural integration.
- FORBIDDEN: any visual arrangement that suggests the objects are fused or merged.
- The image MUST be purely SIDE_BY_SIDE (two separate objects) - NO hybrid/fusion elements whatsoever.
- If the image shows ANY fusion or hybrid appearance, it is INVALID.
- INVALID if only one object is visible. Both A and B must be visible."""
    
    image_prompt = f"""YOU ARE A PROFESSIONAL ADVERTISING PHOTOGRAPHER (ACE Engine V2).
YOU MUST FOLLOW ALL RULES BELOW. NO EXCEPTIONS.

{object_identity}

{camera_instruction}

LAYOUT INSTRUCTION:
{layout_instruction}

BACKGROUND AND LIGHTING RULE (CRITICAL - ENGINE V2):
- The background MUST be: {background_classic_of_C}
- This is the classic natural background of Object A (the dominant object C).
- C (Object A's projection) controls lighting, texture, composition, and background.
- D (Object B's projection) must NOT change C's background or lighting.
- Background and lighting must match ONLY C. D must not affect background/lighting.
- NEVER use studio backgrounds, black backgrounds, gradients, or abstract scenes.
- The entire scene's lighting, texture, and composition must be determined by C only.

STYLE RULES (Engine V2 - 07H):
- Ultra-realistic photography ONLY.
- Looks like a real camera photograph.
- Natural lighting, correct perspective, real materials.
- NO illustration, NO 3D render, NO CGI, NO AI-art look.
- Full realistic photo only.

HEADLINE RULES (INSIDE IMAGE - Engine V2 - 06H):
- The headline "{headline}" MUST appear INSIDE the image.
- The headline must be clearly readable and visually integrated into the composition.
- Headline must be 3-7 words, includes product name "{product_name}", original promise (not quote/description).
- Place headline near/above/below silhouettes, NEVER on them.
- The headline must be placed on the background area (on A's background), NEVER on top of the projections.

SAFE MARGIN RULE (CRITICAL - PREVENTS CLIPPING):
- ALL text (the headline) MUST be positioned at least 8-12% away from EVERY edge of the canvas.
- Keep the headline at least 8-12% from the top edge, bottom edge, left edge, and right edge.
- This ensures the headline is NEVER cut off or clipped at the edges.
- The headline MUST remain fully visible within the safe margin zone.

COMPOSITION RULE (HEADLINE PLACEMENT):
- The headline MUST be placed on the background area (on A's background), NEVER on top of the projections.
- The headline must be positioned in an area of the background that does NOT overlap with Object A or Object B projections.
- The headline must remain fully visible and readable, positioned in a clear background area.
- If the headline is too long to fit safely, shorten or rephrase it to 3-7 words while still including the product name "{product_name}".

PRODUCT NAME: {product_name}
PRODUCT DESCRIPTION: {product_description}

TASK:
Generate ONE final advertising image that fully follows ALL the rules above (ACE Engine V2).
Do not explain. Do not describe."""
    
    return image_prompt


def generate_image(product_name, product_description, headline, ad_size, attempt, ad_goal, used_A_list=None, used_pairs_list=None, request_id=None):
    """Generate image using OpenAI DALL-E with ACE Engine V2 logic."""
    if not client:
        raise ValueError("OpenAI API key not configured")
    
    # Map ad_size to OpenAI format
    size_map = {
        "1024x1024": "1024x1024",
        "1024x1536": "1024x1536",
        "1536x1024": "1536x1024"
    }
    openai_size = size_map.get(ad_size, "1024x1024")
    
    # Log used_pairs list before selection
    req_id_str = f" request_id={request_id}" if request_id else ""
    if used_pairs_list:
        logger.info(f"USED_PAIRS{req_id_str}: {used_pairs_list}")
    else:
        logger.info(f"USED_PAIRS{req_id_str}: [] (first attempt)")
    
    # Step 1: Pick two objects using V2 engine logic (100 associations -> filter -> select A/B with overlap_class)
    # Handle retry with +10 associations for NO_SIMILARITY (V2 policy)
    # For attempt==2, add FULL_OVERLAP search loop (up to 12 tries)
    num_associations = 100
    max_association_retries = 3  # Max 3 retries with +10 associations each
    objects = None
    
    # For attempt==2, do a FULL_OVERLAP search loop (up to 12 tries)
    if attempt == 2:
        max_hybrid_search_tries = 12
        
        logger.info(f"HYBRID_SEARCH_V2 attempt=2 starting search for FULL_OVERLAP (max {max_hybrid_search_tries} tries) request_id={request_id}")
        
        for search_try in range(max_hybrid_search_tries):
            try:
                # Call pick_two_objects_v2 - it will handle association generation and filtering internally
                temp_objects = pick_two_objects_v2(product_name, product_description, headline, ad_goal, used_pairs_list, attempt, request_id, num_associations)
                overlap_class = temp_objects.get("overlap_class", "")
                
                logger.info(f"HYBRID_SEARCH_V2 attempt=2 try={search_try + 1} overlap_class={overlap_class} A={temp_objects.get('A', '')} B={temp_objects.get('B', '')}")
                
                # Check if we found FULL_OVERLAP
                if overlap_class == "FULL_OVERLAP":
                    # Found a valid PERFECT_HYBRID candidate!
                    objects = temp_objects
                    objects["layout"] = "PERFECT_HYBRID"
                    logger.info(f"HYBRID_FOUND_V2 attempt=2 try={search_try + 1} A={objects['A']} B={objects['B']} overlap_class={overlap_class}")
                    break
                elif overlap_class == "SIMILAR_ONLY":
                    # Valid SIDE_BY_SIDE candidate, but continue searching for FULL_OVERLAP
                    # Store it temporarily but keep searching
                    if objects is None:
                        objects = temp_objects
                        objects["layout"] = "SIDE_BY_SIDE"
                    continue
                else:
                    # NO_SIMILARITY or invalid - continue searching
                    logger.info(f"HYBRID_SEARCH_V2 attempt=2 try={search_try + 1} overlap_class={overlap_class} (invalid), continuing search")
                    continue
                    
            except ValueError as e:
                # NO_SIMILARITY error or validation error - continue searching
                logger.info(f"HYBRID_SEARCH_V2 attempt=2 try={search_try + 1} error: {str(e)}, continuing search")
                continue
            except Exception as e:
                logger.warning(f"HYBRID_SEARCH_V2 attempt=2 try={search_try + 1} error: {str(e)}, continuing search")
                continue  # Continue searching on error
        
        # If we found FULL_OVERLAP, use it
        if objects is not None and objects.get("overlap_class") == "FULL_OVERLAP":
            A = objects["A"]
            B = objects["B"]
            C_projection_description = objects["C_projection_description"]
            D_projection_description = objects["D_projection_description"]
            overlap_class = objects["overlap_class"]
            layout = objects["layout"]
            background_classic_of_C = objects["background_classic_of_C"]
            c_is_dominant = objects.get("c_is_dominant", True)
        elif objects is not None and objects.get("overlap_class") == "SIMILAR_ONLY":
            # Found valid SIDE_BY_SIDE (but not PERFECT_HYBRID)
            logger.info(f"HYBRID_NOT_FOUND_V2 attempt=2 found valid SIDE_BY_SIDE A={objects['A']} B={objects['B']}")
            A = objects["A"]
            B = objects["B"]
            C_projection_description = objects["C_projection_description"]
            D_projection_description = objects["D_projection_description"]
            overlap_class = objects["overlap_class"]
            layout = objects["layout"]
            background_classic_of_C = objects["background_classic_of_C"]
            c_is_dominant = objects.get("c_is_dominant", True)
        else:
            # No valid result found, fall through to normal retry logic
            logger.warning(f"HYBRID_NOT_FOUND_V2 attempt=2 no valid result found, using normal retry logic")
            objects = None
    
    # Normal retry logic (for attempts 1, 3, or if attempt 2 search failed)
    if objects is None:
        for assoc_retry in range(max_association_retries):
            try:
                objects = pick_two_objects_v2(product_name, product_description, headline, ad_goal, used_pairs_list, attempt, request_id, num_associations)
                A = objects["A"]
                B = objects["B"]
                C_projection_description = objects["C_projection_description"]
                D_projection_description = objects["D_projection_description"]
                overlap_class = objects["overlap_class"]
                layout = objects["layout"]
                background_classic_of_C = objects["background_classic_of_C"]
                c_is_dominant = objects.get("c_is_dominant", True)
                
                # Validate overlap_class is valid (FULL_OVERLAP or SIMILAR_ONLY only)
                if overlap_class not in ["FULL_OVERLAP", "SIMILAR_ONLY"]:
                    # NO_SIMILARITY or invalid - increase associations and retry
                    if assoc_retry < max_association_retries - 1:
                        num_associations += 10
                        logger.info(f"NO_SIMILARITY_V2 retrying with +10 associations (now {num_associations}) attempt={assoc_retry + 1}")
                        continue
                    else:
                        raise ValueError(f"Object selection V2: Invalid overlap_class '{overlap_class}' after max retries")
                
                # Success - break out of retry loop
                break
            except ValueError as e:
                # This is the NO_SIMILARITY error or validation error from pick_two_objects_v2 - trigger association regeneration
                if assoc_retry < max_association_retries - 1:
                    num_associations += 10
                    logger.info(f"NO_SIMILARITY_V2 triggering association regeneration (now {num_associations}) attempt={assoc_retry + 1}")
                    continue
                else:
                    # Max retries reached - raise error (will be caught as retryable)
                    logger.warning(f"Object selection V2 failed after {max_association_retries} retries: {str(e)}")
                    raise RetryableError(f"Object selection failed after {max_association_retries} retries: {str(e)}", 503)
            except Exception as e:
                if assoc_retry < max_association_retries - 1:
                    num_associations += 10
                    logger.warning(f"Object selection V2 failed (retry {assoc_retry + 1}): {str(e)}, increasing associations to {num_associations}")
                    continue
                else:
                    # Max retries reached - raise error (will be caught as retryable)
                    logger.warning(f"Object selection V2 failed after {max_association_retries} retries: {str(e)}")
                    raise RetryableError(f"Object selection failed after {max_association_retries} retries: {str(e)}", 503)
    
    # Log successful selection of pair for this attempt
    pair_key = create_pair_key(A, B)
    req_id_str = f" request_id={request_id}" if request_id else ""
    logger.info(f"SELECTED_PAIR_V2{req_id_str} attempt={attempt} key={pair_key} (A={A}, B={B})")
    
    # Debug log: print selected A, B, layout, overlap_class, and openai_size (NOT full prompt, NOT secrets)
    logger.info(f"SELECTED A/B V2 attempt={attempt}: A={A}, B={B}, layout={layout}, overlap_class={overlap_class}, ad_size={ad_size} (OpenAI size: {openai_size})")
    
    # Log layout decision
    req_id_str = f" request_id={request_id}" if request_id else ""
    logger.info(f"LAYOUT_DECISION_V2{req_id_str} layout={layout}")
    
    # Step 2: Build image prompt using V2 engine rules
    image_prompt = build_image_prompt_v2(product_name, product_description, headline, A, B, C_projection_description, D_projection_description, layout, background_classic_of_C, ad_size)
    
    # Debug log: V2 render mode
    render_mode = "PERFECT_HYBRID" if layout == "PERFECT_HYBRID" else "SIDE_BY_SIDE"
    req_id_str = f" request_id={request_id}" if request_id else ""
    logger.info(f"V2_RENDER{req_id_str} layout={layout} A={A} B={B} mode={render_mode}")

    # Debug log: print image prompt, ad_size, and attempt
    logger.debug("=== IMAGE PROMPT SENT TO OPENAI (V2) ===")
    logger.debug(image_prompt)
    logger.debug("=== END IMAGE PROMPT ===")

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
        logger.debug(f"OpenAI image response keys: {response_keys}")
        
        # Use b64_json from OpenAI response (no URL fetch needed)
        image_base64 = response.data[0].b64_json
        
        image_data_url = f"data:image/jpeg;base64,{image_base64}"
        # Return image_data_url, selected A, and selected B for tracking
        return image_data_url, A, B
    except RetryableError:
        # Propagate transient errors - do NOT consume attempt
        raise
    except Exception as e:
        # Handle non-transient errors
        raise ValueError(f"Image generation failed: {str(e)}")


def create_zip_in_memory(image_base64, marketing_text, attempt, timeout_seconds=5):
    """Create a ZIP file in memory containing the image and text file.
    
    Memory-safe implementation:
    - Builds ZIP incrementally using BytesIO
    - Explicitly releases image data after writing
    - Closes all buffers after use
    - Does not keep multiple base64 copies in memory
    
    Args:
        image_base64: Base64-encoded image (with or without data URL prefix)
        marketing_text: Marketing text content
        attempt: Attempt number (1, 2, or 3)
        timeout_seconds: Maximum time allowed for ZIP creation (default 5)
    
    Returns:
        tuple: (zip_base64, size_bytes)
    
    Raises:
        TimeoutError: If ZIP creation takes longer than timeout_seconds
        Exception: If ZIP creation fails
    """
    start_time = time.time()
    zip_buffer = None
    image_data = None
    
    try:
        # Extract base64 part if it's a data URL
        if ',' in image_base64:
            image_base64_only = image_base64.split(',')[1]
        else:
            image_base64_only = image_base64
        
        # Check timeout before starting
        elapsed = time.time() - start_time
        if elapsed >= timeout_seconds:
            raise TimeoutError(f"ZIP creation timeout: {elapsed:.2f}s >= {timeout_seconds}s")
        
        # Decode image from base64 (do this once, release immediately after use)
        image_data = base64.b64decode(image_base64_only)
        
        # Check timeout after decode
        elapsed = time.time() - start_time
        if elapsed >= timeout_seconds:
            del image_data  # Release before raising
            raise TimeoutError(f"ZIP creation timeout: {elapsed:.2f}s >= {timeout_seconds}s")
        
        # Create ZIP buffer
        zip_buffer = io.BytesIO()
        
        # Build ZIP incrementally
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Write image to ZIP
            zip_file.writestr(f"ad_{attempt}.jpg", image_data)
            
            # Check timeout after writing image
            elapsed = time.time() - start_time
            if elapsed >= timeout_seconds:
                raise TimeoutError(f"ZIP creation timeout: {elapsed:.2f}s >= {timeout_seconds}s")
            
            # Immediately release image_data after writing (don't keep it in memory)
            del image_data
            image_data = None
            
            # Add marketing text file
            zip_file.writestr(f"ad_{attempt}.txt", marketing_text)
            
            # Check timeout after writing text
            elapsed = time.time() - start_time
            if elapsed >= timeout_seconds:
                raise TimeoutError(f"ZIP creation timeout: {elapsed:.2f}s >= {timeout_seconds}s")
        
        # Get ZIP bytes
        zip_buffer.seek(0)
        zip_bytes = zip_buffer.getvalue()
        size_bytes = len(zip_bytes)
        
        # Encode to base64
        zip_base64 = base64.b64encode(zip_bytes).decode('utf-8')
        
        # Explicitly close and release buffer
        zip_buffer.close()
        zip_buffer = None
        
        # Release zip_bytes reference (base64 encoding already done)
        del zip_bytes
        
        return zip_base64, size_bytes
        
    except TimeoutError:
        # Re-raise timeout errors
        raise
    except Exception as e:
        # Clean up on any error
        if image_data is not None:
            del image_data
        if zip_buffer is not None:
            try:
                zip_buffer.close()
            except:
                pass
        raise


@app.route('/generate', methods=['POST'], strict_slashes=False)
@app.route('/generate/', methods=['POST'], strict_slashes=False)
@app.route('/api/generate', methods=['POST'], strict_slashes=False)
@app.route('/api/generate/', methods=['POST'], strict_slashes=False)
@app.route('/api/generate-one', methods=['POST'], strict_slashes=False)
@app.route('/api/generate-one/', methods=['POST'], strict_slashes=False)
def generate():
    """Generate a single ad with real OpenAI generation (Phase 2).
    
    Handles POST requests to:
    - /generate
    - /generate/
    - /api/generate
    - /api/generate/
    - /api/generate-one
    - /api/generate-one/
    
    Session management:
    - Attempt 1: Creates new session_id, locks inputs, returns session_id
    - Attempts 2-3: Requires session_id, uses locked inputs from server state
    - After attempt 3: Session is deleted (full reset)
    """
    # Generate request ID for this request
    request_id = generate_request_id()
    
    # Log route hit
    route_path = request.path
    logger.info(f"ROUTE_HIT {route_path} request_id={request_id}")
    
    # Clean up expired sessions before processing
    cleanup_expired_sessions()
    
    # Top-level try/except to catch ALL exceptions and prevent 502s
    try:
        # Get JSON body
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "Request body must be JSON", "request_id": request_id}), 400
        
        # Get session_id (optional for attempt 1, required for attempts 2-3)
        session_id = data.get("session_id")
        attempt = data.get("attempt")
        
        # Validate attempt
        if attempt is None:
            return jsonify({"error": "attempt is required and must be 1, 2, or 3", "request_id": request_id}), 400
        
        if not isinstance(attempt, int) or attempt not in [1, 2, 3]:
            return jsonify({"error": "attempt must be exactly 1, 2, or 3", "request_id": request_id}), 400
        
        # Session handling
        if attempt == 1:
            # Attempt 1: Create new session
            if session_id and session_id in session_storage:
                # Session already exists - this is an error (attempt 1 should create new session)
                return jsonify({"error": "session_id provided for attempt 1, but session already exists. Use attempt 2 or 3 to continue existing session.", "request_id": request_id}), 400
            
            # Get inputs from request
            product_name = data.get("product_name")
            product_description = data.get("product_description")
            ad_size = data.get("ad_size")
            
            # Validate required fields for attempt 1
            if not product_name:
                return jsonify({"error": "product_name is required and must be a non-empty string", "request_id": request_id}), 400
            
            if not product_description:
                return jsonify({"error": "product_description is required and must be a non-empty string", "request_id": request_id}), 400
            
            if not ad_size:
                return jsonify({"error": "ad_size is required and must be a non-empty string", "request_id": request_id}), 400
            
            if not isinstance(product_name, str) or not product_name.strip():
                return jsonify({"error": "product_name must be a non-empty string", "request_id": request_id}), 400
            
            if not isinstance(product_description, str) or not product_description.strip():
                return jsonify({"error": "product_description must be a non-empty string", "request_id": request_id}), 400
            
            if not isinstance(ad_size, str) or not ad_size.strip():
                return jsonify({"error": "ad_size must be a non-empty string", "request_id": request_id}), 400
            
            if ad_size not in VALID_AD_SIZES:
                return jsonify({
                    "error": f"ad_size must be exactly one of: {', '.join(sorted(VALID_AD_SIZES))}",
                    "request_id": request_id
                }), 400
            
            # Create new session
            new_session_id = str(uuid.uuid4())
            session_storage[new_session_id] = {
                "product_name": product_name.strip(),
                "product_description": product_description.strip(),
                "selected_size": ad_size.strip(),
                "attempt": 1,
                "used_pairs": [],
                "created_at": time.time()
            }
            session_id = new_session_id
            logger.info(f"SESSION_CREATED session_id={session_id} request_id={request_id}")
        
        else:
            # Attempts 2-3: Require session_id and use locked inputs
            if not session_id:
                return jsonify({"error": "session_id is required for attempts 2 and 3", "request_id": request_id}), 400
            
            if session_id not in session_storage:
                return jsonify({"error": "session_id not found. Session may have expired or been reset.", "request_id": request_id}), 400
            
            # Get locked inputs from session state (ignore any client-provided values)
            session_data = session_storage[session_id]
            product_name = session_data["product_name"]
            product_description = session_data["product_description"]
            ad_size = session_data["selected_size"]
            
            # Validate attempt matches session state
            if session_data["attempt"] != attempt - 1:
                return jsonify({
                    "error": f"Session state mismatch. Expected attempt {session_data['attempt'] + 1}, got {attempt}",
                    "request_id": request_id
                }), 400
            
            # Update session attempt counter
            session_data["attempt"] = attempt
            logger.info(f"SESSION_CONTINUE session_id={session_id} attempt={attempt} request_id={request_id}")
        
        # Get used_pairs from session state
        session_data = session_storage[session_id]
        used_pairs_list = session_data["used_pairs"]
        
        # Check OpenAI API key - fail fast with clear error
        if not OPENAI_API_KEY or not OPENAI_API_KEY.strip() or not client:
            return jsonify({"error": "Server misconfigured: OPENAI_API_KEY is not set", "request_id": request_id}), 500
        
        # Log generation start
        logger.info(f"GEN_START request_id={request_id} session_id={session_id} attempt={attempt} size={ad_size} product_name={product_name[:50]}")
        
        # Log used_pairs for this session
        if used_pairs_list:
            logger.info(f"USED_PAIRS session_id={session_id} request_id={request_id}: {used_pairs_list}")
        else:
            logger.info(f"USED_PAIRS session_id={session_id} request_id={request_id}: [] (first attempt)")
        
        # Stage: TEXT_GOAL - Generate distinct ad_goal for this attempt
        logger.info(f"GEN_STAGE request_id={request_id} stage=TEXT_GOAL")
        try:
            ad_goal = generate_ad_goal(product_name, product_description, attempt)
            logger.info(f"OPENAI_TEXT_OK request_id={request_id} stage=TEXT_GOAL")
        except Exception as e:
            logger.exception(f"GEN_FAIL request_id={request_id} stage=TEXT_GOAL error={type(e).__name__}: {str(e)}")
            raise
        
        # Stage: TEXT_OBJECTS - Generate headline and marketing text (with ad_goal)
        logger.info(f"GEN_STAGE request_id={request_id} stage=TEXT_OBJECTS")
        try:
            headline, marketing_text = generate_headline_and_text(product_name, product_description, attempt, ad_goal)
            logger.info(f"OPENAI_TEXT_OK request_id={request_id} stage=TEXT_OBJECTS")
        except Exception as e:
            logger.exception(f"GEN_FAIL request_id={request_id} stage=TEXT_OBJECTS error={type(e).__name__}: {str(e)}")
            raise
        
        # Stage: IMAGE - Generate image (with ad_goal, used_pairs_list, and request_id) - returns (image_data_url, selected_A, selected_B)
        # Note: used_A_list is no longer needed - we only track pairs now
        logger.info(f"GEN_STAGE request_id={request_id} stage=IMAGE")
        try:
            image_data_url, selected_A, selected_B = generate_image(product_name, product_description, headline, ad_size, attempt, ad_goal, None, used_pairs_list, request_id)
            logger.info(f"OPENAI_IMAGE_OK request_id={request_id} stage=IMAGE")
        except Exception as e:
            logger.exception(f"GEN_FAIL request_id={request_id} stage=IMAGE error={type(e).__name__}: {str(e)}")
            raise
        
        # Track the selected pair (A,B) in session state (add to used_pairs_list if not already present)
        # NOTE: This only happens on successful generation - transient errors do NOT update used_pairs
        if selected_A and selected_B:
            pair_key = create_pair_key(selected_A, selected_B)
            if pair_key not in used_pairs_list:
                session_data["used_pairs"].append(pair_key)
                logger.info(f"TRACKED used_pair session_id={session_id} request_id={request_id}: {pair_key} (A={selected_A}, B={selected_B}) (total: {len(session_data['used_pairs'])})")
            else:
                logger.warning(f"WARNING: Selected pair '{pair_key}' (A={selected_A}, B={selected_B}) was already in used_pairs_list, but was selected anyway session_id={session_id} request_id={request_id}")
        
        # Stage: ZIP - Extract base64 from data URL for ZIP creation
        logger.info(f"GEN_STAGE request_id={request_id} stage=ZIP")
        try:
            # Extract base64 from data URL (keep reference minimal)
            image_base64 = image_data_url.split(',')[1] if ',' in image_data_url else image_data_url
            
            # Create ZIP in memory with timeout guard (5 seconds max)
            # This prevents worker crash from memory pressure or slow ZIP creation
            zip_base64, zip_size_bytes = create_zip_in_memory(image_base64, marketing_text, attempt, timeout_seconds=5)
            
            # Immediately release image_base64 reference after ZIP creation
            del image_base64
            
            # Log ZIP ready with size
            logger.info(f"ZIP_READY request_id={request_id} size_bytes={zip_size_bytes}")
            logger.info(f"ZIP_OK request_id={request_id} stage=ZIP")
        except TimeoutError as e:
            # ZIP creation took too long - return 503 (retryable) without consuming attempt
            logger.warning(f"GEN_FAIL request_id={request_id} stage=ZIP timeout: {str(e)} - RETRYABLE_ERROR (no attempt consumed)")
            logger.info(f"RETURNING RETRYABLE_ERROR (no attempt consumed) request_id={request_id}")
            return jsonify({
                "error": "RETRYABLE_ERROR",
                "code": 503,
                "message": "Server under load. Please try again.",
                "request_id": request_id
            }), 503
        except Exception as e:
            # Other ZIP errors - return 503 (retryable) without consuming attempt
            logger.exception(f"GEN_FAIL request_id={request_id} stage=ZIP error={type(e).__name__}: {str(e)} - RETRYABLE_ERROR (no attempt consumed)")
            logger.info(f"RETURNING RETRYABLE_ERROR (no attempt consumed) request_id={request_id}")
            return jsonify({
                "error": "RETRYABLE_ERROR",
                "code": 503,
                "message": "Server under load. Please try again.",
                "request_id": request_id
            }), 503
        
        # Return single ad object with session_id
        ad = {
            "ad_id": attempt,
            "headline": headline,
            "marketing_text_50_words": marketing_text,
            "image_data_url": image_data_url,
            "zip_base64": zip_base64,
            "zip_filename": f"ad_{attempt}.zip",
            "session_id": session_id,
            "request_id": request_id
        }
        
        # After attempt 3 succeeds, delete the session (full reset)
        if attempt == 3:
            del session_storage[session_id]
            logger.info(f"SESSION_RESET session_id={session_id} request_id={request_id} (attempt 3 completed, session deleted)")
        
        logger.info(f"Successfully generated ad (request_id={request_id}, attempt={attempt}, session_id={session_id})")
        return jsonify(ad), 200
    
    except RetryableError as e:
        # Transient OpenAI error after max retries - return 503
        # This does NOT consume an attempt - used_A is only updated after successful generation
        logger.exception(f"GEN_FAIL request_id={request_id} stage=openai_retry code={e.code} - RETRYABLE_ERROR (no attempt consumed)")
        logger.info(f"RETURNING RETRYABLE_ERROR (no attempt consumed) request_id={request_id}")
        return jsonify({
            "error": "RETRYABLE_ERROR",
            "code": e.code,
            "message": "Temporary overload. Please try again.",
            "request_id": request_id
        }), 503
    except ValueError as e:
        # Non-transient validation errors - return 400
        logger.exception(f"GEN_FAIL request_id={request_id} stage=validation error={type(e).__name__}: {str(e)}")
        return jsonify({"error": str(e), "request_id": request_id}), 400
    except (RateLimitError, APIConnectionError, APITimeoutError) as e:
        # Unhandled transient OpenAI errors (shouldn't happen if retry logic works, but catch anyway)
        error_code = getattr(e, 'status_code', None) or getattr(e, 'code', None) or 429
        logger.exception(f"GEN_FAIL request_id={request_id} stage=openai_transient type={type(e).__name__} code={error_code} - RETRYABLE_ERROR (no attempt consumed)")
        logger.info(f"RETURNING RETRYABLE_ERROR (no attempt consumed) request_id={request_id}")
        return jsonify({
            "error": "RETRYABLE_ERROR",
            "code": error_code,
            "message": "Temporary overload. Please try again.",
            "request_id": request_id
        }), 503
    except APIError as e:
        # Check if it's a 5xx server error (transient)
        status_code = getattr(e, 'status_code', None)
        if status_code and 500 <= status_code < 600:
            logger.exception(f"GEN_FAIL request_id={request_id} stage=openai_5xx code={status_code} - RETRYABLE_ERROR (no attempt consumed)")
            logger.info(f"RETURNING RETRYABLE_ERROR (no attempt consumed) request_id={request_id}")
            return jsonify({
                "error": "RETRYABLE_ERROR",
                "code": status_code,
                "message": "Temporary overload. Please try again.",
                "request_id": request_id
            }), 503
        else:
            # Non-transient API error (4xx except 429) - return 400
            error_msg = getattr(e, 'message', None) or str(e)
            logger.exception(f"GEN_FAIL request_id={request_id} stage=openai_4xx code={status_code} message={error_msg}")
            return jsonify({"error": f"OpenAI API error: {error_msg}", "request_id": request_id}), 400
    except Exception as e:
        # Catch-all for any other unexpected exceptions - prevent 502
        # Log full traceback with logger.exception
        logger.exception(f"GEN_FAIL request_id={request_id} stage=unexpected error={type(e).__name__}: {str(e)}")
        # Do not expose internal error details to client
        return jsonify({
            "error": "INTERNAL_ERROR",
            "message": "Server error",
            "request_id": request_id
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=False)

