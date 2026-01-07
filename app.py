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

# In-memory storage for tracking used Object A per product
# Key: (product_name, product_description) -> value: list of used A objects
used_A_tracker = {}

# In-memory storage for tracking used object pairs (A,B) per product
# Key: (product_name, product_description) -> value: list of canonical pair keys (order-insensitive)
used_pairs_tracker = {}

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
        print(f"AD_GOAL attempt={attempt}: {ad_goal}")
        return ad_goal
    except RetryableError:
        # Propagate transient errors - do NOT consume attempt
        raise
    except Exception as e:
        # Fallback for non-transient errors only
        ad_goal = f"Advertising goal for attempt {attempt}: highlight different aspect of {product_name}"
        print(f"AD_GOAL attempt={attempt}: {ad_goal} (fallback)")
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


def pick_two_objects(product_name, product_description, headline, ad_goal, used_A_list=None, used_pairs_list=None, attempt=None):
    """Select two physical objects (A and B), projections, overlap assessment, layout, and background using text model.
    
    Args:
        product_name: Product name
        product_description: Product description
        headline: Generated headline
        ad_goal: Distinct advertising goal for this attempt
        used_A_list: List of Object A values that have been used in previous attempts (forbidden)
        used_pairs_list: List of canonical pair keys (A|B) that have been used in previous attempts (forbidden)
        attempt: Attempt number (1, 2, or 3) - used to trigger hybrid search for attempt 2
    
    Returns:
        dict with keys: A, B, C_projection_description, D_projection_description, overlap_assessment, 
                        layout, background_classic_of_C, c_is_dominant
    """
    if not client:
        raise ValueError("OpenAI API key not configured")
    
    if used_A_list is None:
        used_A_list = []
    
    if used_pairs_list is None:
        used_pairs_list = []
    
    # Build forbidden A list text
    forbidden_A_text = ""
    if used_A_list:
        forbidden_A_text = f"\n\nCRITICAL: The following objects have already been used as Object A in previous attempts and MUST NOT be selected: {', '.join(used_A_list)}\nYou MUST select a DIFFERENT object for A that is NOT in this list."
    
    # Build forbidden pairs list text
    forbidden_pairs_text = ""
    if used_pairs_list:
        # Convert canonical keys back to readable format for the prompt
        readable_pairs = []
        for pair_key in used_pairs_list:
            parts = pair_key.split("|")
            if len(parts) == 2:
                readable_pairs.append(f"({parts[0]}, {parts[1]})")
        if readable_pairs:
            forbidden_pairs_text = f"\n\nCRITICAL: The following object pairs have already been used in previous attempts and MUST NOT be selected (in any order): {', '.join(readable_pairs)}\nYou MUST select a DIFFERENT pair (A,B) that is NOT in this list. The pair is forbidden regardless of which object is A and which is B."
    
    forbidden_text = forbidden_A_text + forbidden_pairs_text
    
    def make_selection_prompt():
        return f"""You are an ACE engine object selector. Select two physical objects for an advertisement with strict geometric overlap assessment.

Product Name: {product_name}
Product Description: {product_description}
Headline: {headline}
Advertising Goal: {ad_goal}

Rules:
1. Generate a list of 80 physical, real, associative objects based on the product, headline, and advertising goal.
2. Objects must be simple, everyday, familiar physical objects — NOT ideas, symbols, abstract concepts, or illustrations.
3. Objects should be non-functional / not functionally linked.
4. Do NOT pick objects containing text/logos/letters/numbers/external graphics (unless inherent like playing cards, dice dots, engraved compass letters).
{forbidden_text}

Selection:
- A = object with central meaning to the ad goal (MUST be different from previously used A objects)
- B = object used for conceptual emphasis (but pairing is still only by shape similarity)

Projection Selection:
- C_projection_description = how to view Object A to maximize its silhouette area (camera angle, perspective, orientation)
- D_projection_description = how to view Object B to maximize its silhouette area (camera angle, perspective, orientation)
- Both projections must be clean silhouettes, no confusing details
- Choose the projection with the largest visible dominant area for each object

Geometric Overlap Assessment (CRITICAL):
Compare the simplified shapes E (from C) and F (from D) after permitted adjustments (scale/angle/proportion without distortion).
- "NEAR_GEOMETRIC_OVERLAP": E and F can reach almost geometric overlap - the projection silhouettes are nearly identical after permitted adjustments. This is clear and immediate to an average human eye.
- "ONLY_SIMILAR": E and F are similar but NOT nearly geometric overlap - they share some shape similarity but cannot reach near-geometric overlap even with adjustments.
- "NO_SIMILARITY": E and F have no clear shape similarity - an average human eye would not see them as similar.

Layout Decision:
- layout = "HYBRID" ONLY if overlap_assessment == "NEAR_GEOMETRIC_OVERLAP"
- layout = "SIDE_BY_SIDE" if overlap_assessment is "ONLY_SIMILAR" or "NO_SIMILARITY"
- Never force HYBRID if overlap is not clear and immediate to an average human eye

Background:
- background_classic_of_C = classic natural background for the dominant object (A's projection C)
- c_is_dominant = true (C always controls lighting/texture/composition/background)

Hybrid Mode Assessment (CRITICAL for HYBRID):
- hybrid_mode = "PROJECTION_REPLACEMENT" ONLY if D can geometrically replace an equivalent structural element of C (e.g., one book in a shelf becomes a laptop, branches become USB cables).
- hybrid_mode = "SIDE_BY_SIDE" if D would be placed on top of, resting on, or beside C as a separate object.
- HYBRID layout is ONLY allowed if hybrid_mode == "PROJECTION_REPLACEMENT" AND overlap_assessment == "NEAR_GEOMETRIC_OVERLAP"

Return ONLY valid JSON with these exact keys:
{{
  "A": "object name",
  "B": "object name",
  "C_projection_description": "how to view A to maximize silhouette area (camera angle/perspective)",
  "D_projection_description": "how to view B to maximize silhouette area (camera angle/perspective)",
  "overlap_assessment": "NEAR_GEOMETRIC_OVERLAP" or "ONLY_SIMILAR" or "NO_SIMILARITY",
  "hybrid_mode": "PROJECTION_REPLACEMENT" (only if true geometric embedding is possible) or "SIDE_BY_SIDE",
  "layout": "HYBRID" (only if overlap_assessment is NEAR_GEOMETRIC_OVERLAP AND hybrid_mode is PROJECTION_REPLACEMENT) or "SIDE_BY_SIDE",
  "background_classic_of_C": "description of classic natural background",
  "c_is_dominant": true
}}

Do not include any explanation or other text."""
    
    # HYBRID_SEARCH: For attempt 2 (or any chosen attempt), try up to 8 different A/B candidates to find NEAR_GEOMETRIC_OVERLAP
    if attempt == 2:
        max_hybrid_search_tries = 8
        best_side_by_side_result = None  # Store best SIDE_BY_SIDE result as fallback
        
        print(f"HYBRID_SEARCH attempt=2 starting search for NEAR_GEOMETRIC_OVERLAP (max {max_hybrid_search_tries} tries)")
        
        for search_try in range(max_hybrid_search_tries):
            try:
                selection_prompt = make_selection_prompt()
                response = retry_openai_call(
                    lambda: client.chat.completions.create(
                        model=TEXT_MODEL,
                        messages=[
                            {"role": "system", "content": "You are an ACE engine object selector. Always return valid JSON only."},
                            {"role": "user", "content": selection_prompt}
                        ],
                        temperature=0.8,
                        response_format={"type": "json_object"}
                    ),
                    operation_name="hybrid_search_object_selection"
                )
                
                result = json.loads(response.choices[0].message.content.strip())
                
                # Validate fields exist
                A = result.get("A", "").strip()
                B = result.get("B", "").strip()
                C_projection_description = result.get("C_projection_description", "").strip()
                D_projection_description = result.get("D_projection_description", "").strip()
                overlap_assessment = result.get("overlap_assessment", "").strip()
                hybrid_mode = result.get("hybrid_mode", "SIDE_BY_SIDE").strip()
                layout = result.get("layout", "SIDE_BY_SIDE")
                background_classic_of_C = result.get("background_classic_of_C", "")
                c_is_dominant = result.get("c_is_dominant", True)
                
                # Check if A is in forbidden list
                if used_A_list and A.lower() in [used.lower() for used in used_A_list]:
                    print(f"HYBRID_SEARCH attempt=2 try={search_try + 1} REJECTED A (forbidden): {A}")
                    continue  # Skip this result and try again
                
                # Check if pair (A,B) is in forbidden pairs list
                pair_key = create_pair_key(A, B)
                if used_pairs_list and pair_key in used_pairs_list:
                    print(f"HYBRID_SEARCH attempt=2 try={search_try + 1} REJECTED PAIR: A={A}, B={B} (forbidden pair)")
                    continue  # Skip this result and try again
                
                # Validate overlap_assessment
                if overlap_assessment not in ["NEAR_GEOMETRIC_OVERLAP", "ONLY_SIMILAR", "NO_SIMILARITY"]:
                    overlap_assessment = "ONLY_SIMILAR"
                
                # Validate hybrid_mode
                if hybrid_mode not in ["PROJECTION_REPLACEMENT", "SIDE_BY_SIDE"]:
                    hybrid_mode = "SIDE_BY_SIDE"
                
                # Check if we found NEAR_GEOMETRIC_OVERLAP
                if overlap_assessment == "NEAR_GEOMETRIC_OVERLAP" and hybrid_mode == "PROJECTION_REPLACEMENT":
                    # Found a valid HYBRID candidate!
                    layout = "HYBRID"
                    print(f"HYBRID_FOUND attempt=2 try={search_try + 1} A={A} B={B} overlap={overlap_assessment} hybrid_mode={hybrid_mode}")
                    
                    # Validate and return
                    if not A or not B:
                        A = "product object"
                        B = "complementary object"
                    if not C_projection_description:
                        C_projection_description = "front view maximizing silhouette area"
                    if not D_projection_description:
                        D_projection_description = "front view maximizing silhouette area"
                    
                    return {
                        "A": A,
                        "B": B,
                        "C_projection_description": C_projection_description,
                        "D_projection_description": D_projection_description,
                        "overlap_assessment": overlap_assessment,
                        "hybrid_mode": hybrid_mode,
                        "layout": layout,
                        "background_classic_of_C": background_classic_of_C,
                        "c_is_dominant": c_is_dominant
                    }
                else:
                    # Not a HYBRID candidate, but might be a good SIDE_BY_SIDE fallback
                    if overlap_assessment == "ONLY_SIMILAR":
                        layout = "SIDE_BY_SIDE"
                        # Store as best SIDE_BY_SIDE result if we don't have one yet
                        if best_side_by_side_result is None:
                            best_side_by_side_result = {
                                "A": A,
                                "B": B,
                                "C_projection_description": C_projection_description,
                                "D_projection_description": D_projection_description,
                                "overlap_assessment": overlap_assessment,
                                "hybrid_mode": hybrid_mode,
                                "layout": layout,
                                "background_classic_of_C": background_classic_of_C,
                                "c_is_dominant": c_is_dominant
                            }
                    
                    print(f"HYBRID_SEARCH attempt=2 try={search_try + 1} overlap={overlap_assessment} hybrid_mode={hybrid_mode} layout={layout} (not HYBRID, continuing search)")
                    continue  # Continue searching
                    
            except Exception as e:
                print(f"HYBRID_SEARCH attempt=2 try={search_try + 1} error: {str(e)}, continuing search")
                continue  # Continue searching on error
        
        # If we get here, we didn't find a NEAR_GEOMETRIC_OVERLAP after max_hybrid_search_tries
        if best_side_by_side_result:
            print(f"HYBRID_NOT_FOUND attempt=2 fallback SIDE_BY_SIDE A={best_side_by_side_result['A']} B={best_side_by_side_result['B']}")
            return best_side_by_side_result
        else:
            print(f"HYBRID_NOT_FOUND attempt=2 no valid result found, using fallback")
            # Fallback to default
            return {
                "A": "product object",
                "B": "complementary object",
                "C_projection_description": "front view maximizing silhouette area",
                "D_projection_description": "front view maximizing silhouette area",
                "overlap_assessment": "ONLY_SIMILAR",
                "hybrid_mode": "SIDE_BY_SIDE",
                "layout": "SIDE_BY_SIDE",
                "background_classic_of_C": "natural background",
                "c_is_dominant": True
            }
    
    # For attempts 1 and 3, use the original retry logic
    selection_prompt = make_selection_prompt()
    
    # Retry up to 5 times if A is in forbidden list OR pair is forbidden OR if overlap_assessment is NO_SIMILARITY
    max_internal_retries = 5
    best_result = None  # Store best "ONLY_SIMILAR" result as fallback
    
    for retry_attempt in range(max_internal_retries):
        try:
            response = retry_openai_call(
                lambda: client.chat.completions.create(
                    model=TEXT_MODEL,
                    messages=[
                        {"role": "system", "content": "You are an ACE engine object selector. Always return valid JSON only."},
                        {"role": "user", "content": selection_prompt}
                    ],
                    temperature=0.8,
                    response_format={"type": "json_object"}
                ),
                operation_name="object_selection"
            )
            
            result = json.loads(response.choices[0].message.content.strip())
            
            # Validate fields exist
            A = result.get("A", "").strip()
            B = result.get("B", "").strip()
            C_projection_description = result.get("C_projection_description", "").strip()
            D_projection_description = result.get("D_projection_description", "").strip()
            overlap_assessment = result.get("overlap_assessment", "").strip()
            hybrid_mode = result.get("hybrid_mode", "SIDE_BY_SIDE").strip()
            layout = result.get("layout", "SIDE_BY_SIDE")
            background_classic_of_C = result.get("background_classic_of_C", "")
            c_is_dominant = result.get("c_is_dominant", True)
            
            # Check if A is in forbidden list
            if used_A_list and A.lower() in [used.lower() for used in used_A_list]:
                print(f"REJECTED A (forbidden): {A}")
                if retry_attempt < max_internal_retries - 1:
                    print(f"FORBIDDEN_HIT retrying selector... attempt={retry_attempt + 1} forbidden_A={A}")
                    continue  # Retry
                else:
                    print(f"FORBIDDEN_HIT max retries reached, using A anyway: {A}")
            
            # Check if pair (A,B) is in forbidden pairs list
            pair_key = create_pair_key(A, B)
            if used_pairs_list and pair_key in used_pairs_list:
                print(f"REJECTED PAIR: A={A}, B={B} (forbidden pair)")
                if retry_attempt < max_internal_retries - 1:
                    print(f"FORBIDDEN_PAIR_HIT retrying selector... attempt={retry_attempt + 1} forbidden_pair={pair_key}")
                    continue  # Retry
                else:
                    print(f"FORBIDDEN_PAIR_HIT max retries reached, using pair anyway: A={A}, B={B}")
            
            # Validate overlap_assessment
            if overlap_assessment not in ["NEAR_GEOMETRIC_OVERLAP", "ONLY_SIMILAR", "NO_SIMILARITY"]:
                overlap_assessment = "ONLY_SIMILAR"  # Default to conservative
            
            # Validate hybrid_mode
            if hybrid_mode not in ["PROJECTION_REPLACEMENT", "SIDE_BY_SIDE"]:
                hybrid_mode = "SIDE_BY_SIDE"  # Default to conservative
            
            # Enforce layout based on overlap_assessment AND hybrid_mode
            # HYBRID is ONLY allowed if BOTH conditions are met:
            # 1. overlap_assessment == "NEAR_GEOMETRIC_OVERLAP"
            # 2. hybrid_mode == "PROJECTION_REPLACEMENT"
            if overlap_assessment == "NEAR_GEOMETRIC_OVERLAP" and hybrid_mode == "PROJECTION_REPLACEMENT":
                layout = "HYBRID"
            elif overlap_assessment == "ONLY_SIMILAR":
                layout = "SIDE_BY_SIDE"
                # Store as best result for fallback
                if best_result is None:
                    best_result = {
                        "A": A,
                        "B": B,
                        "C_projection_description": C_projection_description,
                        "D_projection_description": D_projection_description,
                        "overlap_assessment": overlap_assessment,
                        "hybrid_mode": hybrid_mode,
                        "layout": layout,
                        "background_classic_of_C": background_classic_of_C,
                        "c_is_dominant": c_is_dominant
                    }
            elif overlap_assessment == "NO_SIMILARITY":
                # Re-pick a new pair (retry)
                if retry_attempt < max_internal_retries - 1:
                    print(f"NO_SIMILARITY retrying selector... attempt={retry_attempt + 1} A={A} B={B}")
                    continue  # Retry to get a better pair
                else:
                    # Max retries reached - use best "ONLY_SIMILAR" result or fallback to SIDE_BY_SIDE
                    if best_result:
                        print(f"NO_SIMILARITY max retries reached, using best ONLY_SIMILAR result")
                        return best_result
                    else:
                        # No good result found, force SIDE_BY_SIDE
                        layout = "SIDE_BY_SIDE"
                        overlap_assessment = "ONLY_SIMILAR"
            
            # Validate layout matches overlap_assessment AND hybrid_mode
            if layout == "HYBRID" and (overlap_assessment != "NEAR_GEOMETRIC_OVERLAP" or hybrid_mode != "PROJECTION_REPLACEMENT"):
                layout = "SIDE_BY_SIDE"
            
            # Fallback if missing critical fields
            if not A or not B:
                A = "product object"
                B = "complementary object"
            
            if not C_projection_description:
                C_projection_description = "front view maximizing silhouette area"
            if not D_projection_description:
                D_projection_description = "front view maximizing silhouette area"
            
            # Log successful selection (A passed forbidden check or was not forbidden)
            # Note: attempt number is not available here, will be logged in generate_image
            
            return {
                "A": A,
                "B": B,
                "C_projection_description": C_projection_description,
                "D_projection_description": D_projection_description,
                "overlap_assessment": overlap_assessment,
                "hybrid_mode": hybrid_mode,
                "layout": layout,
                "background_classic_of_C": background_classic_of_C,
                "c_is_dominant": c_is_dominant
            }
        except Exception as e:
            if retry_attempt < max_internal_retries - 1:
                print(f"Warning: Object selection failed (retry {retry_attempt + 1}): {str(e)}")
                continue
            else:
                # Fallback on error after all retries
                print(f"Warning: Object selection failed after {max_internal_retries} retries: {str(e)}, using fallback")
                if best_result:
                    return best_result
                return {
                    "A": "product object",
                    "B": "complementary object",
                    "C_projection_description": "front view maximizing silhouette area",
                    "D_projection_description": "front view maximizing silhouette area",
                    "overlap_assessment": "ONLY_SIMILAR",
                    "hybrid_mode": "SIDE_BY_SIDE",
                    "layout": "SIDE_BY_SIDE",
                    "background_classic_of_C": "natural background",
                    "c_is_dominant": True
                }
    
    # Should not reach here, but fallback
    if best_result:
        return best_result
    return {
        "A": "product object",
        "B": "complementary object",
        "C_projection_description": "front view maximizing silhouette area",
        "D_projection_description": "front view maximizing silhouette area",
        "overlap_assessment": "ONLY_SIMILAR",
        "hybrid_mode": "SIDE_BY_SIDE",
        "layout": "SIDE_BY_SIDE",
        "background_classic_of_C": "natural background",
        "c_is_dominant": True
    }


def generate_image(product_name, product_description, headline, ad_size, attempt, ad_goal, used_A_list=None, used_pairs_list=None):
    """Generate image using OpenAI DALL-E with strict two-object enforcement."""
    if not client:
        raise ValueError("OpenAI API key not configured")
    
    # Map ad_size to OpenAI format
    size_map = {
        "1024x1024": "1024x1024",
        "1024x1536": "1024x1536",
        "1536x1024": "1536x1024"
    }
    openai_size = size_map.get(ad_size, "1024x1024")
    
    # Log used_A list before selection
    if used_A_list:
        print(f"USED_A so far: {used_A_list}")
    else:
        print(f"USED_A so far: [] (first attempt)")
    
    # Log used_pairs list before selection
    if used_pairs_list:
        print(f"USED_PAIRS: {used_pairs_list}")
    else:
        print(f"USED_PAIRS: [] (first attempt)")
    
    # Step 1: Pick two objects, projections, overlap assessment, hybrid_mode, layout, and background (with ad_goal, used_A_list, used_pairs_list, and attempt)
    objects = pick_two_objects(product_name, product_description, headline, ad_goal, used_A_list, used_pairs_list, attempt)
    A = objects["A"]
    B = objects["B"]
    C_projection_description = objects["C_projection_description"]
    D_projection_description = objects["D_projection_description"]
    overlap_assessment = objects["overlap_assessment"]
    hybrid_mode = objects.get("hybrid_mode", "SIDE_BY_SIDE")
    layout = objects["layout"]
    background_classic_of_C = objects["background_classic_of_C"]
    c_is_dominant = objects.get("c_is_dominant", True)
    
    # Log successful selection of A for this attempt
    print(f"SELECTED A for attempt {attempt}: {A}")
    
    # Log successful selection of pair for this attempt
    pair_key = create_pair_key(A, B)
    print(f"SELECTED PAIR attempt={attempt}: A={A}, B={B} (pair_key: {pair_key})")
    
    # Debug log: print selected A, B, layout, overlap_assessment, hybrid_mode, and openai_size (NOT full prompt, NOT secrets)
    print(f"SELECTED A/B attempt={attempt}: A={A}, B={B}, layout={layout}, overlap_assessment={overlap_assessment}, hybrid_mode={hybrid_mode}, ad_size={ad_size} (OpenAI size: {openai_size})")
    
    # Step 2: Build strict image prompt with explicit A/B/projections/layout/background
    # Camera angle instructions based on projection descriptions
    camera_instruction = f"""CAMERA ANGLE INSTRUCTIONS (CRITICAL):
- Object A (C projection): {C_projection_description}
- Object B (D projection): {D_projection_description}
- The camera angle and perspective MUST match these projection descriptions exactly.
- Both objects must be viewed from the angles that maximize their silhouette areas."""
    
    if layout == "HYBRID":
        # HYBRID is ONLY allowed if BOTH overlap_assessment is NEAR_GEOMETRIC_OVERLAP AND hybrid_mode is PROJECTION_REPLACEMENT
        if overlap_assessment != "NEAR_GEOMETRIC_OVERLAP" or hybrid_mode != "PROJECTION_REPLACEMENT":
            # Force SIDE_BY_SIDE if conditions not met
            layout = "SIDE_BY_SIDE"
            print(f"HYBRID_REJECTED: overlap_assessment={overlap_assessment}, hybrid_mode={hybrid_mode}, forcing SIDE_BY_SIDE")
        
        layout_instruction = f"""Create a TRUE ACE HYBRID with PROJECTION REPLACEMENT (MANDATORY):
- The final image MUST show ONE single physical object in the scene.
- Object B's projection (D, simplified as F) must GEOMETRICALLY REPLACE an equivalent structural element of Object A's projection (C, simplified as E).
- D must REPLACE PART OF C within C's classic structure - D becomes an integral part of C's form.
- The silhouette F must be nearly geometrically overlapped with silhouette E after permitted adjustments (scale/angle/proportion without distortion).
- This overlap must be clear and immediate to an average human eye - the shapes must be nearly identical.
- Present the HYBRID at an angle that maximizes both projections' visibility while keeping full photographic realism.

CRITICAL SINGLE-OBJECT REQUIREMENTS (MANDATORY):
- The final silhouette MUST read as one contiguous object - no visible seams of two separate items.
- Object B must NOT appear as a separate object - it must be fully integrated into Object A's structure.
- The image must look like ONE unified physical object, not two objects placed together.
- No visible boundaries, edges, or seams that suggest two separate items.
- The composition must read as a single, cohesive physical entity.

CRITICAL ANTI-STACKING RULES (MANDATORY):
- D must REPLACE a structural element of C, NOT be placed on top of C.
- FORBIDDEN: "on top of", "resting on", "placed on", "sitting on", "lying on", "inside of", "next to", or any stacking/positioning arrangement.
- FORBIDDEN: D as a separate object positioned above, beside, or within C.
- REQUIRED: D must be geometrically embedded INTO C's structure, replacing an equivalent part.
- Examples of VALID hybrid: shelf of books where ONE BOOK IS A LAPTOP (laptop replaces book), tree where branches are USB cables (cables replace branches).
- Examples of INVALID hybrid: laptop lying on an open book (this is stacking, not replacement), laptop next to a book (this is side-by-side, not replacement).

NEGATIVE CONSTRAINT:
- If you cannot make it look like ONE single physical object with no visible seams, REJECT this layout and output SIDE_BY_SIDE instead.
- If the final image shows any residual side-by-side appearance or two distinct objects, it is INVALID and must be SIDE_BY_SIDE."""
    else:  # SIDE_BY_SIDE
        layout_instruction = f"""Place Object A (C projection) and Object B (D projection) SIDE BY SIDE at the same angle.
- Highlight maximal similar area between the projections.
- Place them close together, emphasizing their shape similarity.
- Both objects must be viewed from angles matching their projection descriptions."""
    
    image_prompt = f"""YOU ARE A PROFESSIONAL ADVERTISING PHOTOGRAPHER.
YOU MUST FOLLOW ALL RULES BELOW. NO EXCEPTIONS.

MANDATORY OBJECTS (BOTH MUST APPEAR):
- Object A: {A}
- Object B: {B}
- YOU MUST SHOW BOTH OBJECT A AND OBJECT B IN THE IMAGE.
- NEVER show only one object. NEVER show only Object A. NEVER show only Object B.
- BOTH objects must be clearly visible and recognizable.

{camera_instruction}

LAYOUT INSTRUCTION:
{layout_instruction}

BACKGROUND AND LIGHTING RULE (CRITICAL - ENGINE 05H):
- The background MUST be: {background_classic_of_C}
- This is the classic natural background of Object A (the dominant object C).
- C (Object A's projection) controls lighting, texture, composition, and background.
- D (Object B's projection) must NOT change C's background or lighting.
- Background and lighting must match ONLY C. D must not affect background/lighting.
- NEVER use studio backgrounds, black backgrounds, gradients, or abstract scenes.
- The entire scene's lighting, texture, and composition must be determined by C only.

STYLE RULES:
- Ultra-realistic photography ONLY.
- Looks like a real camera photograph.
- Natural lighting, correct perspective, real materials.
- NO illustration, NO 3D render, NO CGI, NO AI-art look.

HEADLINE RULES (INSIDE IMAGE):
- The headline "{headline}" MUST appear INSIDE the image.
- The headline must be clearly readable and visually integrated into the composition.
- Place it above/below/next-to the objects (never on them).

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

CRITICAL ENFORCEMENT:
- YOU MUST SHOW BOTH OBJECT A ({A}) AND OBJECT B ({B}).
- The image MUST contain BOTH objects. Single-object images are FORBIDDEN.
- BOTH objects must be clearly visible and recognizable in the final image.

PRODUCT NAME: {product_name}
PRODUCT DESCRIPTION: {product_description}

TASK:
Generate ONE final advertising image that fully follows ALL the rules above.
The image MUST show BOTH Object A and Object B. Do not explain. Do not describe."""

    # Debug log: print image prompt, ad_size, and attempt
    print("=== IMAGE PROMPT SENT TO OPENAI ===")
    print(image_prompt)
    print("=== END IMAGE PROMPT ===")

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
        print(f"OpenAI image response keys: {response_keys}")
        
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


def create_zip_in_memory(image_base64, marketing_text, attempt):
    """Create a ZIP file in memory containing the image and text file."""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Decode image from base64 and add to ZIP
        image_data = base64.b64decode(image_base64.split(',')[1] if ',' in image_base64 else image_base64)
        zip_file.writestr(f"ad_{attempt}.jpg", image_data)
        
        # Add marketing text file
        zip_file.writestr(f"ad_{attempt}.txt", marketing_text)
    
    zip_buffer.seek(0)
    zip_base64 = base64.b64encode(zip_buffer.getvalue()).decode('utf-8')
    
    return zip_base64


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
    """
    # Generate request ID for this request
    request_id = generate_request_id()
    
    # Log route hit
    route_path = request.path
    logger.info(f"ROUTE_HIT {route_path} request_id={request_id}")
    
    # Top-level try/except to catch ALL exceptions and prevent 502s
    try:
        # Get JSON body
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "Request body must be JSON", "request_id": request_id}), 400
        
        # Validate required fields
        product_name = data.get("product_name")
        product_description = data.get("product_description")
        ad_size = data.get("ad_size")
        attempt = data.get("attempt")
        
        # Check all fields are present
        if not product_name:
            return jsonify({"error": "product_name is required and must be a non-empty string", "request_id": request_id}), 400
        
        if not product_description:
            return jsonify({"error": "product_description is required and must be a non-empty string", "request_id": request_id}), 400
        
        if not ad_size:
            return jsonify({"error": "ad_size is required and must be a non-empty string", "request_id": request_id}), 400
        
        if attempt is None:
            return jsonify({"error": "attempt is required and must be 1, 2, or 3", "request_id": request_id}), 400
        
        # Check all fields are strings (except attempt)
        if not isinstance(product_name, str) or not product_name.strip():
            return jsonify({"error": "product_name must be a non-empty string", "request_id": request_id}), 400
        
        if not isinstance(product_description, str) or not product_description.strip():
            return jsonify({"error": "product_description must be a non-empty string", "request_id": request_id}), 400
        
        if not isinstance(ad_size, str) or not ad_size.strip():
            return jsonify({"error": "ad_size must be a non-empty string", "request_id": request_id}), 400
        
        # Validate attempt
        if not isinstance(attempt, int) or attempt not in [1, 2, 3]:
            return jsonify({"error": "attempt must be exactly 1, 2, or 3", "request_id": request_id}), 400
        
        # Validate ad_size is one of the allowed values
        if ad_size not in VALID_AD_SIZES:
            return jsonify({
                "error": f"ad_size must be exactly one of: {', '.join(sorted(VALID_AD_SIZES))}",
                "request_id": request_id
            }), 400
        
        # Check OpenAI API key - fail fast with clear error
        if not OPENAI_API_KEY or not OPENAI_API_KEY.strip() or not client:
            return jsonify({"error": "Server misconfigured: OPENAI_API_KEY is not set", "request_id": request_id}), 500
        
        # Create product key for tracking used_A
        product_key = (product_name.strip().lower(), product_description.strip().lower())
        
        # Get or initialize used_A list for this product
        if product_key not in used_A_tracker:
            used_A_tracker[product_key] = []
        used_A_list = used_A_tracker[product_key]
        
        # Get or initialize used_pairs list for this product
        if product_key not in used_pairs_tracker:
            used_pairs_tracker[product_key] = []
        used_pairs_list = used_pairs_tracker[product_key]
        
        # Generate distinct ad_goal for this attempt
        ad_goal = generate_ad_goal(product_name, product_description, attempt)
        
        # Generate headline and marketing text (with ad_goal)
        headline, marketing_text = generate_headline_and_text(product_name, product_description, attempt, ad_goal)
        
        # Generate image (with ad_goal, used_A_list, and used_pairs_list) - returns (image_data_url, selected_A, selected_B)
        image_data_url, selected_A, selected_B = generate_image(product_name, product_description, headline, ad_size, attempt, ad_goal, used_A_list, used_pairs_list)
        
        # Track the selected A (add to used_A_list if not already present)
        # NOTE: This only happens on successful generation - transient errors do NOT update used_A
        if selected_A:
            # Check if already tracked (case-insensitive)
            already_tracked = selected_A.lower() in [used.lower() for used in used_A_list]
            if not already_tracked:
                used_A_tracker[product_key].append(selected_A)
                print(f"TRACKED used_A for product: {selected_A} (total: {len(used_A_tracker[product_key])})")
            else:
                print(f"WARNING: Selected A '{selected_A}' was already in used_A_list, but was selected anyway")
        
        # Track the selected pair (A,B) (add to used_pairs_list if not already present)
        # NOTE: This only happens on successful generation - transient errors do NOT update used_pairs
        if selected_A and selected_B:
            pair_key = create_pair_key(selected_A, selected_B)
            if pair_key not in used_pairs_list:
                used_pairs_tracker[product_key].append(pair_key)
                print(f"TRACKED used_pair for product: {pair_key} (A={selected_A}, B={selected_B}) (total: {len(used_pairs_tracker[product_key])})")
            else:
                print(f"WARNING: Selected pair '{pair_key}' (A={selected_A}, B={selected_B}) was already in used_pairs_list, but was selected anyway")
        
        # Extract base64 from data URL for ZIP creation
        image_base64 = image_data_url.split(',')[1] if ',' in image_data_url else image_data_url
        
        # Create ZIP in memory
        zip_base64 = create_zip_in_memory(image_base64, marketing_text, attempt)
        
        # Return single ad object
        ad = {
            "ad_id": attempt,
            "headline": headline,
            "marketing_text_50_words": marketing_text,
            "image_data_url": image_data_url,
            "zip_base64": zip_base64,
            "zip_filename": f"ad_{attempt}.zip",
            "request_id": request_id
        }
        
        logger.info(f"Successfully generated ad (request_id={request_id}, attempt={attempt})")
        return jsonify(ad), 200
    
    except RetryableError as e:
        # Transient OpenAI error after max retries - return 503
        # This does NOT consume an attempt - used_A is only updated after successful generation
        logger.warning(f"RETURNING RETRYABLE_ERROR (no attempt consumed) request_id={request_id} code={e.code}")
        return jsonify({
            "error": "RETRYABLE_ERROR",
            "code": e.code,
            "message": "Temporary overload. Please try again.",
            "request_id": request_id
        }), 503
    except ValueError as e:
        # Non-transient validation errors - return 400
        logger.warning(f"Validation error (request_id={request_id}): {str(e)}")
        return jsonify({"error": str(e), "request_id": request_id}), 400
    except (RateLimitError, APIConnectionError, APITimeoutError) as e:
        # Unhandled transient OpenAI errors (shouldn't happen if retry logic works, but catch anyway)
        error_code = getattr(e, 'status_code', None) or getattr(e, 'code', None) or 429
        logger.warning(f"RETURNING RETRYABLE_ERROR (no attempt consumed) - unhandled transient error request_id={request_id} type={type(e).__name__} code={error_code}")
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
            logger.warning(f"RETURNING RETRYABLE_ERROR (no attempt consumed) - OpenAI 5xx error request_id={request_id} code={status_code}")
            return jsonify({
                "error": "RETRYABLE_ERROR",
                "code": status_code,
                "message": "Temporary overload. Please try again.",
                "request_id": request_id
            }), 503
        else:
            # Non-transient API error (4xx except 429) - return 400
            error_msg = getattr(e, 'message', None) or str(e)
            logger.warning(f"RETURNING 400 - non-transient OpenAI error request_id={request_id} code={status_code} message={error_msg}")
            return jsonify({"error": f"OpenAI API error: {error_msg}", "request_id": request_id}), 400
    except Exception as e:
        # Catch-all for any other unexpected exceptions - prevent 502
        # Log full traceback
        logger.exception(f"Unexpected error in /generate (request_id={request_id}): {type(e).__name__}: {str(e)}")
        # Do not expose internal error details to client
        return jsonify({
            "error": "INTERNAL_ERROR",
            "message": "Server error",
            "request_id": request_id
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=False)

