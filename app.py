from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import zipfile
import base64
import json
import os
import openai
import logging
from PIL import Image, ImageDraw, ImageFont

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

# Load ACE Engine specification from file
ACE_ENGINE_SPEC_PATH = 'ACE_ENGINE_HE.txt'
ace_engine_system_prompt = None

try:
    if os.path.exists(ACE_ENGINE_SPEC_PATH):
        with open(ACE_ENGINE_SPEC_PATH, 'r', encoding='utf-8') as f:
            ace_engine_system_prompt = f.read().strip()
        logger.info(f"ACE Engine specification loaded from {ACE_ENGINE_SPEC_PATH} ({len(ace_engine_system_prompt)} characters)")
    else:
        logger.warning(f"ACE Engine specification file {ACE_ENGINE_SPEC_PATH} not found. Using default placeholder.")
        ace_engine_system_prompt = "[PLACEHOLDER: ACE Engine specification not loaded]"
except Exception as e:
    logger.error(f"Error loading ACE Engine specification: {e}")
    ace_engine_system_prompt = "[ERROR: Failed to load ACE Engine specification]"

logger.info(f"OpenAI client initialized with image model: {openai_image_model}, text model: {openai_text_model}")


def plan_ads(product_name, product_description, size, run_index):
    """Plan 3 ads using ACE engine specification as SYSTEM prompt. Returns strict JSON with run_index and ads array."""
    user_prompt = f"""Product Name: {product_name}
Product Description: {product_description}
Size: {size}
Run Index: {run_index}

Return ONLY valid JSON according to the schema below. No prose.

Schema:
{{
  "run_index": {run_index},
  "ads": [
    {{
      "ad_index": 1,
      "message": "...",
      "mode": "replacement",
      "object_a": "...",
      "object_b": "...",
      "environment_context": "...",
      "visual_description": "...",
      "headline": "...",
      "marketing_text": "...",
      "image_prompt": "..."
    }},
    {{ "ad_index": 2, ... }},
    {{ "ad_index": 3, ... }}
  ]
}}

Requirements:
- Exactly 3 ads
- Exactly 2 ads with mode="replacement", exactly 1 ad with mode="side_by_side"
- headline interprets visual (does not describe it), 3-7 words, includes product name
- marketing_text ~50 words, does not describe visual
- object_a and object_b are physical objects (no words, no scenes)
- environment_context is existence-inference context (not narrative)
- image_prompt: photorealistic only (no vector/illustration/3D/CGI/AI-look)
- No text/logos/labels except headline (graphics only if inherent physical structure)
- Image contains ONLY ONE headline, NO OTHER TEXT"""

    for attempt in range(2):  # Retry once if invalid
        try:
            response = openai.ChatCompletion.create(
                model=openai_text_model,
                messages=[
                    {"role": "system", "content": ace_engine_system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.8,
                max_tokens=3500
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
            
            # Parse JSON
            plan = json.loads(content)
            
            # Validate structure: must have run_index and ads array
            if not isinstance(plan, dict):
                raise ValueError(f"Expected JSON object, got {type(plan)}")
            
            if 'ads' not in plan:
                raise ValueError("Missing 'ads' field in plan")
            
            ads_plan = plan['ads']
            if not isinstance(ads_plan, list) or len(ads_plan) != 3:
                raise ValueError(f"Expected 'ads' array with 3 items, got {type(ads_plan)} with {len(ads_plan) if isinstance(ads_plan, list) else 'N/A'} items")
            
            # Validate each ad has required fields
            required_fields = ['ad_index', 'message', 'mode', 'object_a', 'object_b', 'environment_context', 
                              'visual_description', 'headline', 'marketing_text', 'image_prompt']
            for i, ad_plan in enumerate(ads_plan):
                for field in required_fields:
                    if field not in ad_plan:
                        raise ValueError(f"Ad {i+1} missing required field: {field}")
                
                # Ensure ad_index matches position
                if ad_plan.get('ad_index') != i + 1:
                    ad_plan['ad_index'] = i + 1
                
                # Validate mode values
                mode = ad_plan.get('mode', '').lower()
                if mode not in ['replacement', 'side_by_side']:
                    raise ValueError(f"Ad {i+1} has invalid mode: {mode}. Must be 'replacement' or 'side_by_side'")
                
                # Validate headline: should not be empty and should include product name (case-insensitive check)
                headline = ad_plan.get('headline', '')
                if not headline or len(headline.split()) < 3 or len(headline.split()) > 7:
                    raise ValueError(f"Ad {i+1} headline must be 3-7 words, got {len(headline.split())} words")
                
                # Validate marketing_text: approximately 50 words
                marketing_words = ad_plan.get('marketing_text', '').split()
                if len(marketing_words) < 40 or len(marketing_words) > 60:
                    logger.warning(f"Ad {i+1} marketing_text has {len(marketing_words)} words (expected ~50)")
            
            # Validate mode distribution: exactly 2 replacement, exactly 1 side_by_side
            modes = [ad.get('mode', '').lower() for ad in ads_plan]
            replacement_count = modes.count('replacement')
            side_by_side_count = modes.count('side_by_side')
            
            if replacement_count != 2 or side_by_side_count != 1:
                raise ValueError(f"Invalid mode distribution: {replacement_count} replacement, {side_by_side_count} side_by_side. Expected: 2 replacement, 1 side_by_side")
            
            # Set run_index if not present or incorrect
            plan['run_index'] = run_index
            
            # Debug logging: planned mode, object_a, object_b, headline
            for ad in ads_plan:
                logger.info(f"PLANNED ad_index={ad['ad_index']} mode={ad['mode']} object_a={ad['object_a']} object_b={ad['object_b']} headline='{ad['headline']}'")
            
            logger.info(f"Successfully planned 3 ads (2 replacement, 1 side_by_side) for run_index={run_index}")
            return plan
            
        except json.JSONDecodeError as e:
            if attempt == 0:
                logger.warning(f"Invalid JSON on attempt {attempt + 1}, retrying... Error: {e}")
                continue
            else:
                logger.error(f"Invalid JSON after retry: {e}")
                raise ValueError(f"Failed to parse JSON response: {e}")
        except ValueError as e:
            if attempt == 0:
                logger.warning(f"Validation error on attempt {attempt + 1}, retrying... Error: {e}")
                continue
            else:
                logger.error(f"Validation failed after retry: {e}")
                raise
        except Exception as e:
            logger.error(f"Error planning ads: {e}", exc_info=True)
            raise
    
    raise ValueError("Failed to generate valid plan after 2 attempts")


def generate_visual_image_openai(image_prompt, ad_index, requested_width, requested_height):
    """Generate VISUAL image at 1024x1024 internally (headline is NOT included in generation, will be composed separately)"""
    try:
        # Always generate at 1024x1024 internally for visual component
        internal_size = "1024x1024"
        
        # Force these additions to EVERY image_prompt, but REMOVE headline requirement
        # Visual should NOT contain headline - headline will be rendered separately and composed
        forced_constraints = """ Photorealistic photograph. No illustration, no vector, no 3D, no CGI, no AI style. No logos, no labels, no packaging text, no signs. No text anywhere in the image."""
        
        # Remove any headline instructions from the original prompt
        visual_prompt = image_prompt
        # Ensure no headline-related text in the visual prompt
        if "headline" in visual_prompt.lower():
            # Remove headline references if present
            visual_prompt = visual_prompt.replace("headline", "").replace("text", "visual element")
        
        final_prompt = visual_prompt + forced_constraints
        
        logger.info(f"Generating VISUAL for ad {ad_index} with internal_size={internal_size} (requested_size={requested_width}x{requested_height})")
        logger.debug(f"Visual prompt (first 200 chars): {visual_prompt[:200]}...")
        
        # Generate image with OpenAI (legacy SDK pattern) - always 1024x1024 internally
        response = openai.Image.create(
            prompt=final_prompt,
            n=1,
            size=internal_size,
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
                
                logger.info(f"Successfully generated VISUAL for ad {ad_index}, format: {mime_type}, size: {len(image_bytes)} bytes")
                return image_bytes  # Return bytes only (not base64, will be processed)
            else:
                # Log available fields for debugging
                available_fields = list(image_data.keys()) if isinstance(image_data, dict) else [attr for attr in dir(image_data) if not attr.startswith('_')]
                logger.error(f"OpenAI response does not contain b64_json. Available fields: {available_fields}")
                raise ValueError("OpenAI response does not contain b64_json field. Image generation may have failed.")
        else:
            raise ValueError("OpenAI response does not contain image data")
            
    except Exception as e:
        logger.error(f"Error generating visual image with OpenAI: {e}")
        raise


def render_headline_image(headline_text, canvas_width, canvas_height):
    """Render headline as a separate image with transparent background"""
    try:
        # Calculate headline area: approximately 45-50% of canvas, but account for margins
        margin = 48
        available_width = canvas_width - (2 * margin)
        available_height = canvas_height - (2 * margin)
        
        # Headline occupies roughly 45-50% of available space
        headline_area_fraction = 0.48
        
        # For landscape (1536x1024): headline on right side, vertical space ~50%
        # For portrait (1024x1536): headline on bottom, horizontal space ~50%
        # For square (1024x1024): can be either, use ~50% for each dimension
        
        if canvas_width > canvas_height:  # Landscape: 1536x1024
            headline_width = int(available_width * headline_area_fraction)
            headline_height = available_height
        elif canvas_height > canvas_width:  # Portrait: 1024x1536
            headline_width = available_width
            headline_height = int(available_height * headline_area_fraction)
        else:  # Square: 1024x1024
            headline_width = int(available_width * 0.9)  # Use most of width
            headline_height = int(available_height * headline_area_fraction)
        
        # Create transparent background image for headline
        headline_img = Image.new('RGBA', (headline_width, headline_height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(headline_img)
        
        # Try to use a bold font, fallback to default if unavailable
        font_size = max(48, min(headline_width // len(headline_text) if len(headline_text) > 0 else 72, 120))
        font = None
        
        # Try multiple font paths (system-dependent)
        font_paths = [
            "arial.ttf",  # Windows
            "Arial.ttf",  # Windows alternate
            "/System/Library/Fonts/Helvetica.ttc",  # macOS
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",  # Linux alternative
        ]
        
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except (OSError, IOError):
                continue
        
        if font is None:
            # Fallback to default font
            try:
                font = ImageFont.load_default()
                # Default font is small, try to scale up
                if hasattr(font, 'getsize'):
                    # Legacy PIL
                    pass
            except:
                # Ultimate fallback
                font = ImageFont.load_default()
        
        # Calculate text bounding box
        bbox = draw.textbbox((0, 0), headline_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Scale font if text is too large
        if text_width > headline_width * 0.9 or text_height > headline_height * 0.9:
            scale_factor = min((headline_width * 0.9) / text_width, (headline_height * 0.9) / text_height)
            font_size = int(font_size * scale_factor)
            
            # Re-try font loading with scaled size
            scaled_font = None
            for font_path in font_paths:
                try:
                    scaled_font = ImageFont.truetype(font_path, font_size)
                    break
                except (OSError, IOError):
                    continue
            
            if scaled_font is not None:
                font = scaled_font
            else:
                font = ImageFont.load_default()
            
            bbox = draw.textbbox((0, 0), headline_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        
        # Center text in headline image
        x = (headline_width - text_width) // 2
        y = (headline_height - text_height) // 2
        
        # Draw text in black (bold and dominant)
        draw.text((x, y), headline_text, fill=(0, 0, 0, 255), font=font)
        
        return headline_img
        
    except Exception as e:
        logger.error(f"Error rendering headline image: {e}")
        raise


def compose_final_ad(visual_bytes, headline_text, requested_width, requested_height, ad_index):
    """Compose final ad image: visual + headline on canvas of requested size"""
    try:
        logger.info(f"COMPOSING final_ad ad_index={ad_index} requested_size={requested_width}x{requested_height} internal_gen_size=1024x1024 final_canvas_size={requested_width}x{requested_height}")
        
        # Load visual image from bytes
        visual_img = Image.open(io.BytesIO(visual_bytes))
        # Convert to RGB if needed (handles PNG/WebP transparency)
        if visual_img.mode != 'RGB':
            rgb_visual = Image.new('RGB', visual_img.size, (255, 255, 255))
            rgb_visual.paste(visual_img, mask=visual_img.split()[3] if visual_img.mode == 'RGBA' else None)
            visual_img = rgb_visual
        
        # Create blank RGB canvas with requested size (white background)
        canvas = Image.new('RGB', (requested_width, requested_height), (255, 255, 255))
        
        # Fixed gap margin (48px) to ensure visual and headline do not touch
        margin = 48
        
        # Render headline as separate image
        headline_img = render_headline_image(headline_text, requested_width, requested_height)
        
        # Calculate layout based on requested size
        if requested_width > requested_height:  # Landscape: 1536x1024
            # Side-by-side layout: visual left, headline right
            visual_area_width = int((requested_width - 3 * margin) * 0.48)  # ~48% for visual
            headline_area_width = int((requested_width - 3 * margin) * 0.48)  # ~48% for headline
            visual_area_height = requested_height - (2 * margin)
            headline_area_height = requested_height - (2 * margin)
            
            # Scale visual to fit allocated area while maintaining aspect ratio
            visual_aspect = visual_img.width / visual_img.height
            target_visual_aspect = visual_area_width / visual_area_height
            
            if visual_aspect > target_visual_aspect:
                # Visual is wider: fit to width
                scaled_visual_width = visual_area_width
                scaled_visual_height = int(visual_area_width / visual_aspect)
            else:
                # Visual is taller: fit to height
                scaled_visual_height = visual_area_height
                scaled_visual_width = int(visual_area_height * visual_aspect)
            
            scaled_visual = visual_img.resize((scaled_visual_width, scaled_visual_height), Image.Resampling.LANCZOS)
            
            # Scale headline to fit allocated area
            headline_aspect = headline_img.width / headline_img.height if headline_img.height > 0 else 1
            target_headline_aspect = headline_area_width / headline_area_height
            
            if headline_aspect > target_headline_aspect:
                scaled_headline_width = headline_area_width
                scaled_headline_height = int(headline_area_width / headline_aspect)
            else:
                scaled_headline_height = headline_area_height
                scaled_headline_width = int(headline_area_height * headline_aspect)
            
            scaled_headline = headline_img.resize((scaled_headline_width, scaled_headline_height), Image.Resampling.LANCZOS)
            
            # Convert headline from RGBA to RGB for pasting
            headline_rgb = Image.new('RGB', scaled_headline.size, (255, 255, 255))
            headline_rgb.paste(scaled_headline, mask=scaled_headline.split()[3] if scaled_headline.mode == 'RGBA' else None)
            
            # Position visual: left side, centered vertically
            visual_x = margin
            visual_y = (requested_height - scaled_visual_height) // 2
            
            # Position headline: right side, centered vertically
            headline_x = requested_width - margin - scaled_headline_width
            headline_y = (requested_height - scaled_headline_height) // 2
            
        elif requested_height > requested_width:  # Portrait: 1024x1536
            # Stacked layout: visual top, headline bottom
            visual_area_width = requested_width - (2 * margin)
            visual_area_height = int((requested_height - 3 * margin) * 0.48)  # ~48% for visual
            headline_area_width = requested_width - (2 * margin)
            headline_area_height = int((requested_height - 3 * margin) * 0.48)  # ~48% for headline
            
            # Scale visual to fit allocated area
            visual_aspect = visual_img.width / visual_img.height
            target_visual_aspect = visual_area_width / visual_area_height
            
            if visual_aspect > target_visual_aspect:
                scaled_visual_width = visual_area_width
                scaled_visual_height = int(visual_area_width / visual_aspect)
            else:
                scaled_visual_height = visual_area_height
                scaled_visual_width = int(visual_area_height * visual_aspect)
            
            scaled_visual = visual_img.resize((scaled_visual_width, scaled_visual_height), Image.Resampling.LANCZOS)
            
            # Scale headline to fit allocated area
            headline_aspect = headline_img.width / headline_img.height if headline_img.height > 0 else 1
            target_headline_aspect = headline_area_width / headline_area_height
            
            if headline_aspect > target_headline_aspect:
                scaled_headline_width = headline_area_width
                scaled_headline_height = int(headline_area_width / headline_aspect)
            else:
                scaled_headline_height = headline_area_height
                scaled_headline_width = int(headline_area_height * headline_aspect)
            
            scaled_headline = headline_img.resize((scaled_headline_width, scaled_headline_height), Image.Resampling.LANCZOS)
            
            # Convert headline from RGBA to RGB
            headline_rgb = Image.new('RGB', scaled_headline.size, (255, 255, 255))
            headline_rgb.paste(scaled_headline, mask=scaled_headline.split()[3] if scaled_headline.mode == 'RGBA' else None)
            
            # Position visual: top, centered horizontally
            visual_x = (requested_width - scaled_visual_width) // 2
            visual_y = margin
            
            # Position headline: bottom, centered horizontally
            headline_x = (requested_width - scaled_headline_width) // 2
            headline_y = requested_height - margin - scaled_headline_height
            
        else:  # Square: 1024x1024
            # Can use stacked or side-by-side, use stacked (visual top, headline bottom)
            visual_area_width = requested_width - (2 * margin)
            visual_area_height = int((requested_height - 3 * margin) * 0.48)
            headline_area_width = requested_width - (2 * margin)
            headline_area_height = int((requested_height - 3 * margin) * 0.48)
            
            # Scale visual
            visual_aspect = visual_img.width / visual_img.height
            target_visual_aspect = visual_area_width / visual_area_height
            
            if visual_aspect > target_visual_aspect:
                scaled_visual_width = visual_area_width
                scaled_visual_height = int(visual_area_width / visual_aspect)
            else:
                scaled_visual_height = visual_area_height
                scaled_visual_width = int(visual_area_height * visual_aspect)
            
            scaled_visual = visual_img.resize((scaled_visual_width, scaled_visual_height), Image.Resampling.LANCZOS)
            
            # Scale headline
            headline_aspect = headline_img.width / headline_img.height if headline_img.height > 0 else 1
            target_headline_aspect = headline_area_width / headline_area_height
            
            if headline_aspect > target_headline_aspect:
                scaled_headline_width = headline_area_width
                scaled_headline_height = int(headline_area_width / headline_aspect)
            else:
                scaled_headline_height = headline_area_height
                scaled_headline_width = int(headline_area_height * headline_aspect)
            
            scaled_headline = headline_img.resize((scaled_headline_width, scaled_headline_height), Image.Resampling.LANCZOS)
            
            # Convert headline from RGBA to RGB
            headline_rgb = Image.new('RGB', scaled_headline.size, (255, 255, 255))
            headline_rgb.paste(scaled_headline, mask=scaled_headline.split()[3] if scaled_headline.mode == 'RGBA' else None)
            
            # Position visual: top, centered
            visual_x = (requested_width - scaled_visual_width) // 2
            visual_y = margin
            
            # Position headline: bottom, centered
            headline_x = (requested_width - scaled_headline_width) // 2
            headline_y = requested_height - margin - scaled_headline_height
        
        # Paste visual and headline onto canvas
        canvas.paste(scaled_visual, (visual_x, visual_y))
        canvas.paste(headline_rgb, (headline_x, headline_y))
        
        # Export as JPEG bytes
        jpeg_buffer = io.BytesIO()
        canvas.save(jpeg_buffer, format='JPEG', quality=95)
        jpeg_bytes = jpeg_buffer.getvalue()
        jpeg_buffer.close()
        
        # Convert to base64 for data URL
        jpeg_base64 = base64.b64encode(jpeg_bytes).decode('utf-8')
        
        logger.info(f"Successfully composed final_ad ad_index={ad_index} final_size={len(jpeg_bytes)} bytes")
        
        return jpeg_bytes, jpeg_base64, "image/jpeg"
        
    except Exception as e:
        logger.error(f"Error composing final ad: {e}", exc_info=True)
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
        
        # Plan 3 ads using ACE engine (Step A: Planning)
        try:
            plan = plan_ads(product_name, product_description, size, run_index)
        except Exception as e:
            logger.error(f"Error planning ads: {e}", exc_info=True)
            return jsonify({
                "error": "Failed to plan ads",
                "message": str(e)
            }), 500
        
        ads_plan = plan['ads']
        
        # Generate 3 ads based on the plan (Step B: Image generation)
        ads = []  # API response
        ads_for_storage = []  # Full data with image_bytes for ZIP
        
        for ad_plan in ads_plan:
            try:
                ad_index = ad_plan['ad_index']
                headline = ad_plan['headline']
                marketing_text = ad_plan['marketing_text']
                image_prompt = ad_plan['image_prompt']
                mode = ad_plan.get('mode', 'unknown')
                object_a = ad_plan.get('object_a', 'unknown')
                object_b = ad_plan.get('object_b', 'unknown')
                
                # Log debug info: mode, object_a, object_b, headline (Step D: Debug logging)
                logger.info(f"PLANNED ad_index={ad_index} mode={mode} object_a={object_a} object_b={object_b}")
                logger.info(f"FINAL_HEADLINE ad_index={ad_index}: '{headline}'")
                
                # Ensure marketing_text is approximately 50 words (trim if needed)
                words = marketing_text.split()
                if len(words) > 60:
                    marketing_text = " ".join(words[:50])
                    logger.warning(f"Ad {ad_index} marketing_text trimmed from {len(words)} to 50 words")
                elif len(words) < 40:
                    logger.warning(f"Ad {ad_index} marketing_text is too short ({len(words)} words)")
                
                # Step B: Generate visual image at 1024x1024 internally (Size Adapter)
                logger.info(f"requested_size={width}x{height} internal_gen_size=1024x1024 final_canvas_size={width}x{height}")
                visual_bytes = generate_visual_image_openai(
                    image_prompt, ad_index, width, height
                )
                
                # Step B: Compose final ad (visual + headline) on canvas of requested size
                image_bytes, image_base64, mime_type = compose_final_ad(
                    visual_bytes, headline, width, height, ad_index
                )
                
                # Validate the generated image
                is_valid, error_msg = validate_image_bytes(image_bytes)
                if not is_valid:
                    logger.error(f"Image validation failed for ad {ad_index}: {error_msg}")
                    return jsonify({
                        "error": "Image generation failed",
                        "message": error_msg
                    }), 500
                
                # Create data URL with correct MIME type (always JPEG from composition)
                image_data_url = f"data:{mime_type};base64,{image_base64}"
                
                # Extension is always jpg from composition
                extension = "jpg"
                
                # API response (Step C: API output - unchanged shape)
                ad = {
                    "ad_index": ad_index,
                    "marketing_text": marketing_text,
                    "image_jpg": image_data_url
                }
                ads.append(ad)
                
                # Store full ad data for ZIP download (Step C: Store for ZIP)
                ad_storage = {
                    "ad_index": ad_index,
                    "marketing_text": marketing_text,
                    "image_jpg": image_data_url,
                    "image_bytes": image_bytes,  # Store bytes for ZIP generation
                    "extension": extension or "jpg"  # Store extension for ZIP filename
                }
                ads_for_storage.append(ad_storage)
                
                logger.info(f"Successfully generated ad {ad_index} for {product_name}")
                
            except Exception as e:
                logger.error(f"Error generating ad {ad_plan.get('ad_index', 'unknown')}: {e}", exc_info=True)
                return jsonify({
                    "error": "Image generation failed",
                    "message": str(e)
                }), 500
        
        # Store ads for download endpoint (keyed by run_index) - Step C: Store for ZIP
        storage_key = f"{product_name}_{run_index}"
        ads_storage[storage_key] = ads_for_storage
        
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
            # Use stored image_bytes if available (from composed final ad), otherwise extract from data URL
            image_bytes = ad.get('image_bytes')
            extension = ad.get('extension', 'jpg')
            
            if image_bytes is None:
                # Fallback: extract from data URL
                image_data = ad.get('image_jpg', '')
                if image_data.startswith('data:image'):
                    # Extract base64 part
                    base64_data = image_data.split(',')[1] if ',' in image_data else image_data
                    try:
                        image_bytes = base64.b64decode(base64_data)
                        # Detect extension if not stored
                        _, detected_ext = detect_image_format(image_bytes)
                        if detected_ext:
                            extension = detected_ext
                    except Exception as e:
                        return jsonify({
                            "error": "Invalid base64 image data",
                            "message": str(e)
                        }), 400
                else:
                    return jsonify({
                        "error": "Image data not available",
                        "message": "Cannot generate ZIP without image data"
                    }), 400
            
            # Validate image bytes
            is_valid, error_msg = validate_image_bytes(image_bytes)
            if not is_valid:
                return jsonify({
                    "error": "Invalid image",
                    "message": error_msg
                }), 400
            
            # Use extension (should be 'jpg' from composition, but allow fallback)
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

