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
    """Plan 3 ads following ACE_ENGINE_HE.txt specification exactly. Returns strict JSON with run_index and ads array."""
    user_prompt = f"""Product Name: {product_name}
Product Description: {product_description}
Size: {size}
Run Index: {run_index}

Follow the ACE engine specification EXACTLY. Execute these steps in order:

STEP 1: Define 3 distinct advertising goals (מטרות פרסום)
- Derive from product description
- Each goal is an advertising message (מסר פרסומי)
- Goals must be different from each other

STEP 2: For each advertising goal, generate exactly 100 associations
- Each association must be ONE physical, tangible, photographable object
- No scenes, no words, no non-objects
- Objects can express emotion/idea/action/state, but must be physical objects
- Objects may express meaning of the advertising goal (perceptual meaning only)

STEP 3: Group objects into graphic shapes
- Sort objects into simple graphic shapes (max 25 shapes total)
- Each object belongs to one shape category

STEP 4: Pair objects by shape similarity (hierarchical pairing)
- Create pairs based on shape similarity
- Order pairs from highest shape similarity to lowest
- Consider advertising message when pairing (all objects are associations of the goal)

STEP 5: Split pairs into two groups
- Group 1: Pairs with very high shape similarity → mode "replacement" (החלפה)
- Group 2: Pairs with lower shape similarity → mode "side_by_side" (זה לצד זה)
- Reject pairs with no similarity (cannot appear in visual)

STEP 6: Assign modes to 3 ads (arbitrary assignment)
- Exactly 2 ads with mode="replacement"
- Exactly 1 ad with mode="side_by_side"
- Assignment to advertising goals is arbitrary

STEP 7: For each ad, determine:
- object_a: first physical object from selected pair
- object_b: second physical object from selected pair
- environment_context: existence-inference context for objects (NOT a scene, NOT narrative, adds NO message)
  * In replacement mode: environment is the replaced object's environment
- visual_description: matches the mode rules
  * replacement: one object replaces another, replacing object stands in replaced object's environment
  * side_by_side: both objects appear together side by side
- headline: INTERPRETS the visual, does NOT describe visual/objects/environment, 3-7 words, includes "{product_name}"
- marketing_text: ~50 words, loyal to advertising message, does NOT describe visual, does NOT add new message
- image_prompt: photorealistic photograph description (realistic photography only, NO vector/illustration/3D/CGI/AI effects/synthetic look). 
  * Visual structure: always based on a PAIR of objects (even in replacement mode)
  * No text/logos/labels/packaging text anywhere (graphics only if physically inherent: playing cards, dice, engraved compass)
  * If object requires textual interpretation to understand it → FORBIDDEN
  * Describe the visual composition without mentioning headline (headline is rendered separately)

Return ONLY valid JSON according to this schema. No prose, no markdown:

{{
  "run_index": {run_index},
  "ads": [
    {{
      "ad_index": 1,
      "message": "advertising goal 1",
      "mode": "replacement",
      "object_a": "physical object",
      "object_b": "physical object",
      "environment_context": "existence-inference context (not scene, not narrative)",
      "visual_description": "description matching mode rules",
      "headline": "interprets visual, 3-7 words, includes {product_name}",
      "marketing_text": "~50 words, loyal to message, does not describe visual",
      "image_prompt": "photorealistic visual description (no headline mentioned)"
    }},
    {{ "ad_index": 2, ... }},
    {{ "ad_index": 3, ... }}
  ]
}}

CRITICAL: Exactly 2 ads with mode="replacement", exactly 1 with mode="side_by_side"."""

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


def build_visual_prompt_v3(mode, object_a, object_b, environment_context, visual_description):
    """Build short, strict visual prompt (v3) following ACE_ENGINE_HE.txt. Max 950 chars after compression."""
    mode_lower = mode.lower()
    
    if mode_lower == "replacement":
        # SWAP mode: object A replaces object B in B's environment
        prompt = f"SWAP: {object_a} replaces {object_b} in {object_b}'s environment. {object_a} stands where {object_b} would be. Photorealistic product photography, DSLR, realistic lighting. Physical objects: {object_a}, {object_b}. Environment: {environment_context}. No text, no logos, no labels, no typography."
    elif mode_lower == "side_by_side":
        # SIDE-BY-SIDE mode: both objects shown together
        prompt = f"SIDE-BY-SIDE: {object_a} and {object_b} shown together side by side. Photorealistic product photography, DSLR, realistic lighting. Physical objects: {object_a}, {object_b}. Environment: {environment_context}. No text, no logos, no labels, no typography."
    else:
        # Fallback
        prompt = f"Photorealistic product photography, DSLR, realistic lighting. Objects: {object_a}, {object_b}. Environment: {environment_context}. No text, no logos, no labels."
    
    return prompt


def clamp_prompt(prompt, max_len=950):
    """Compress prompt if too long. Never removes: mode, objectA/objectB, 'photorealistic photo', 'no text/logos/labels'"""
    if len(prompt) <= max_len:
        logger.info(f"prompt_len={len(prompt)} (within limit)")
        return prompt
    
    original_len = len(prompt)
    logger.warning(f"Prompt too long: {original_len} chars, compressing to max {max_len}")
    
    # Split into sentences/clauses
    parts = prompt.split('.')
    
    # Identify critical parts that must be kept
    critical_keywords = ['SWAP:', 'SIDE-BY-SIDE:', 'photorealistic', 'DSLR', 'realistic lighting', 
                        'No text', 'no logos', 'no labels', 'no typography']
    
    # Keep critical parts first
    essential_parts = []
    optional_parts = []
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        is_critical = any(keyword.lower() in part.lower() for keyword in critical_keywords)
        # Also check if it contains object names (likely critical)
        if 'object' in part.lower() or 'Physical' in part or 'Environment' in part:
            is_critical = True
        
        if is_critical:
            essential_parts.append(part)
        else:
            optional_parts.append(part)
    
    # Build compressed prompt starting with essential parts
    compressed = '. '.join(essential_parts)
    
    # Add optional parts if space allows
    if len(compressed) < max_len:
        remaining_space = max_len - len(compressed) - 10  # Reserve for separators
        for part in optional_parts:
            candidate = compressed + '. ' + part
            if len(candidate) <= max_len:
                compressed = candidate
            else:
                break
    
    # Final safety: truncate if still too long (shouldn't happen with critical parts only)
    if len(compressed) > max_len:
        # Remove extra spaces, shorten long words
        compressed = compressed[:max_len-3].rstrip() + '...'
    
    # Ensure it ends properly
    if not compressed.endswith('.'):
        compressed += '.'
    
    logger.info(f"prompt_len={len(compressed)} (compressed from {original_len})")
    return compressed


def generate_visual_image_openai(mode, object_a, object_b, environment_context, visual_description, ad_index, requested_width, requested_height):
    """Generate VISUAL image at 1024x1024 internally following ACE_ENGINE_HE.txt rules exactly. Visual does NOT include headline (rendered separately)."""
    try:
        # Always generate at 1024x1024 internally for visual component (Size Adapter)
        internal_size = "1024x1024"
        
        # Build mode-specific visual prompt using v3 builder (short, strict)
        raw_prompt = build_visual_prompt_v3(mode, object_a, object_b, environment_context, visual_description)
        
        # Compress prompt to ensure <= 950 chars
        final_prompt = clamp_prompt(raw_prompt, max_len=950)
        
        # Log generation start
        logger.info(f"Generating VISUAL (ACE_ENGINE_HE.txt) for ad {ad_index} with internal_size={internal_size} (requested_size={requested_width}x{requested_height})")
        
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
                return image_bytes, final_prompt  # Return bytes and prompt for logging
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


def render_headline_image(headline_text, headline_area_width, headline_area_height):
    """Render headline as a separate image with transparent background. Headline must be large, bold, highly readable."""
    try:
        # Headline area is exactly allocated (50% of canvas minus gap)
        headline_width = headline_area_width
        headline_height = headline_area_height
        
        # Create transparent background image for headline
        headline_img = Image.new('RGBA', (headline_width, headline_height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(headline_img)
        
        # Calculate font size to fill ~80-90% of headline area (large, bold, highly readable)
        # Make it LARGE and BOLD for equal dominance (50% of canvas)
        # Headline must be equally prominent as visual
        initial_font_size = max(96, min(headline_height // 1.5, int(headline_width / max(len(headline_text), 1) * 0.6) if headline_text else 120))
        font_size = initial_font_size
        font = None
        
        # Try multiple font paths (system-dependent) - prefer bold fonts
        font_paths = [
            "arial.ttf",  # Windows
            "Arial.ttf",  # Windows alternate
            "/System/Library/Fonts/Helvetica.ttc",  # macOS
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux (bold)
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",  # Linux alternative (bold)
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux fallback
        ]
        
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except (OSError, IOError):
                continue
        
        if font is None:
            # Fallback to default font - try to make it as large as possible
            try:
                font = ImageFont.load_default()
            except:
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
    """Compose final ad image: visual + headline on canvas of requested size following ACE_ENGINE_HE.txt deterministic layout"""
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
        
        # Fixed gap (48px) to ensure visual and headline do not touch (ACE_ENGINE_HE.txt: הפרדה בין רכיבים)
        gap = 48
        
        # Deterministic layout template per ACE_ENGINE_HE.txt: exactly 50% visual, 50% headline with fixed gap
        # Equal dominance (בולטות שווה): visual and headline are equally prominent
        if requested_width > requested_height:  # Landscape: 1536x1024
            # Left half = visual, right half = headline (big text), with fixed gap
            # Each gets exactly 50% of width (minus gap)
            available_width = requested_width - gap
            visual_area_width = available_width // 2  # Exactly 50% for visual
            headline_area_width = available_width // 2  # Exactly 50% for headline
            visual_area_height = requested_height  # Full height
            headline_area_height = requested_height  # Full height
            
            # Render headline with allocated area
            headline_img = render_headline_image(headline_text, headline_area_width, headline_area_height)
            
            # Scale visual to fill allocated area (50% width, full height)
            visual_aspect = visual_img.width / visual_img.height
            target_visual_aspect = visual_area_width / visual_area_height
            
            if visual_aspect > target_visual_aspect:
                # Visual is wider: fit to height, center horizontally
                scaled_visual_height = visual_area_height
                scaled_visual_width = int(visual_area_height * visual_aspect)
            else:
                # Visual is taller: fit to width
                scaled_visual_width = visual_area_width
                scaled_visual_height = int(visual_area_width / visual_aspect)
            
            scaled_visual = visual_img.resize((scaled_visual_width, scaled_visual_height), Image.Resampling.LANCZOS)
            
            # Headline is already rendered at correct size, just convert to RGB
            headline_rgb = Image.new('RGB', headline_img.size, (255, 255, 255))
            headline_rgb.paste(headline_img, mask=headline_img.split()[3] if headline_img.mode == 'RGBA' else None)
            
            # Position visual: left half, centered vertically
            visual_x = 0
            visual_y = (requested_height - scaled_visual_height) // 2
            
            # Position headline: right half (after gap), centered vertically
            headline_x = visual_area_width + gap
            headline_y = (requested_height - headline_img.height) // 2
            
        elif requested_height > requested_width:  # Portrait: 1024x1536
            # Top half = visual, bottom half = headline, with fixed gap
            available_height = requested_height - gap
            visual_area_width = requested_width  # Full width
            visual_area_height = available_height // 2  # Exactly 50% for visual
            headline_area_width = requested_width  # Full width
            headline_area_height = available_height // 2  # Exactly 50% for headline
            
            # Render headline with allocated area
            headline_img = render_headline_image(headline_text, headline_area_width, headline_area_height)
            
            # Scale visual to fill allocated area (full width, 50% height)
            visual_aspect = visual_img.width / visual_img.height
            target_visual_aspect = visual_area_width / visual_area_height
            
            if visual_aspect > target_visual_aspect:
                # Visual is wider: fit to height, center horizontally
                scaled_visual_height = visual_area_height
                scaled_visual_width = int(visual_area_height * visual_aspect)
            else:
                # Visual is taller: fit to width
                scaled_visual_width = visual_area_width
                scaled_visual_height = int(visual_area_width / visual_aspect)
            
            scaled_visual = visual_img.resize((scaled_visual_width, scaled_visual_height), Image.Resampling.LANCZOS)
            
            # Headline is already rendered at correct size, just convert to RGB
            headline_rgb = Image.new('RGB', headline_img.size, (255, 255, 255))
            headline_rgb.paste(headline_img, mask=headline_img.split()[3] if headline_img.mode == 'RGBA' else None)
            
            # Position visual: top half, centered horizontally
            visual_x = (requested_width - scaled_visual_width) // 2
            visual_y = 0
            
            # Position headline: bottom half (after gap), centered horizontally
            headline_x = (requested_width - headline_img.width) // 2
            headline_y = visual_area_height + gap
            
        else:  # Square: 1024x1024
            # Top half = visual (scaled to fill), bottom half = headline (big text), with fixed gap
            available_height = requested_height - gap
            visual_area_width = requested_width  # Full width
            visual_area_height = available_height // 2  # Exactly 50% for visual
            headline_area_width = requested_width  # Full width
            headline_area_height = available_height // 2  # Exactly 50% for headline
            
            # Render headline with allocated area
            headline_img = render_headline_image(headline_text, headline_area_width, headline_area_height)
            
            # Scale visual to fill allocated area (full width, 50% height)
            visual_aspect = visual_img.width / visual_img.height
            target_visual_aspect = visual_area_width / visual_area_height
            
            if visual_aspect > target_visual_aspect:
                # Visual is wider: fit to height, center horizontally
                scaled_visual_height = visual_area_height
                scaled_visual_width = int(visual_area_height * visual_aspect)
            else:
                # Visual is taller: fit to width
                scaled_visual_width = visual_area_width
                scaled_visual_height = int(visual_area_width / visual_aspect)
            
            scaled_visual = visual_img.resize((scaled_visual_width, scaled_visual_height), Image.Resampling.LANCZOS)
            
            # Headline is already rendered at correct size, just convert to RGB
            headline_rgb = Image.new('RGB', headline_img.size, (255, 255, 255))
            headline_rgb.paste(headline_img, mask=headline_img.split()[3] if headline_img.mode == 'RGBA' else None)
            
            # Position visual: top half, centered horizontally
            visual_x = (requested_width - scaled_visual_width) // 2
            visual_y = 0
            
            # Position headline: bottom half (after gap), centered horizontally
            headline_x = (requested_width - headline_img.width) // 2
            headline_y = visual_area_height + gap
        
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
                environment_context = ad_plan.get('environment_context', '')
                visual_description = ad_plan.get('visual_description', '')
                
                # Mode: use "SWAP" for replacement (החלפה), "SIDE_BY_SIDE" for side_by_side (זה לצד זה)
                mode_label = "SWAP" if mode.lower() == "replacement" else "SIDE_BY_SIDE"
                
                # Ensure marketing_text is approximately 50 words (trim if needed) per ACE rules
                words = marketing_text.split()
                if len(words) > 60:
                    marketing_text = " ".join(words[:50])
                    logger.warning(f"Ad {ad_index} marketing_text trimmed from {len(words)} to 50 words")
                elif len(words) < 40:
                    logger.warning(f"Ad {ad_index} marketing_text is too short ({len(words)} words, expected ~50)")
                
                # Step B: Generate visual image at 1024x1024 internally (Size Adapter) following ACE_ENGINE_HE.txt
                logger.info(f"requested_size={width}x{height} internal_gen_size=1024x1024 final_canvas_size={width}x{height}")
                visual_bytes, visual_prompt = generate_visual_image_openai(
                    mode, object_a, object_b, environment_context, visual_description, ad_index, width, height
                )
                
                # Log per ad with ALL required fields (MANDATORY per requirements)
                logger.info(f"EngineDocument=ACE_ENGINE_HE.txt ad_index={ad_index} mode={mode_label} objectA={object_a} objectB={object_b} headline_text='{headline}' visual_prompt='{visual_prompt}'")
                
                # Step B: Compose final ad (visual + headline) on canvas of requested size with 50/50 dominance
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

