"""
ACE Engine Interface

This module defines the engine interface for generating ads.
STEP 2: Core Decision Logic - implements canonical decision rules for A/B pair evaluation.
STEP 2.5: Object & Input Sanitation - hard exclusions before geometry evaluation.
STEP 3: Goal → 80 Associations → Object Pool - builds intent and object candidates.
"""

from typing import Dict, Any, Tuple, Optional, List
from PIL import Image, ImageDraw, ImageFont
import io
import hashlib
import re
from enum import Enum
from dataclasses import dataclass


# ============================================================================
# STEP 2: CORE DECISION LOGIC - Shared Types and State
# ============================================================================

class HybridType(Enum):
    """Classification of hybrid types"""
    CORE_GEOMETRIC = "core_geometric"  # Based on outer silhouette, unlimited
    MATERIAL_ANALOGY = "material_analogy"  # Material transformation, 1 per batch
    STRUCTURAL_MORPHOLOGY = "structural_morphology"  # Biological/architectural structure, 1 per batch
    STRUCTURAL_PATTERN = "structural_pattern"  # Micro-structure/repeated units, 1 per batch
    SIDE_BY_SIDE = "side_by_side"  # No hybrid, forced side-by-side


class BatchQuotaState:
    """Tracks quota usage across a 3-ad batch"""
    def __init__(self):
        self.material_analogy_used = False
        self.structural_morphology_used = False
        self.structural_exception_used = False
    
    def can_use_material_analogy(self) -> bool:
        """Check if material analogy quota is available"""
        return not self.material_analogy_used
    
    def use_material_analogy(self):
        """Mark material analogy as used"""
        self.material_analogy_used = True
    
    def can_use_structural_morphology(self) -> bool:
        """Check if structural morphology quota is available"""
        return not self.structural_morphology_used
    
    def use_structural_morphology(self):
        """Mark structural morphology as used"""
        self.structural_morphology_used = True
    
    def can_use_structural_exception(self) -> bool:
        """Check if structural exception quota is available"""
        return not self.structural_exception_used
    
    def use_structural_exception(self):
        """Mark structural exception as used"""
        self.structural_exception_used = True


def quota_state_from_dict(d: Optional[Dict[str, bool]]) -> BatchQuotaState:
    """
    Create BatchQuotaState from dictionary.
    
    Args:
        d: Dictionary with keys: material_analogy_used, structural_morphology_used, structural_exception_used
           If None or missing keys, defaults to False
    
    Returns:
        BatchQuotaState instance
    """
    state = BatchQuotaState()
    if d:
        state.material_analogy_used = d.get('material_analogy_used', False)
        state.structural_morphology_used = d.get('structural_morphology_used', False)
        state.structural_exception_used = d.get('structural_exception_used', False)
    return state


def quota_state_to_dict(state: BatchQuotaState) -> Dict[str, bool]:
    """
    Convert BatchQuotaState to dictionary.
    
    Args:
        state: BatchQuotaState instance
    
    Returns:
        Dictionary with quota flags
    """
    return {
        'material_analogy_used': state.material_analogy_used,
        'structural_morphology_used': state.structural_morphology_used,
        'structural_exception_used': state.structural_exception_used
    }


# ============================================================================
# STEP 2.5: OBJECT & INPUT SANITATION (Hard Exclusions Before Geometry)
# ============================================================================
#
# HARD PROHIBITIONS (no exceptions, no quotas can override):
# 1. Objects with readable text are FORBIDDEN
# 2. Objects with logos/brands are FORBIDDEN
# 3. Objects with labels/packaging text are FORBIDDEN
# 4. Objects with communicative graphics are FORBIDDEN
# 5. Objects that are purely symbolic (symbolic-only) are FORBIDDEN
#
# STRICT STRUCTURAL-GRAPHICS EXCEPTION:
# - Graphics are allowed ONLY if:
#   - has_structural_graphics_only == True
#   - AND no text/logo/label/communicative graphics present
#   - AND graphics are physically embedded (e.g., playing cards, dice pips, engraved compass)
#   - AND graphics are non-communicative (not used for branding/labeling)
#
# ASSOCIATION REPRESENTATION RULE:
# - Object must be a practical tool/carrier for the association
# - Purely symbolic objects (is_symbolic_only == True) are rejected
# - Object must have physical presence, not just conceptual meaning
# ============================================================================

@dataclass
class ObjectCandidate:
    """
    Data model for object candidates with fields to support hard filtering.
    
    Fields:
    - name: Object name/identifier
    - is_physical_object: True if object is physical and photographable
    - contains_readable_text: True if object has readable text (prohibited)
    - contains_logo_or_brand: True if object has logos/brands (prohibited)
    - has_label_or_packaging_text: True if object has labels/packaging text (prohibited)
    - has_communicative_graphics: True if object has communicative graphics (prohibited)
    - has_structural_graphics_only: True if graphics are physically embedded and non-communicative (allowed exception)
    - is_symbolic_only: True if object is purely symbolic (e.g., graduation cap for "education") (prohibited)
    - association_rank: Rank in association list (1..80)
    - shape_family: Geometric shape family for overlap calculation (default "unknown")
    """
    name: str
    is_physical_object: bool
    contains_readable_text: bool
    contains_logo_or_brand: bool
    has_label_or_packaging_text: bool
    has_communicative_graphics: bool
    has_structural_graphics_only: bool
    is_symbolic_only: bool
    association_rank: int
    shape_family: str = "unknown"


def is_object_graphically_eligible(obj: ObjectCandidate) -> bool:
    """
    Hard filter: Check if object passes graphic/text/label prohibitions.
    
    HARD PROHIBITIONS (immediate rejection):
    - contains_readable_text → REJECT
    - contains_logo_or_brand → REJECT
    - has_label_or_packaging_text → REJECT
    - has_communicative_graphics → REJECT
    
    STRICT STRUCTURAL-GRAPHICS EXCEPTION:
    - Graphics allowed ONLY if has_structural_graphics_only == True
    - AND no text/logo/label/communicative graphics present
    
    Args:
        obj: ObjectCandidate to evaluate
    
    Returns:
        True if object passes graphic eligibility, False if rejected
    """
    # Hard prohibition: reject if contains any readable text
    if obj.contains_readable_text:
        return False
    
    # Hard prohibition: reject if contains logos or brands
    if obj.contains_logo_or_brand:
        return False
    
    # Hard prohibition: reject if has labels or packaging text
    if obj.has_label_or_packaging_text:
        return False
    
    # Hard prohibition: reject if has communicative graphics
    if obj.has_communicative_graphics:
        return False
    
    # Structural graphics exception: allowed ONLY if:
    # - has_structural_graphics_only == True
    # - AND all text/logo/label prohibitions already passed above
    # If object has graphics but NOT structural-only, it's already rejected by has_communicative_graphics check
    
    # Object passes graphic eligibility
    return True


def is_object_association_eligible(obj: ObjectCandidate) -> bool:
    """
    Hard filter: Check if object passes association representation rule.
    
    ASSOCIATION REPRESENTATION RULE:
    - Object must be a practical tool/carrier for the association
    - Purely symbolic objects (is_symbolic_only == True) are REJECTED
    - Object must have physical presence, not just conceptual meaning
    
    Args:
        obj: ObjectCandidate to evaluate
    
    Returns:
        True if object passes association eligibility, False if rejected
    """
    # Hard prohibition: reject if object is purely symbolic
    if obj.is_symbolic_only:
        return False
    
    # Object must be a physical object (practical tool/carrier)
    if not obj.is_physical_object:
        return False
    
    # Object passes association eligibility
    return True


def sanitize_candidate_pool(objs: List[ObjectCandidate]) -> List[ObjectCandidate]:
    """
    Apply hard sanitation filters to object candidate pool.
    
    Filters applied (in order):
    1. Graphic eligibility (text/logo/label/communicative graphics prohibitions)
    2. Association eligibility (symbolic-only prohibition)
    
    Both filters must pass for object to remain in pool.
    
    Args:
        objs: List of ObjectCandidate objects to sanitize
    
    Returns:
        Sanitized list of eligible ObjectCandidate objects
    
    Raises:
        ValueError: If sanitization removes all candidates and no valid pool exists
    """
    sanitized = []
    
    for obj in objs:
        # Apply both hard filters
        if is_object_graphically_eligible(obj) and is_object_association_eligible(obj):
            sanitized.append(obj)
    
    # Hard gate: if no valid candidates remain, FAIL explicitly
    if len(sanitized) == 0:
        raise ValueError("No valid object candidates after sanitation")
    
    return sanitized


# ============================================================================
# STEP 6: MARKETING TEXT GENERATION (Deterministic, Engine-Level)
# ============================================================================
# Generate persuasive marketing text that supports the advertising goal
# without describing the visual. Plain text only, ~50 words (±10).
# ============================================================================

def count_words(text: str) -> int:
    """
    Count words in plain text (simple whitespace-based).
    
    Args:
        text: Plain text string
    
    Returns:
        Word count (integer)
    """
    return len(text.split())


def generate_marketing_text(product_name: str, product_description: str, goal: str, ad_index: int = 0) -> str:
    """
    Generate marketing text that supports the advertising goal.
    
    Hard constraints:
    - Plain text only (no bullets, no formatting)
    - Length: exactly 40-60 words (enforced)
    - Must include product_name exactly as given
    - Must support and reinforce the (hidden) advertising goal conceptually
    - MUST NOT literally describe the visual (no "in the image", "shown", "pictured", "you see", etc.)
    - Tone: persuasive marketing, not technical
    - Deterministic but varied per ad_index (different sentences for each ad)
    
    Args:
        product_name: Name of the product (must appear in output)
        product_description: Description of the product (for context)
        goal: Advertising goal from ALLOWED_GOALS (hidden, not in output)
        ad_index: Index of ad in batch (0, 1, or 2) for variation
    
    Returns:
        Marketing text string (40-60 words, includes product_name, sanitized)
    """
    # Goal-specific sentence templates (deterministic, no external APIs)
    # Each template set supports the goal conceptually without describing visuals
    templates = {
        "speed": [
            f"{product_name} delivers unmatched speed and performance.",
            f"Experience the power of {product_name} with lightning-fast results.",
            f"{product_name} accelerates your success with instant responsiveness.",
            f"Get ahead faster with {product_name} and its rapid capabilities.",
            f"{product_name} brings you swift solutions that save valuable time.",
            f"Unlock rapid performance with {product_name} and its instant capabilities.",
            f"{product_name} transforms speed into tangible results every time.",
            f"Discover the velocity advantage that {product_name} provides daily."
        ],
        "safety": [
            f"{product_name} provides reliable protection and peace of mind.",
            f"Trust {product_name} for secure and safe performance every time.",
            f"With {product_name}, you get dependable safety and robust security.",
            f"{product_name} ensures your protection with proven reliability.",
            f"Count on {product_name} for secure and trustworthy results.",
            f"Experience unwavering security through {product_name} and its protection.",
            f"{product_name} stands guard with consistent safety measures always.",
            f"Rely on {product_name} for comprehensive security and peace."
        ],
        "durability": [
            f"{product_name} is built to last with exceptional strength and resilience.",
            f"Invest in {product_name} for long-lasting quality and enduring performance.",
            f"{product_name} offers robust durability that stands the test of time.",
            f"Choose {product_name} for reliable toughness and lasting value.",
            f"{product_name} delivers solid construction and permanent reliability.",
            f"Experience lasting quality through {product_name} and its resilient design.",
            f"{product_name} maintains strength through years of continuous use.",
            f"Build on the foundation that {product_name} provides permanently."
        ],
        "freshness": [
            f"{product_name} brings you crisp, clean quality that feels new every day.",
            f"Experience the vibrant freshness of {product_name} with every use.",
            f"{product_name} maintains pure, natural freshness that revitalizes.",
            f"Enjoy the clean, crisp benefits of {product_name} consistently.",
            f"{product_name} delivers fresh, pure results that feel alive.",
            f"Revitalize your experience with {product_name} and its natural purity.",
            f"{product_name} keeps things vibrant and new throughout daily use.",
            f"Feel the renewal that {product_name} brings to every moment."
        ],
        "clarity": [
            f"{product_name} provides crystal-clear results with precise focus.",
            f"See the difference with {product_name} and its sharp clarity.",
            f"{product_name} delivers transparent, clear outcomes you can trust.",
            f"Experience perfect clarity with {product_name} and its defined precision.",
            f"{product_name} brings you sharp, focused clarity in every detail.",
            f"Gain clear understanding through {product_name} and its precise approach.",
            f"{product_name} illuminates every aspect with perfect transparency.",
            f"Navigate with confidence using {product_name} and its clear guidance."
        ],
        "efficiency": [
            f"{product_name} maximizes your productivity with smart, streamlined performance.",
            f"Optimize your results with {product_name} and its efficient design.",
            f"{product_name} delivers effective solutions that work smarter, not harder.",
            f"Experience peak efficiency with {product_name} and its optimized approach.",
            f"{product_name} provides intelligent, productive solutions that save resources.",
            f"Streamline your workflow with {product_name} and its intelligent systems.",
            f"{product_name} transforms effort into achievement through smart optimization.",
            f"Multiply your results using {product_name} and its efficient methods."
        ],
        "comfort": [
            f"{product_name} offers gentle comfort and relaxing ease for everyday use.",
            f"Enjoy the soft, cozy benefits of {product_name} throughout your day.",
            f"{product_name} provides pleasant comfort that makes everything easier.",
            f"Feel the smooth, gentle comfort of {product_name} consistently.",
            f"{product_name} delivers relaxing ease and comfortable satisfaction.",
            f"Embrace the warmth that {product_name} brings to every experience.",
            f"{product_name} creates a soothing environment for daily activities.",
            f"Find your perfect comfort zone with {product_name} and its gentle touch."
        ],
        "precision": [
            f"{product_name} delivers exact accuracy with meticulous attention to detail.",
            f"Experience precise results with {product_name} and its refined accuracy.",
            f"{product_name} provides careful precision that ensures perfect outcomes.",
            f"Trust {product_name} for accurate, detailed results every single time.",
            f"{product_name} brings you exact precision with careful, refined quality.",
            f"Achieve perfection through {product_name} and its meticulous craftsmanship.",
            f"{product_name} ensures flawless execution with its precise engineering.",
            f"Master every detail with {product_name} and its exacting standards."
        ]
    }
    
    # Select template set for the goal
    if goal not in templates:
        goal = "efficiency"  # Fallback
    
    # Create deterministic seed based on product info and ad_index for variation
    seed_string = f"{product_name}{product_description}{goal}{ad_index}"
    seed_hash = hashlib.sha256(seed_string.encode('utf-8')).hexdigest()
    seed_int = int(seed_hash[:8], 16)  # Use first 8 hex chars as integer
    
    # Select 3-4 different sentences using seed (deterministic but varied per ad_index)
    # Avoid consecutive selection to reduce repetition
    available_templates = templates[goal]
    num_templates = len(available_templates)
    num_sentences = 4  # Target 4 sentences
    
    # Select unique indices deterministically (avoid duplicates)
    selected_indices = set()
    k = 0
    while len(selected_indices) < num_sentences and k < num_templates * 2:
        # Use seed with different multipliers to get diverse indices
        candidate_idx = (seed_int + k * 7) % num_templates
        if candidate_idx not in selected_indices:
            selected_indices.add(candidate_idx)
        k += 1
    
    # Convert to sorted list for deterministic order, then select first num_sentences
    selected_indices = sorted(list(selected_indices))[:num_sentences]
    selected_sentences = [available_templates[idx] for idx in selected_indices]
    
    # Anti-repetition: Check if multiple sentences start with same 1-2 words
    # If so, replace one with an alternate that starts differently but includes product_name
    alternate_templates = {
        "speed": [
            f"Rapid performance defines {product_name} and its capabilities.",
            f"Speed becomes reality through {product_name} and its design.",
            f"Velocity meets reliability in {product_name} and its approach."
        ],
        "safety": [
            f"Protection you can trust comes from {product_name} and its design.",
            f"Security finds its foundation in {product_name} and its reliability.",
            f"Peace of mind starts with {product_name} and its approach."
        ],
        "durability": [
            f"Longevity defines {product_name} and its construction.",
            f"Endurance becomes reality through {product_name} and its design.",
            f"Strength meets reliability in {product_name} and its approach."
        ],
        "freshness": [
            f"Vitality you can feel comes from {product_name} and its quality.",
            f"Renewal finds its source in {product_name} and its nature.",
            f"Revitalization starts with {product_name} and its approach."
        ],
        "clarity": [
            f"Precision you can trust comes from {product_name} and its design.",
            f"Transparency finds its foundation in {product_name} and its quality.",
            f"Focus starts with {product_name} and its approach."
        ],
        "efficiency": [
            f"Productivity you can count on comes from {product_name} and its design.",
            f"Optimization finds its source in {product_name} and its systems.",
            f"Streamlining starts with {product_name} and its approach."
        ],
        "comfort": [
            f"Ease you can feel comes from {product_name} and its design.",
            f"Relaxation finds its source in {product_name} and its quality.",
            f"Gentleness starts with {product_name} and its approach."
        ],
        "precision": [
            f"Accuracy you can trust comes from {product_name} and its design.",
            f"Exactness finds its foundation in {product_name} and its quality.",
            f"Perfection starts with {product_name} and its approach."
        ]
    }
    
    # Check for repetitive sentence starts (same 1-2 words)
    sentence_starts = {}
    for i, sentence in enumerate(selected_sentences):
        # Get first 2 words (or first word if only one)
        words = sentence.split()[:2]
        start_key = " ".join(words).lower()
        if start_key in sentence_starts:
            # Found repetition - replace with alternate
            if goal in alternate_templates and len(alternate_templates[goal]) > 0:
                alt_idx = (seed_int + i * 11) % len(alternate_templates[goal])
                selected_sentences[i] = alternate_templates[goal][alt_idx]
                # Update start key for new sentence
                new_words = selected_sentences[i].split()[:2]
                start_key = " ".join(new_words).lower()
        sentence_starts[start_key] = i
    
    # Remove any duplicate sentences (exact matches)
    unique_sentences = []
    seen = set()
    for sentence in selected_sentences:
        if sentence not in seen:
            unique_sentences.append(sentence)
            seen.add(sentence)
    
    # Ensure we have at least 3 sentences
    if len(unique_sentences) < 3:
        # Add more from templates if needed
        selected_indices_set = set(selected_indices)  # Keep set for fast lookup
        for idx in range(num_templates):
            if idx not in selected_indices_set:
                candidate = available_templates[idx]
                if candidate not in seen:
                    unique_sentences.append(candidate)
                    seen.add(candidate)
                    if len(unique_sentences) >= 3:
                        break
    
    selected_sentences = unique_sentences[:num_sentences]
    
    # Build base text
    base_text = " ".join(selected_sentences)
    
    # Sanitation: Remove forbidden words/phrases related to visual description
    forbidden_visual_words = [
        "in the image", "shown", "pictured", "you see", "visible", "appears",
        "next to", "beside", "background", "silhouette", "object", "hybrid",
        "side-by-side", "visual", "picture", "illustration", "graphic"
    ]
    forbidden_technical_words = [
        "API", "Flask", "Backend", "Prompt", "Engine", "algorithm", "model",
        "אלגוריתם", "מודל", "בתמונה", "רואים", "מופיע", "ליד", "רקע",
        "סילואטה", "אובייקט", "היבריד", "תמונה", "ויזואל"
    ]
    
    # Apply sanitation
    sanitized_text = base_text
    for forbidden in forbidden_visual_words + forbidden_technical_words:
        # Case-insensitive replacement
        sanitized_text = re.sub(re.escape(forbidden), "", sanitized_text, flags=re.IGNORECASE)
        sanitized_text = re.sub(r'\s+', ' ', sanitized_text)  # Normalize whitespace
    
    # Ensure single paragraph (replace \n with space)
    sanitized_text = sanitized_text.replace('\n', ' ').replace('\r', ' ')
    sanitized_text = ' '.join(sanitized_text.split())  # Normalize all whitespace
    
    # Enforce 40-60 word constraint
    word_count = count_words(sanitized_text)
    
    if word_count < 40:
        # Add neutral short sentences to reach minimum (anti-cliché replacements)
        neutral_sentences = [
            "Built for consistent results.",
            "Designed for everyday reliability.",
            "Made to support confident decisions.",
            "Created to stay dependable over time.",
            "Engineered for lasting performance.",
            "Crafted for reliable daily use.",
            "Developed to maintain quality consistently."
        ]
        # Use seed to select which neutral sentence to add
        neutral_idx = (seed_int // 1000) % len(neutral_sentences)
        if neutral_sentences[neutral_idx] not in sanitized_text:
            sanitized_text += " " + neutral_sentences[neutral_idx]
            word_count = count_words(sanitized_text)
        # Add more if still short
        if word_count < 40:
            neutral_idx = (seed_int // 2000) % len(neutral_sentences)
            if neutral_sentences[neutral_idx] not in sanitized_text:
                sanitized_text += " " + neutral_sentences[neutral_idx]
                word_count = count_words(sanitized_text)
            # If still short, add one more
            if word_count < 40:
                neutral_idx = (seed_int // 3000) % len(neutral_sentences)
                if neutral_sentences[neutral_idx] not in sanitized_text:
                    sanitized_text += " " + neutral_sentences[neutral_idx]
                    word_count = count_words(sanitized_text)
    
    if word_count > 60:
        # Trim to maximum by removing last sentence
        sentences = sanitized_text.split(". ")
        while count_words(". ".join(sentences)) > 60 and len(sentences) > 1:
            sentences.pop()
        sanitized_text = ". ".join(sentences)
        if not sanitized_text.endswith("."):
            sanitized_text += "."
        word_count = count_words(sanitized_text)
    
    # Final verification: ensure product_name is included
    final_text = sanitized_text.strip()
    if product_name not in final_text:
        # Safety: prepend product_name if missing
        final_text = f"{product_name}. {final_text}"
        word_count = count_words(final_text)
        # Trim if exceeds limit after adding
        if word_count > 60:
            sentences = final_text.split(". ")
            while count_words(". ".join(sentences)) > 60 and len(sentences) > 1:
                sentences.pop()
            final_text = ". ".join(sentences)
            if not final_text.endswith("."):
                final_text += "."
    
    return final_text


# ============================================================================
# STEP 7: HEADLINE GENERATION (Deterministic, Engine-Level)
# ============================================================================
# Generate headline that includes product_name, 5-6 words exactly.
# Declarative/atmospheric only, no questions, commands, or quotes.
# Does not reveal goal or describe visual.
# ============================================================================

def generate_headline(product_name: str, goal: str, ad_index: int = 0) -> str:
    """
    Generate headline that includes product_name, exactly 5-6 words.
    
    Hard constraints:
    - Exactly 5-6 words (enforced with automatic correction)
    - Must include product_name as part of the 5-6 words
    - Declarative/atmospheric statement only (no questions, commands, quotes)
    - Does not reveal goal or describe visual
    - Deterministic but varied per ad_index (different template for each ad)
    
    Args:
        product_name: Name of the product (must appear in headline)
        goal: Advertising goal (hidden, used for template selection only)
        ad_index: Index of ad in batch (0, 1, or 2) for template variation
    
    Returns:
        Headline string (exactly 5-6 words, includes product_name)
    """
    # Goal-specific templates (deterministic, declarative/atmospheric)
    # Each template produces 5-6 words and includes {product_name}
    templates = {
        "speed": [
            f"{product_name} moves with unmatched velocity.",
            f"{product_name} accelerates beyond expectations.",
            f"{product_name} delivers instant performance today.",
            f"{product_name} transforms speed into results.",
            f"{product_name} brings rapid transformation forward."
        ],
        "safety": [
            f"{product_name} protects what matters most.",
            f"{product_name} ensures reliable security always.",
            f"{product_name} provides trusted safety today.",
            f"{product_name} delivers secure peace of mind.",
            f"{product_name} stands guard for you."
        ],
        "durability": [
            f"{product_name} lasts through every challenge.",
            f"{product_name} endures beyond expectations always.",
            f"{product_name} stands strong over time.",
            f"{product_name} remains resilient through years.",
            f"{product_name} builds lasting strength today."
        ],
        "freshness": [
            f"{product_name} brings new vitality forward.",
            f"{product_name} delivers crisp freshness today.",
            f"{product_name} maintains pure quality always.",
            f"{product_name} revives with natural energy.",
            f"{product_name} awakens fresh possibilities now."
        ],
        "clarity": [
            f"{product_name} reveals truth with precision.",
            f"{product_name} brings sharp focus forward.",
            f"{product_name} delivers crystal clear vision.",
            f"{product_name} illuminates every detail perfectly.",
            f"{product_name} shows the way clearly."
        ],
        "efficiency": [
            f"{product_name} maximizes productivity every day.",
            f"{product_name} optimizes results with intelligence.",
            f"{product_name} streamlines work effortlessly forward.",
            f"{product_name} delivers smart solutions today.",
            f"{product_name} transforms effort into achievement."
        ],
        "comfort": [
            f"{product_name} offers gentle ease always.",
            f"{product_name} brings soft comfort forward.",
            f"{product_name} delivers relaxing peace today.",
            f"{product_name} wraps you in warmth.",
            f"{product_name} creates cozy moments now."
        ],
        "precision": [
            f"{product_name} achieves exact accuracy always.",
            f"{product_name} delivers meticulous perfection today.",
            f"{product_name} measures with careful precision.",
            f"{product_name} ensures perfect detail forward.",
            f"{product_name} crafts refined excellence now."
        ]
    }
    
    # Select template set for the goal
    if goal not in templates:
        goal = "efficiency"  # Fallback
    
    # Select template deterministically based on ad_index (varied per ad)
    template_index = ad_index % len(templates[goal])
    headline = templates[goal][template_index]
    
    # Enforce 5-6 word constraint with automatic correction
    word_count = count_words(headline)
    
    if word_count < 5:
        # Add neutral word to reach minimum
        neutral_words = ["now", "always", "today", "forward", "here"]
        headline += " " + neutral_words[0]
        word_count = count_words(headline)
    
    if word_count > 6:
        # Remove last word if exceeds maximum
        words = headline.split()
        while len(words) > 6:
            words.pop()
        headline = " ".join(words)
        word_count = count_words(headline)
    
    # Final verification: ensure product_name is included and word count is 5-6
    if product_name not in headline:
        # Safety: prepend product_name if missing (should not happen)
        headline = f"{product_name} {headline}"
        word_count = count_words(headline)
        # Trim if needed
        if word_count > 6:
            words = headline.split()
            while len(words) > 6:
                words.pop()
            headline = " ".join(words)
    
    # Final word count check
    final_word_count = count_words(headline)
    if final_word_count < 5 or final_word_count > 6:
        # Last resort: use simple template
        headline = f"{product_name} delivers excellence today."
        if count_words(headline) != 5:
            # Force 5 words
            headline = f"{product_name} brings excellence forward now."
    
    return headline.strip()


# ============================================================================
# STEP 8: SILHOUETTE PLACEHOLDER GENERATION (Visual Representation)
# ============================================================================
# Draw placeholder silhouettes that reflect HYBRID or SIDE_BY_SIDE mode.
# Maintains safe margins and avoids overlap with headline area.
# ============================================================================

def draw_silhouette_placeholder(
    draw: ImageDraw.Draw,
    width: int,
    height: int,
    shape_family_a: str,
    shape_family_b: str,
    hybrid_type: HybridType
) -> None:
    """
    Draw placeholder silhouettes based on shape families and hybrid type.
    
    Rules:
    - SIDE_BY_SIDE: Two separate shapes (left/right) with clear gap
    - HYBRID: One merged shape (center) or significant overlap
    - Shapes drawn in center area, avoiding headline zone
    - No text labels or metadata on image
    
    Args:
        draw: PIL ImageDraw object
        width: Image width
        height: Image height
        shape_family_a: Shape family for object A
        shape_family_b: Shape family for object B
        hybrid_type: HYBRID or SIDE_BY_SIDE mode
    """
    # Safe margins: headline is at height // 8, so silhouette starts below
    silhouette_top = height // 4  # Start below headline area
    silhouette_bottom = height * 3 // 4  # End before bottom
    silhouette_height = silhouette_bottom - silhouette_top
    silhouette_center_y = (silhouette_top + silhouette_bottom) // 2
    
    # Helper function to draw shape based on shape_family
    def draw_shape_by_family(x_center: int, y_center: int, size: int, shape: str, fill_color: str = "gray"):
        """Draw a shape at given center coordinates"""
        half_size = size // 2
        
        if shape in ["circle", "oval"]:
            # Ellipse
            bbox = [x_center - half_size, y_center - half_size // 2, 
                   x_center + half_size, y_center + half_size // 2]
            draw.ellipse(bbox, fill=fill_color, outline="black", width=2)
        
        elif shape in ["rectangle", "square", "box"]:
            # Rectangle
            bbox = [x_center - half_size, y_center - half_size,
                   x_center + half_size, y_center + half_size]
            draw.rectangle(bbox, fill=fill_color, outline="black", width=2)
        
        elif shape in ["triangle"]:
            # Triangle
            points = [
                (x_center, y_center - half_size),  # Top
                (x_center - half_size, y_center + half_size),  # Bottom left
                (x_center + half_size, y_center + half_size)   # Bottom right
            ]
            draw.polygon(points, fill=fill_color, outline="black", width=2)
        
        elif shape in ["teardrop", "drop"]:
            # Teardrop (ellipse + triangle)
            bbox = [x_center - half_size // 2, y_center - half_size // 2,
                   x_center + half_size // 2, y_center + half_size // 2]
            draw.ellipse(bbox, fill=fill_color, outline="black", width=2)
            # Add triangle point at bottom
            points = [
                (x_center, y_center + half_size // 2),
                (x_center - half_size // 2, y_center + half_size),
                (x_center + half_size // 2, y_center + half_size)
            ]
            draw.polygon(points, fill=fill_color, outline="black", width=2)
        
        elif shape in ["leaf", "blade", "wing"]:
            # Leaf shape (elongated ellipse)
            bbox = [x_center - half_size, y_center - half_size // 3,
                   x_center + half_size, y_center + half_size // 3]
            draw.ellipse(bbox, fill=fill_color, outline="black", width=2)
        
        elif shape in ["bottle", "cylinder"]:
            # Bottle/cylinder (rectangle with narrow top)
            # Main body
            bbox = [x_center - half_size // 2, y_center - half_size // 2,
                   x_center + half_size // 2, y_center + half_size]
            draw.rectangle(bbox, fill=fill_color, outline="black", width=2)
            # Narrow neck
            neck_bbox = [x_center - half_size // 4, y_center - half_size,
                        x_center + half_size // 4, y_center - half_size // 2]
            draw.rectangle(neck_bbox, fill=fill_color, outline="black", width=2)
        
        else:
            # Default: circle for unknown shapes
            bbox = [x_center - half_size, y_center - half_size,
                   x_center + half_size, y_center + half_size]
            draw.ellipse(bbox, fill=fill_color, outline="black", width=2)
    
    # Determine shape size based on mode
    if hybrid_type == HybridType.SIDE_BY_SIDE:
        # Two separate shapes with gap
        shape_size = min(width, silhouette_height) // 4
        gap = width // 8
        
        # Left shape (object A)
        left_x = width // 4
        draw_shape_by_family(left_x, silhouette_center_y, shape_size, shape_family_a, "lightblue")
        
        # Right shape (object B)
        right_x = width * 3 // 4
        draw_shape_by_family(right_x, silhouette_center_y, shape_size, shape_family_b, "lightcoral")
    
    else:
        # HYBRID: One merged shape in center
        shape_size = min(width, silhouette_height) // 3
        
        # Draw merged shape (use shape_family_a as base, with hint of shape_family_b)
        center_x = width // 2
        draw_shape_by_family(center_x, silhouette_center_y, shape_size, shape_family_a, "lightgray")
        
        # Overlay hint of shape_b (smaller, slightly offset) to show fusion
        if shape_family_b != shape_family_a:
            offset_x = center_x + shape_size // 4
            offset_y = silhouette_center_y - shape_size // 4
            draw_shape_by_family(offset_x, offset_y, shape_size // 2, shape_family_b, "lightsteelblue")


# ============================================================================
# STEP 3: GOAL → 80 ASSOCIATIONS → OBJECT POOL
# ============================================================================
# This step builds INTENT → ASSOCIATIONS → OBJECT CANDIDATES.
# No visual output, environment, headline, lighting, or composition rules.
# Deterministic implementation with predefined libraries.
# ============================================================================

# Allowed advertising goals (fixed set)
ALLOWED_GOALS = ["speed", "safety", "durability", "freshness", "clarity", "efficiency", "comfort", "precision"]

# Goal keyword mapping (deterministic classification)
GOAL_KEYWORDS = {
    "speed": ["fast", "quick", "rapid", "speed", "velocity", "acceleration", "swift", "instant", "immediate"],
    "safety": ["safe", "secure", "protection", "shield", "guard", "safety", "reliable", "trust", "secure"],
    "durability": ["durable", "strong", "tough", "lasting", "endurance", "resilient", "robust", "sturdy", "long-lasting"],
    "freshness": ["fresh", "new", "crisp", "clean", "pure", "natural", "organic", "vibrant", "alive"],
    "clarity": ["clear", "sharp", "precise", "focused", "crisp", "distinct", "transparent", "visible", "defined"],
    "efficiency": ["efficient", "effective", "productive", "optimized", "streamlined", "smart", "intelligent", "optimal"],
    "comfort": ["comfort", "comfortable", "relaxing", "soft", "gentle", "cozy", "pleasant", "easy", "smooth"],
    "precision": ["precise", "accurate", "exact", "detailed", "refined", "perfected", "meticulous", "careful"]
}

# Predefined associations library (exactly 80 per goal)
# Using expanded lists to reach exactly 80 associations per goal
ASSOCIATIONS_LIBRARY = {
    "speed": [
        "racing car", "jet engine", "arrow", "lightning bolt", "cheetah", "falcon", "bullet train", "rocket",
        "sports car", "motorcycle", "wind", "tornado", "hurricane", "waterfall", "river current", "ocean wave",
        "athlete running", "sprinter", "marathon runner", "cyclist", "skier", "surfer", "skateboard", "roller skates",
        "propeller", "turbine", "fan blade", "rotor", "spinning wheel", "gear", "pulley", "chain",
        "meteor", "comet", "satellite", "spacecraft", "drone", "airplane", "helicopter", "glider",
        "speedboat", "yacht", "sailboat", "kayak", "canoe", "raft", "submarine", "hovercraft",
        "racehorse", "greyhound", "antelope", "gazelle", "deer", "rabbit", "squirrel", "bird",
        "stream", "brook", "creek", "rapids", "cascade", "torrent", "flood", "tsunami",
        "whirlwind", "cyclone", "vortex", "spiral", "helix", "coil", "spring", "elastic",
        "momentum", "velocity", "acceleration", "thrust", "propulsion", "force", "energy", "power"
    ],
    "safety": [
        "helmet", "seatbelt", "airbag", "safety net", "guardrail", "barrier", "fence", "wall",
        "lock", "key", "padlock", "safe", "vault", "strongbox", "security system", "alarm",
        "fire extinguisher", "smoke detector", "sprinkler", "emergency exit", "first aid kit", "bandage", "splint", "stretcher",
        "life jacket", "life preserver", "buoy", "lifeline", "rope", "harness", "anchor", "mooring",
        "shield", "armor", "protective gear", "safety glasses", "goggles", "gloves", "boots", "vest",
        "guard dog", "watchdog", "security guard", "police officer", "soldier", "bodyguard", "sentinel", "lookout",
        "fortress", "castle", "fort", "bunker", "shelter", "refuge", "sanctuary", "haven",
        "insurance", "warranty", "guarantee", "protection", "coverage", "security", "safety", "reliability",
        "stability", "balance", "foundation", "base", "support", "pillar", "column", "beam",
        "caution", "warning", "alert", "signal", "beacon", "lighthouse", "flare", "siren"
    ],
    "durability": [
        "steel beam", "concrete block", "stone wall", "brick", "marble", "granite", "diamond", "titanium",
        "oak tree", "redwood", "cedar", "pine", "bamboo", "ironwood", "teak", "mahogany",
        "chain", "cable", "rope", "wire", "cord", "thread", "fiber", "strand",
        "bridge", "dam", "tunnel", "viaduct", "aqueduct", "arch", "column", "pillar",
        "mountain", "rock", "boulder", "cliff", "ridge", "peak", "summit", "plateau",
        "foundation", "base", "footing", "support", "frame", "structure", "skeleton", "framework",
        "armor", "shield", "protection", "barrier", "defense", "fortification", "rampart", "bulwark",
        "endurance", "stamina", "resilience", "toughness", "strength", "hardness", "rigidity", "solidity",
        "permanence", "stability", "constancy", "persistence", "continuity", "longevity", "endurance", "lasting",
        "reinforcement", "strengthening", "fortification", "consolidation", "solidification", "hardening", "tempering", "annealing"
    ],
    "freshness": [
        "morning dew", "spring water", "rain", "snow", "ice", "frost", "mist", "fog",
        "green leaf", "fresh flower", "bud", "sprout", "seedling", "sapling", "blossom", "bloom",
        "fruit", "vegetable", "herb", "spice", "grain", "seed", "nut", "berry",
        "breeze", "wind", "air", "oxygen", "atmosphere", "sky", "cloud", "sunlight",
        "ocean", "sea", "lake", "river", "stream", "pond", "pool", "fountain",
        "crystal", "gem", "pearl", "coral", "shell", "stone", "mineral", "quartz",
        "dawn", "sunrise", "morning", "daybreak", "twilight", "dusk", "evening", "night",
        "clean", "pure", "clear", "bright", "luminous", "radiant", "shining", "glowing",
        "new", "renewed", "revived", "restored", "refreshed", "rejuvenated", "regenerated", "reborn",
        "vitality", "energy", "vigor", "strength", "health", "wellness", "fitness", "vibrancy"
    ],
    "clarity": [
        "crystal", "glass", "lens", "mirror", "prism", "diamond", "ice", "water",
        "window", "pane", "transparency", "clarity", "visibility", "sharpness", "focus", "precision",
        "magnifying glass", "telescope", "microscope", "binoculars", "periscope", "kaleidoscope", "spectroscope", "monocle",
        "light", "beam", "ray", "shine", "glow", "illumination", "brightness", "luminosity",
        "eye", "pupil", "iris", "retina", "vision", "sight", "perception", "observation",
        "map", "chart", "diagram", "blueprint", "plan", "scheme", "layout", "design",
        "compass", "navigator", "guide", "beacon", "lighthouse", "signal", "marker", "indicator",
        "definition", "explanation", "description", "specification", "detail", "particular", "aspect", "element",
        "understanding", "comprehension", "insight", "awareness", "knowledge", "wisdom", "intelligence", "enlightenment",
        "purity", "simplicity", "plainness", "straightforwardness", "directness", "honesty", "truth", "reality"
    ],
    "efficiency": [
        "gear", "cog", "wheel", "pulley", "lever", "fulcrum", "mechanism", "machine",
        "engine", "motor", "turbine", "generator", "transformer", "converter", "adapter", "connector",
        "pipeline", "conduit", "channel", "duct", "tube", "pipe", "hose", "conduit",
        "circuit", "pathway", "route", "course", "track", "trail", "way", "direction",
        "system", "network", "grid", "matrix", "array", "pattern", "structure", "organization",
        "tool", "instrument", "device", "apparatus", "equipment", "machinery", "implement", "utensil",
        "optimization", "streamlining", "simplification", "refinement", "improvement", "enhancement", "upgrade", "advancement",
        "productivity", "output", "yield", "result", "outcome", "achievement", "accomplishment", "success",
        "speed", "quickness", "rapidity", "swiftness", "celerity", "velocity", "pace", "rate",
        "economy", "thrift", "frugality", "conservation", "preservation", "saving", "efficiency", "effectiveness"
    ],
    "comfort": [
        "pillow", "cushion", "mattress", "blanket", "quilt", "comforter", "duvet", "sheet",
        "sofa", "armchair", "recliner", "chaise", "ottoman", "footstool", "bench", "stool",
        "bed", "hammock", "swing", "rocker", "glider", "cradle", "bassinet", "crib",
        "slippers", "socks", "robe", "pajamas", "sweater", "cardigan", "hoodie", "jacket",
        "warmth", "coziness", "softness", "gentleness", "tenderness", "kindness", "care", "attention",
        "relaxation", "rest", "repose", "ease", "leisure", "peace", "tranquility", "serenity",
        "support", "assistance", "help", "aid", "relief", "solace", "consolation", "comfort",
        "safety", "security", "protection", "shelter", "refuge", "sanctuary", "haven", "retreat",
        "familiarity", "home", "hearth", "fireplace", "warmth", "light", "glow", "radiance",
        "contentment", "satisfaction", "happiness", "joy", "pleasure", "delight", "enjoyment", "bliss"
    ],
    "precision": [
        "compass", "ruler", "scale", "caliper", "micrometer", "gauge", "meter", "measure",
        "clock", "watch", "timer", "chronometer", "stopwatch", "hourglass", "sundial", "pendulum",
        "laser", "beam", "ray", "light", "signal", "pulse", "wave", "frequency",
        "target", "bullseye", "mark", "spot", "point", "dot", "pixel", "coordinate",
        "needle", "pin", "spike", "nail", "tack", "staple", "rivet", "screw",
        "knife", "blade", "razor", "scalpel", "chisel", "awl", "drill", "bit",
        "arrow", "dart", "projectile", "missile", "bullet", "pellet", "shot", "slug",
        "lens", "focus", "magnification", "zoom", "clarity", "sharpness", "definition", "resolution",
        "calibration", "adjustment", "alignment", "positioning", "placement", "location", "placement", "setting",
        "accuracy", "exactness", "correctness", "rightness", "truth", "veracity", "validity", "authenticity"
    ]
}


def derive_advertising_goal(product_name: str, product_description: str) -> str:
    """
    Derive exactly ONE advertising goal from product information (deterministic).
    
    Rules:
    - Goal must be inferred ONLY from product_name + product_description
    - Goal is NEVER returned to frontend or exposed in output (hidden)
    - No user choice, no configuration
    - Goal must be concrete (from fixed allowed set)
    - Uses deterministic keyword-based classification (no randomness)
    
    Args:
        product_name: Name of the product
        product_description: Description of the product
    
    Returns:
        Single concrete advertising goal string from ALLOWED_GOALS
    
    Raises:
        ValueError: If goal derivation fails (no matching keywords found)
    """
    # Combine product name and description for keyword matching
    text = (product_name + " " + product_description).lower()
    
    # Count keyword matches for each goal
    goal_scores = {}
    for goal, keywords in GOAL_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in text)
        if score > 0:
            goal_scores[goal] = score
    
    # If no matches found, FAIL
    if not goal_scores:
        raise ValueError(f"Failed to derive advertising goal: no matching keywords found in product name/description")
    
    # Return goal with highest score
    # Tie-break: if multiple goals have same highest score, choose first in ALLOWED_GOALS order
    max_score = max(goal_scores.values())
    tied_goals = [goal for goal, score in goal_scores.items() if score == max_score]
    
    # Deterministic tie-break: choose goal that appears first in ALLOWED_GOALS
    for goal in ALLOWED_GOALS:
        if goal in tied_goals:
            return goal
    
    # Fallback (should never happen)
    return tied_goals[0]


def generate_associations(goal: str) -> List[str]:
    """
    Generate EXACTLY 80 associations from the advertising goal (deterministic).
    
    Rules:
    - Generate EXACTLY 80 associations (no more, no less)
    - Associations must be physical, experiential, or functional (not abstract words)
    - No symbols, metaphors, or emotions as standalone items
    - No duplicates, no near-duplicates
    - Ordered list: index 1 = strongest association
    - Associations are internal only (not returned to frontend)
    
    Args:
        goal: Advertising goal string
    
    Returns:
        List of exactly 80 association strings, ordered by strength
    
    Raises:
        ValueError: If goal not found in associations library or count is not exactly 80
    """
    if goal not in ASSOCIATIONS_LIBRARY:
        raise ValueError(f"Goal '{goal}' not found in associations library")
    
    associations = ASSOCIATIONS_LIBRARY[goal]
    
    if len(associations) != 80:
        raise ValueError(f"Expected exactly 80 associations for goal '{goal}', got {len(associations)}")
    
    # Validate no duplicates
    if len(set(associations)) != len(associations):
        raise ValueError(f"Duplicate associations found for goal '{goal}'")
    
    return associations


def determine_shape_family(object_name: str) -> str:
    """
    Determine shape family for an object based on its name (deterministic keyword matching).
    
    Args:
        object_name: Name of the object
    
    Returns:
        Shape family string (e.g., "circle", "rectangle", "bottle", etc.) or "unknown"
    """
    name_lower = object_name.lower()
    
    # Circle family
    if any(kw in name_lower for kw in ["ball", "sphere", "orb", "globe", "bead", "pearl", "bubble", "circle", "round"]):
        return "circle"
    
    # Oval family
    if any(kw in name_lower for kw in ["oval", "ellipse", "egg", "almond", "teardrop"]):
        return "oval"
    
    # Rectangle family
    if any(kw in name_lower for kw in ["rectangle", "rectangular", "block", "brick", "slab", "plank", "board"]):
        return "rectangle"
    
    # Square family
    if any(kw in name_lower for kw in ["square", "cube", "box", "crate", "package"]):
        return "square"
    
    # Triangle family
    if any(kw in name_lower for kw in ["triangle", "triangular", "pyramid", "cone", "arrowhead"]):
        return "triangle"
    
    # Leaf family
    if any(kw in name_lower for kw in ["leaf", "petal", "blade", "wing", "feather"]):
        return "leaf"
    
    # Teardrop family
    if any(kw in name_lower for kw in ["teardrop", "drop", "raindrop", "waterdrop"]):
        return "teardrop"
    
    # Bottle family
    if any(kw in name_lower for kw in ["bottle", "flask", "vial", "jar", "container"]):
        return "bottle"
    
    # Cylinder family
    if any(kw in name_lower for kw in ["cylinder", "tube", "pipe", "rod", "pole", "column", "pillar"]):
        return "cylinder"
    
    # Box family (separate from square for compatibility)
    if any(kw in name_lower for kw in ["box", "crate", "case", "chest", "container"]):
        return "box"
    
    # Book family
    if any(kw in name_lower for kw in ["book", "notebook", "journal", "tome", "volume"]):
        return "book"
    
    # Fish family
    if any(kw in name_lower for kw in ["fish", "shark", "dolphin", "whale"]):
        return "fish"
    
    # Wing family
    if any(kw in name_lower for kw in ["wing", "airfoil", "sail"]):
        return "wing"
    
    # Blade family
    if any(kw in name_lower for kw in ["blade", "knife", "sword", "razor", "cutting"]):
        return "blade"
    
    # Bolt family
    if any(kw in name_lower for kw in ["bolt", "lightning", "flash", "zigzag"]):
        return "bolt"
    
    # Helmet family
    if any(kw in name_lower for kw in ["helmet", "cap", "hat", "crown"]):
        return "helmet"
    
    # Shield family
    if any(kw in name_lower for kw in ["shield", "plate", "disc", "discus"]):
        return "shield"
    
    # Default: unknown
    return "unknown"


def map_association_to_objects(association: str, association_rank: int) -> List[ObjectCandidate]:
    """
    Map an association to 1-3 ObjectCandidate instances (deterministic).
    
    Rules:
    - Each association maps to 1-3 ObjectCandidate instances
    - Objects must represent the association as a TOOL / CARRIER, not a symbol
    - Objects must be photographable physical objects
    - Populate ALL ObjectCandidate fields explicitly
    - Uses deterministic mapping: association name becomes object name (default pattern)
    
    Args:
        association: Association string
        association_rank: Rank of the association (1..80)
    
    Returns:
        List of 1-3 ObjectCandidate instances
    
    Raises:
        ValueError: If object mapping fails
    """
    # Deterministic mapping: use association name as object name
    # Default pattern: create 1-2 objects based on association
    # Most associations map to 1 object (the association itself as a physical object)
    # Some map to 2 objects (e.g., "arrow" -> ["arrow", "bow"])
    
    # Default object pattern (most associations use this)
    default_pattern = {
        "is_physical_object": True,
        "contains_readable_text": False,
        "contains_logo_or_brand": False,
        "has_label_or_packaging_text": False,
        "has_communicative_graphics": False,
        "has_structural_graphics_only": False,
        "is_symbolic_only": False
    }
    
    # Special mappings for associations that need 2-3 objects
    special_mappings = {
        "arrow": [{"name": "arrow", **default_pattern}, {"name": "bow", **default_pattern}],
        "lock": [{"name": "lock", **default_pattern}, {"name": "key", **default_pattern}],
        "racing car": [{"name": "racing car", **default_pattern}, {"name": "race track", **default_pattern}],
    }
    
    # Check if association has special mapping
    if association in special_mappings:
        objects_data = special_mappings[association]
    else:
        # Default: single object with association name
        objects_data = [{"name": association, **default_pattern}]
    
    if len(objects_data) == 0 or len(objects_data) > 3:
        raise ValueError(f"Expected 1-3 objects for association '{association}', got {len(objects_data)}")
    
    # Convert to ObjectCandidate instances
    candidates = []
    for obj_data in objects_data:
        # Determine shape_family deterministically from object name
        shape_family = determine_shape_family(obj_data["name"])
        candidate = ObjectCandidate(
            name=obj_data["name"],
            association_rank=association_rank,
            is_physical_object=obj_data["is_physical_object"],
            contains_readable_text=obj_data["contains_readable_text"],
            contains_logo_or_brand=obj_data["contains_logo_or_brand"],
            has_label_or_packaging_text=obj_data["has_label_or_packaging_text"],
            has_communicative_graphics=obj_data["has_communicative_graphics"],
            has_structural_graphics_only=obj_data["has_structural_graphics_only"],
            is_symbolic_only=obj_data["is_symbolic_only"],
            shape_family=shape_family
        )
        candidates.append(candidate)
    
    return candidates


def build_object_pool(product_name: str, product_description: str) -> List[ObjectCandidate]:
    """
    Build sanitized object pool from product information.
    
    Pipeline:
    1) goal = derive_advertising_goal(...)
    2) associations = generate_associations(goal)  # exactly 80
    3) For each association (rank 1..80):
       - objects = map_association_to_objects(...)
       - add to pool with correct association_rank
    4) Apply sanitize_candidate_pool(...) from STEP 2.5
    5) If sanitized pool is empty → FAIL explicitly
    
    Args:
        product_name: Name of the product
        product_description: Description of the product
    
    Returns:
        Sanitized list of ObjectCandidate instances
    
    Raises:
        ValueError: If any step fails or sanitized pool is empty
    """
    # Step 1: Derive advertising goal (hidden, not returned to frontend)
    try:
        goal = derive_advertising_goal(product_name, product_description)
    except Exception as e:
        raise ValueError(f"Failed to derive advertising goal: {str(e)}") from e
    
    # Step 2: Generate exactly 80 associations
    try:
        associations = generate_associations(goal)
    except Exception as e:
        raise ValueError(f"Failed to generate associations: {str(e)}") from e
    
    # Step 3: Map each association to 1-3 objects
    # HARD FAIL discipline: if any association cannot map, FAIL immediately
    object_pool = []
    for rank, association in enumerate(associations, start=1):
        try:
            objects = map_association_to_objects(association, rank)
            object_pool.extend(objects)
        except Exception as e:
            # HARD FAIL: do not continue, raise immediately
            raise ValueError(f"Failed to map association '{association}' (rank {rank}) to objects: {str(e)}") from e
    
    if len(object_pool) == 0:
        raise ValueError("Failed to map any associations to objects")
    
    # Step 4: Apply hard sanitation filters
    try:
        sanitized_pool = sanitize_candidate_pool(object_pool)
    except ValueError as e:
        raise ValueError(f"Sanitized object pool is empty: {str(e)}") from e
    
    # Step 5: Validate sanitized pool is not empty
    if len(sanitized_pool) == 0:
        raise ValueError("No valid object candidates after sanitation")
    
    return sanitized_pool


# ============================================================================
# STEP 4: CANDIDATE PAIR GENERATION (A/B pairs from sanitized pool)
# ============================================================================
# Generate deterministic candidate pairs without using geometry as ranking.
# ============================================================================

def build_ranked_object_list(objs: List[ObjectCandidate]) -> List[ObjectCandidate]:
    """
    Build ranked object list sorted by association_rank.
    
    Rules:
    - Sort by association_rank ascending (1 = strongest)
    - Stable order (deterministic)
    
    Args:
        objs: List of ObjectCandidate objects
    
    Returns:
        Sorted list of ObjectCandidate objects (by association_rank ascending)
    """
    # Sort by association_rank ascending (1 = strongest, 80 = weakest)
    # Use stable sort to maintain deterministic order for objects with same rank
    return sorted(objs, key=lambda obj: obj.association_rank)


def generate_candidate_pairs(objs: List[ObjectCandidate], max_pairs: int = 300) -> List[Tuple[ObjectCandidate, ObjectCandidate]]:
    """
    Generate candidate A/B pairs deterministically without using geometry as ranking.
    
    Rules:
    - Pairing must be deterministic
    - Pair candidates primarily from higher-ranked objects first
    - Do NOT use any geometric overlap calculations
    - Avoid trivial duplicates:
      - No (A,B) and (B,A) duplication
      - Do not pair an object with itself
    - Keep the list size bounded (max_pairs)
    
    Strategy:
    - Take the top N objects (N=40) by association_rank
    - Create pairs in nested order i<j until max_pairs reached
    
    Args:
        objs: List of ObjectCandidate objects (should be ranked)
        max_pairs: Maximum number of pairs to generate (default 300)
    
    Returns:
        List of (object_a, object_b) tuples, deterministically ordered
    """
    if len(objs) < 2:
        return []
    
    # Take top N objects (N=40) by association_rank for pairing
    # This ensures we prioritize higher-ranked associations
    top_n = min(40, len(objs))
    top_objects = objs[:top_n]
    
    # Generate pairs in nested order: i < j to avoid (A,B) and (B,A) duplicates
    # Also ensures no object pairs with itself
    pairs = []
    for i in range(len(top_objects)):
        for j in range(i + 1, len(top_objects)):
            if len(pairs) >= max_pairs:
                break
            pairs.append((top_objects[i], top_objects[j]))
        if len(pairs) >= max_pairs:
            break
    
    return pairs


# Global threshold constants (HARD GATES)
GEOMETRIC_OVERLAP_THRESHOLD_HYBRID = 0.70  # 70% - minimum for HYBRID eligibility
GEOMETRIC_OVERLAP_THRESHOLD_SIDE_BY_SIDE = 0.50  # 50% - below this, forced SIDE-BY-SIDE


def calculate_geometric_overlap(object_a_silhouette: Any, object_b_silhouette: Any) -> float:
    """
    Calculate geometric overlap percentage between two object silhouettes.
    
    Deterministic implementation based on shape_family compatibility:
    - Same shape family -> 0.80 (high overlap)
    - Compatible families (circle~oval, rectangle~square, bottle~cylinder, leaf~teardrop, blade~bolt) -> 0.55 (medium)
    - Different families -> 0.40 (low)
    - Unknown shape -> 0.40 (low)
    
    Args:
        object_a_silhouette: ObjectCandidate A (or its shape_family)
        object_b_silhouette: ObjectCandidate B (or its shape_family)
    
    Returns:
        Overlap percentage (0.0 to 1.0)
    """
    # Extract shape_family from objects
    if hasattr(object_a_silhouette, 'shape_family'):
        shape_a = object_a_silhouette.shape_family
    else:
        shape_a = str(object_a_silhouette) if isinstance(object_a_silhouette, str) else "unknown"
    
    if hasattr(object_b_silhouette, 'shape_family'):
        shape_b = object_b_silhouette.shape_family
    else:
        shape_b = str(object_b_silhouette) if isinstance(object_b_silhouette, str) else "unknown"
    
    # If either is unknown, return low overlap
    if shape_a == "unknown" or shape_b == "unknown":
        return 0.40
    
    # Same shape family -> high overlap
    if shape_a == shape_b:
        return 0.80
    
    # Compatible families (deterministic compatibility table)
    compatible_pairs = [
        ("circle", "oval"),
        ("oval", "circle"),
        ("rectangle", "square"),
        ("square", "rectangle"),
        ("bottle", "cylinder"),
        ("cylinder", "bottle"),
        ("leaf", "teardrop"),
        ("teardrop", "leaf"),
        ("blade", "bolt"),
        ("bolt", "blade")
    ]
    
    if (shape_a, shape_b) in compatible_pairs:
        return 0.55
    
    # Different families -> low overlap
    return 0.40


def select_max_projection_view(object: Any) -> Any:
    """
    Select the view with maximum projected silhouette area for an object.
    
    Deterministic implementation: returns the object itself (or its shape_family)
    since shape_family already represents the dominant silhouette characteristic.
    
    Args:
        object: ObjectCandidate to evaluate
    
    Returns:
        The object itself (for overlap calculation using shape_family)
    """
    # Return object as-is (shape_family is already determined from max-projection characteristics)
    return object


def evaluate_geometric_overlap_threshold(overlap_percentage: float) -> Tuple[bool, bool]:
    """
    Evaluate if geometric overlap meets HYBRID eligibility threshold.
    
    HARD GATE: All conditions must be met for HYBRID to be allowed.
    
    Args:
        overlap_percentage: Calculated overlap (0.0 to 1.0)
    
    Returns:
        Tuple of (hybrid_allowed, side_by_side_forced):
        - hybrid_allowed: True if overlap >= 70% (HYBRID eligible)
        - side_by_side_forced: True if overlap < 70% (HYBRID forbidden)
    """
    if overlap_percentage >= GEOMETRIC_OVERLAP_THRESHOLD_HYBRID:
        # Overlap >= 70%: HYBRID eligible (subject to other conditions)
        return (True, False)
    else:
        # Overlap < 70%: HYBRID strictly forbidden
        return (False, True)


def classify_hybrid_type(
    overlap_percentage: float,
    similarity_basis: str,
    requires_material_transformation: bool = False,
    requires_structural_similarity: bool = False,
    requires_micro_structure: bool = False
) -> HybridType:
    """
    Classify the type of hybrid based on similarity basis.
    
    Classification rules:
    - CORE Geometric Hybrid: outer silhouette overlap >= 70%, no material/structure logic
    - Material Analogy: requires material transformation
    - Structural Morphology: biological/architectural structure similarity
    - Structural Pattern: micro-structure/repeated units
    
    Args:
        overlap_percentage: Geometric overlap (0.0 to 1.0)
        similarity_basis: "geometric", "material", "structural", "pattern"
        requires_material_transformation: True if material analogy is needed
        requires_structural_similarity: True if structural morphology is needed
        requires_micro_structure: True if structural pattern exception is needed
    
    Returns:
        HybridType classification
    """
    # If overlap < 70%, HYBRID is forbidden regardless of similarity basis
    if overlap_percentage < GEOMETRIC_OVERLAP_THRESHOLD_HYBRID:
        return HybridType.SIDE_BY_SIDE
    
    # Check if similarity relies on material/structure (Exception types)
    if requires_material_transformation:
        return HybridType.MATERIAL_ANALOGY
    
    if requires_structural_similarity:
        return HybridType.STRUCTURAL_MORPHOLOGY
    
    if requires_micro_structure:
        return HybridType.STRUCTURAL_PATTERN
    
    # If overlap >= 70% and similarity is purely geometric (outer silhouette)
    if similarity_basis == "geometric" and overlap_percentage >= GEOMETRIC_OVERLAP_THRESHOLD_HYBRID:
        return HybridType.CORE_GEOMETRIC
    
    # Default: side-by-side if classification unclear
    return HybridType.SIDE_BY_SIDE


def check_quota_availability(hybrid_type: HybridType, quota_state: BatchQuotaState) -> bool:
    """
    Check if a hybrid type is allowed based on batch quotas.
    
    HARD QUOTAS (1 per 3-ad batch):
    - Material Analogy: 1 max
    - Structural Morphology: 1 max
    - Structural Pattern Exception: 1 max
    - CORE Geometric: unlimited
    
    Args:
        hybrid_type: Classification of the hybrid
        quota_state: Current quota usage state
    
    Returns:
        True if quota allows this hybrid type, False otherwise
    """
    if hybrid_type == HybridType.CORE_GEOMETRIC:
        # CORE geometric hybrids are unlimited
        return True
    
    if hybrid_type == HybridType.MATERIAL_ANALOGY:
        return quota_state.can_use_material_analogy()
    
    if hybrid_type == HybridType.STRUCTURAL_MORPHOLOGY:
        return quota_state.can_use_structural_morphology()
    
    if hybrid_type == HybridType.STRUCTURAL_PATTERN:
        return quota_state.can_use_structural_exception()
    
    # SIDE_BY_SIDE doesn't use quotas
    if hybrid_type == HybridType.SIDE_BY_SIDE:
        return True
    
    return False


def evaluate_ab_pair(
    object_a: Any,
    object_b: Any,
    quota_state: BatchQuotaState,
    similarity_basis: str = "geometric",
    requires_material_transformation: bool = False,
    requires_structural_similarity: bool = False,
    requires_micro_structure: bool = False
) -> Tuple[bool, HybridType, Optional[str]]:
    """
    Evaluate an A/B pair against all core decision rules.
    
    This is the main decision function that applies all canonical rules:
    1. Select max-projection views
    2. Calculate geometric overlap
    3. Evaluate overlap threshold (70% hard gate)
    4. Classify hybrid type
    5. Check quota availability
    
    Args:
        object_a: First object candidate
        object_b: Second object candidate
        quota_state: Batch quota state (for exception quotas)
        similarity_basis: "geometric", "material", "structural", "pattern"
        requires_material_transformation: True if material analogy needed
        requires_structural_similarity: True if structural morphology needed
        requires_micro_structure: True if structural pattern needed
    
    Returns:
        Tuple of (is_valid, hybrid_type, error_message):
        - is_valid: True if pair passes all rules
        - hybrid_type: Classification if valid
        - error_message: None if valid, error description if invalid
    """
    # Step 1: Select max-projection views (maximum silhouette area)
    view_a = select_max_projection_view(object_a)
    view_b = select_max_projection_view(object_b)
    
    # Step 2: Calculate geometric overlap (deterministic based on shape_family)
    overlap_percentage = calculate_geometric_overlap(view_a, view_b)
    
    # Step 3: Evaluate geometric overlap threshold (HARD GATE)
    hybrid_allowed, side_by_side_forced = evaluate_geometric_overlap_threshold(overlap_percentage)
    
    # Step 4: If overlap < 70%, HYBRID is strictly forbidden
    if not hybrid_allowed:
        # HYBRID forbidden, must use SIDE-BY-SIDE
        return (True, HybridType.SIDE_BY_SIDE, None)
    
    # Step 5: Classify hybrid type
    hybrid_type = classify_hybrid_type(
        overlap_percentage,
        similarity_basis,
        requires_material_transformation,
        requires_structural_similarity,
        requires_micro_structure
    )
    
    # Step 6: Check quota availability (for Exception types)
    if not check_quota_availability(hybrid_type, quota_state):
        # Quota exhausted for this exception type
        if hybrid_type in [HybridType.MATERIAL_ANALOGY, HybridType.STRUCTURAL_MORPHOLOGY, HybridType.STRUCTURAL_PATTERN]:
            return (False, hybrid_type, f"Quota exhausted for {hybrid_type.value} (1 per 3-ad batch limit)")
        # If CORE geometric, should never fail quota check
        return (False, hybrid_type, "Unexpected quota check failure")
    
    # Step 7: Mark quota as used if this is an exception type
    if hybrid_type == HybridType.MATERIAL_ANALOGY:
        quota_state.use_material_analogy()
    elif hybrid_type == HybridType.STRUCTURAL_MORPHOLOGY:
        quota_state.use_structural_morphology()
    elif hybrid_type == HybridType.STRUCTURAL_PATTERN:
        quota_state.use_structural_exception()
    # CORE_GEOMETRIC and SIDE_BY_SIDE don't use quotas
    
    # Pair passes all rules
    return (True, hybrid_type, None)


def find_valid_ab_pair(
    candidate_pairs: List[Tuple[ObjectCandidate, ObjectCandidate]],
    quota_state: BatchQuotaState,
    ranked_associations: List[ObjectCandidate]
) -> Tuple[Optional[ObjectCandidate], Optional[ObjectCandidate], HybridType, Optional[str]]:
    """
    Find the first valid A/B pair from candidates that passes all rules.
    
    Candidate Resolution Principle:
    - Geometry is a pass/fail gate, not a ranking tool
    - If multiple candidates pass geometry, higher-ranked association wins
    - NEVER use geometric overlap to choose between candidates
    
    Args:
        candidate_pairs: List of (object_a, object_b) candidate pairs (already sanitized)
        quota_state: Batch quota state
        ranked_associations: Ranked list of associations (size 80) for resolution
    
    Returns:
        Tuple of (object_a, object_b, hybrid_type, error_message):
        - object_a, object_b: Selected pair if valid, None if none found
        - hybrid_type: Classification of the valid pair
        - error_message: None if valid pair found, error if none found
    """
    # Evaluate candidates in ranked order (geometry is pass/fail gate)
    # Note: All candidates in candidate_pairs have already passed hard sanitation filters
    for object_a, object_b in candidate_pairs:
        # For now, assume geometric similarity (will be determined from actual data)
        is_valid, hybrid_type, error_msg = evaluate_ab_pair(
            object_a,
            object_b,
            quota_state,
            similarity_basis="geometric"  # Will be determined from actual analysis
        )
        
        if is_valid:
            # First valid pair found (highest ranked that passes)
            return (object_a, object_b, hybrid_type, None)
    
    # No valid configuration found - FAIL generation
    return (None, None, HybridType.SIDE_BY_SIDE, "No valid A/B pair found that passes all core decision rules")


def generate_ad(
    product_name: str,
    product_description: str,
    size: str,
    quota_state: Optional[BatchQuotaState] = None,
    ad_index: int = 0
) -> Dict[str, Any]:
    """
    Generate a single ad based on product information.
    
    STEP 2: Core Decision Logic integrated.
    The decision logic is ready but requires actual object candidates to evaluate.
    Currently returns placeholder until object generation is implemented.
    
    Args:
        product_name: Name of the product (non-empty string)
        product_description: Description of the product (non-empty string)
        size: Image size, must be one of: "1024x1024", "1536x1024", "1024x1536"
        quota_state: Optional batch quota state (for 3-ad batch quotas)
        ad_index: Index of ad in batch (0, 1, or 2) for variation in marketing text
    
    Returns:
        Dictionary with:
        - "image_bytes_jpg": bytes - JPEG image data
        - "marketing_text": str - Marketing text content
    
    Raises:
        ValueError: If generation fails (will be caught by route handler)
        Exception: For any other engine errors
    """
    # Initialize quota state if not provided (for single ad generation)
    if quota_state is None:
        quota_state = BatchQuotaState()
    
    # ========================================================================
    # STEP 3: GOAL → 80 ASSOCIATIONS → OBJECT POOL
    # ========================================================================
    # Build object pool from product information
    # Goal is hidden (not returned to frontend)
    # Associations are internal only (exactly 80)
    # Only physical, non-symbolic objects enter the pool
    try:
        sanitized_pool = build_object_pool(product_name, product_description)
    except ValueError as e:
        # No valid object pool - FAIL explicitly
        raise ValueError(f"Failed to build object pool: {str(e)}") from e
    
    # ========================================================================
    # STEP 4: CANDIDATE PAIR GENERATION (A/B pairs from sanitized pool)
    # ========================================================================
    # Build ranked object list and generate candidate pairs
    # Pairing is deterministic, prioritizes higher-ranked objects
    # No geometry calculations used for ranking
    ranked_objs = build_ranked_object_list(sanitized_pool)
    candidate_pairs = generate_candidate_pairs(ranked_objs, max_pairs=300)
    
    if len(candidate_pairs) == 0:
        raise ValueError("No candidate pairs generated from object pool")
    
    # ========================================================================
    # STEP 5: Find valid A/B pair using core decision logic
    # ========================================================================
    # Try to find valid pair using geometry and quota rules
    # If geometry is not implemented yet, use temporary bridge
    object_a, object_b, hybrid_type, error_msg = find_valid_ab_pair(
        candidate_pairs,
        quota_state,
        ranked_objs
    )
    
    # If no valid pair found after evaluation, fail explicitly
    if object_a is None or object_b is None:
        raise ValueError(f"No valid A/B pair found: {error_msg}")
    
    # ========================================================================
    # STEP 6 & 7: GOAL DERIVATION (once) + HEADLINE & MARKETING TEXT GENERATION
    # ========================================================================
    # Derive goal once and use for both headline and marketing text
    goal = derive_advertising_goal(product_name, product_description)
    headline = generate_headline(product_name, goal, ad_index)
    marketing_text = generate_marketing_text(product_name, product_description, goal, ad_index)
    
    # ========================================================================
    # STEP 8: IMAGE GENERATION (Placeholder implementation)
    # ========================================================================
    # Parse size
    try:
        width, height = map(int, size.split('x'))
    except (ValueError, AttributeError):
        raise ValueError(f"Invalid size format: {size}")
    
    # Validate size is in allowed list
    allowed_sizes = ["1024x1024", "1536x1024", "1024x1536"]
    if size not in allowed_sizes:
        raise ValueError(f"Size must be one of: {', '.join(allowed_sizes)}")
    
    # Create placeholder image with Pillow
    # White background
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, fallback to default
    try:
        # Try common font paths
        font_large = ImageFont.truetype("arial.ttf", 72)
        font_small = ImageFont.truetype("arial.ttf", 48)
    except (OSError, IOError):
        try:
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 72)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 48)
        except (OSError, IOError):
            # Fallback to default font
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
    
    # ========================================================================
    # STEP 8: SILHOUETTE PLACEHOLDER GENERATION
    # ========================================================================
    # Draw headline at top center with safe margins
    headline_bbox = draw.textbbox((0, 0), headline, font=font_large)
    headline_width = headline_bbox[2] - headline_bbox[0]
    headline_x = (width - headline_width) // 2
    headline_y = height // 12  # Top area with padding, leaving space for silhouette below
    draw.text((headline_x, headline_y), headline, fill='black', font=font_large)
    
    # Draw silhouette placeholder in center area (reflects HYBRID or SIDE_BY_SIDE)
    draw_silhouette_placeholder(
        draw,
        width,
        height,
        object_a.shape_family,
        object_b.shape_family,
        hybrid_type
    )
    
    # Convert to JPEG bytes
    jpeg_buffer = io.BytesIO()
    img.save(jpeg_buffer, format='JPEG', quality=95)
    image_bytes_jpg = jpeg_buffer.getvalue()
    jpeg_buffer.close()
    
    # Return result with updated batch state
    return {
        "image_bytes_jpg": image_bytes_jpg,
        "marketing_text": marketing_text,
        "batch_state": quota_state_to_dict(quota_state)
    }

