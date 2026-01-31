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
import os
import base64
import logging
import random
from enum import Enum
from dataclasses import dataclass
from collections import Counter
from openai import OpenAI

# Logger for engine operations
logger = logging.getLogger(__name__)


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
    - association_key: Original association string from which this object was derived
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
    association_key: str = ""


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
# STEP 8: REAL IMAGE GENERATION (OpenAI GPT Image)
# ============================================================================
# Generate real photorealistic ad images via OpenAI Images API.
# Headline is added via PIL overlay after image generation.
# ============================================================================

def derive_environment_for_object(obj_name: str) -> str:
    """
    Derive minimal environment for an object (deterministic mapping).
    
    Object Environment Rule:
    - Maps common objects to minimal environments
    - Returns "neutral studio surface" if no match found
    - Environment must not include text, signage, symbols, or narrative contexts
    
    Args:
        obj_name: Name of the object
    
    Returns:
        Environment string (minimal, non-narrative)
    """
    obj_lower = obj_name.lower()
    
    # Water/liquid objects
    if any(kw in obj_lower for kw in ["water", "dew", "rain", "snow", "ice", "frost", "mist", "fog", "ocean", "sea", "lake", "river", "stream", "pond", "pool", "fountain", "spring"]):
        return "water surface"
    
    # Bedding/comfort objects
    if any(kw in obj_lower for kw in ["pillow", "cushion", "mattress", "blanket", "quilt", "comforter", "duvet", "sheet", "bed", "hammock"]):
        return "bedding"
    
    # Desk/table objects
    if any(kw in obj_lower for kw in ["desk", "table", "workbench", "bench", "counter", "surface"]):
        return "desk"
    
    # Tool/mechanical objects
    if any(kw in obj_lower for kw in ["tool", "gear", "cog", "wheel", "pulley", "lever", "mechanism", "machine", "engine", "motor", "turbine", "generator", "wrench", "screwdriver", "hammer"]):
        return "tabletop"
    
    # Writing/paper objects
    if any(kw in obj_lower for kw in ["book", "notebook", "journal", "paper", "document", "letter", "pen", "pencil"]):
        return "desk"
    
    # Food objects
    if any(kw in obj_lower for kw in ["fruit", "vegetable", "herb", "spice", "grain", "seed", "nut", "berry", "food"]):
        return "tabletop"
    
    # Nature/plant objects
    if any(kw in obj_lower for kw in ["leaf", "flower", "bud", "sprout", "seedling", "sapling", "blossom", "bloom", "tree", "plant"]):
        return "tabletop"
    
    # Glass/crystal objects
    if any(kw in obj_lower for kw in ["glass", "crystal", "lens", "mirror", "prism", "diamond", "gem", "pearl"]):
        return "tabletop"
    
    # Metal/stone objects
    if any(kw in obj_lower for kw in ["steel", "iron", "metal", "stone", "rock", "brick", "marble", "granite", "diamond", "titanium"]):
        return "tabletop"
    
    # Fabric/textile objects
    if any(kw in obj_lower for kw in ["fabric", "cloth", "textile", "robe", "pajamas", "sweater", "cardigan", "hoodie", "jacket", "socks", "slippers"]):
        return "bedding"
    
    # Default: neutral studio surface
    return "neutral studio surface"


def build_pre_intent_block(product_name: str, product_description: str) -> str:
    """
    Build PRE-INTENT block from product information (deterministic text, no API calls).
    
    Args:
        product_name: Name of the product
        product_description: Description of the product
    
    Returns:
        PRE-INTENT text block (3-5 lines max)
    """
    # Combine product info for analysis
    combined_text = (product_name + " " + product_description).lower()
    
    # CORE IDEA: Extract main concept from product name/description
    # Simple keyword-based extraction (deterministic)
    core_idea = product_name
    if len(product_description) > 0:
        # Use first meaningful phrase from description
        desc_words = product_description.split()[:10]  # First 10 words
        core_idea = f"{product_name} ({' '.join(desc_words)})"
    
    # CREATIVE TENSION: Identify one contrast (deterministic keyword matching)
    tension = "innovation meets tradition"
    if any(word in combined_text for word in ["fast", "speed", "quick", "rapid", "instant"]):
        tension = "speed meets precision"
    elif any(word in combined_text for word in ["safe", "secure", "protection", "shield"]):
        tension = "strength meets elegance"
    elif any(word in combined_text for word in ["fresh", "new", "clean", "pure"]):
        tension = "purity meets power"
    elif any(word in combined_text for word in ["comfort", "soft", "gentle", "cozy"]):
        tension = "comfort meets durability"
    elif any(word in combined_text for word in ["precise", "accurate", "exact", "detailed"]):
        tension = "precision meets simplicity"
    else:
        tension = "form meets function"
    
    # VISUAL DIRECTION: What kind of visual logic we want
    visual_direction = "clean, focused composition with strong silhouette definition"
    if any(word in combined_text for word in ["hybrid", "merge", "combine", "fusion"]):
        visual_direction = "seamless integration where boundaries dissolve into unified form"
    elif any(word in combined_text for word in ["precise", "accurate", "exact"]):
        visual_direction = "crisp, defined edges with clear geometric relationships"
    elif any(word in combined_text for word in ["soft", "comfort", "gentle"]):
        visual_direction = "organic flow with smooth transitions and natural curves"
    
    # ENVIRONMENT DIRECTION: Physical context cue (not decorative)
    environment_direction = "realistic physical context that functionally justifies the composition"
    if any(word in combined_text for word in ["desk", "office", "work", "professional"]):
        environment_direction = "workspace or professional setting that explains functional purpose"
    elif any(word in combined_text for word in ["home", "domestic", "household", "living"]):
        environment_direction = "domestic environment that shows natural integration"
    elif any(word in combined_text for word in ["outdoor", "nature", "natural", "environment"]):
        environment_direction = "natural outdoor context that supports functional logic"
    elif any(word in combined_text for word in ["water", "liquid", "fluid", "aquatic"]):
        environment_direction = "aquatic or fluid environment that explains material interaction"
    
    # Build PRE-INTENT block
    pre_intent = (
        f"PRE-INTENT (interpretation before rendering):\n"
        f"CORE IDEA: {core_idea}\n"
        f"TENSION: {tension}\n"
        f"VISUAL DIRECTION: {visual_direction}\n"
        f"ENVIRONMENT DIRECTION: {environment_direction}"
    )
    
    return pre_intent


def generate_real_image_bytes(
    product_name: str,
    product_description: str,
    size: str,
    object_a: ObjectCandidate,
    object_b: ObjectCandidate,
    hybrid_type: HybridType,
    goal: str
) -> bytes:
    """
    Generate real photorealistic ad image via OpenAI Images API.
    
    Args:
        product_name: Name of the product
        product_description: Description of the product
        size: Image size ("1024x1024", "1536x1024", or "1024x1536")
        object_a: ObjectCandidate for object A
        object_b: ObjectCandidate for object B
        hybrid_type: HybridType (CORE_GEOMETRIC or SIDE_BY_SIDE)
        goal: Advertising goal (hidden, for context)
    
    Returns:
        bytes: JPEG image data
    
    Raises:
        ValueError: If OpenAI API call fails or returns invalid data
    """
    # Get OpenAI API key from environment
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # ENV_FIXED=A: Object A is always the environment
    # Object B is embedded inside A (ENV_EMBED composition mode)
    environment = derive_environment_for_object(object_a.name)
    
    # Build PRE-INTENT block (deterministic, no API calls)
    pre_intent = build_pre_intent_block(product_name, product_description)
    
    # Build photorealistic commercial ad prompt (ENV_FIXED=A, ENV_EMBED composition)
    # ENVIRONMENT LAW — MANDATORY: Environment always overrides aesthetics.
    # SIDE_BY_SIDE is permanently disabled - only HYBRID prompts are generated
    # ENV_EMBED: Object A is the environment/domain (setting, surface, context) ONLY, not a physical object
    # Object B is the only visible subject/object
    prompt = (
            f"{pre_intent}\n\n"
            f"Do not write any of the PRE-INTENT as text in the image.\n\n"
            f"Photorealistic commercial advertising photograph. "
            f"COMPOSITION MODE: ENV_EMBED. "
            f"Show {object_b.name} (Object B) as the only subject, emerging naturally within {object_a.name}'s (Object A) domain environment. "
            f"Object A ({object_a.name}) is the environment/domain setting (surface, context, materials) and must not appear as an object. "
            f"Object B ({object_b.name}) is the only visible object/subject. "
            f"IMPORTANT — STRICT PROHIBITIONS: "
            f"Do NOT depict Object A as a physical object, frame, ring, shell, casing, or outline around B. "
            f"A must be present only as the environment/domain (setting, surface, materials, context) — not as an object. "
            f"Do NOT nest B inside a larger version of A. "
            f"Do NOT wrap, encase, surround, or border B with remnants of A. "
            f"Only ONE object is allowed as a subject: B. A is background/context only. "
            f"Do NOT show two separate objects. "
            f"Do NOT show two objects merged, glued, split, or assembled. "
            f"Do NOT show a half-and-half object. "
            f"Do NOT show two objects in frame. "
            f"This is NOT two objects placed together. "
            f"This is NOT side by side. "
            f"This is NOT assembly or collage. "
            f"This is NOT overlay or composition. "
            f"No collage, no assembly, no glued parts, no two-object composition. "
            f"The composition must read as a SINGLE unified believable commercial photo where B naturally exists within A's environment. "
            f"ENVIRONMENT LAW — MANDATORY: "
            f"Every image MUST be placed in a realistic, coherent physical environment. "
            f"No white studio, no abstract background, no generic gradient, no empty void. "
            f"Environment authority belongs to Object A ({object_a.name}): the scene, surface, props, and context must come from {object_a.name}'s environment. "
            f"The environment must be clearly {object_a.name}'s domain (workspace / surface / setting / etc. depending on {object_a.name}), realistic and minimal. "
            f"Object B ({object_b.name}) must appear as if it naturally emerges from or exists within {object_a.name}'s environment. "
            f"The environment must logically explain why Object B exists in this context. "
            f"HYBRID RULES — ABSOLUTE LAWS: "
            f"Object B must appear as a natural outcome of Object A's environment. "
            f"NOT a glued object, NOT a merged sculpture, NOT an artificial mashup. "
            f"NOT a mix of X and Y. NOT a clever combination. NOT a logical pairing. "
            f"The form must feel INEVITABLE given the environment, not logical or clever. "
            f"If the hybrid can be described as 'a mix of {object_a.name} and {object_b.name}' — it is WRONG. "
            f"The hybrid must feel DISCOVERED, not designed. "
            f"The environment must visually justify WHY Object B exists in this form within Object A's world. "
            f"The environment must CAUSE the object's existence, not merely host it. "
            f"The environment must support the FUNCTIONAL MEANING of the composition. "
            f"The environment must support the composition as a SINGLE unified entity. "
            f"Do NOT show seams, joints, or construction logic. "
            f"Do NOT suggest mechanical assembly. "
            f"Do NOT suggest that two objects existed before the hybrid. "
            f"Object B must appear naturally formed within Object A's context. "
            f"The environment must NOT introduce additional objects that compete with or distract from the composition. "
            f"The environment must reinforce the SILHOUETTE: background contrast must clearly separate the outline, no busy textures behind the silhouette, no visual noise intersecting the contour. "
            f"The environment must make the composition feel INEVITABLE, not accidental, not assembled, not improvised, not clever. "
            f"Object B must appear as if it was BORN FROM Object A's environment, not placed into it afterward. "
            f"The connection must feel DISCOVERED, not designed. "
            f"Environment: {environment} (realistic physical context from {object_a.name}'s world where {object_b.name} would naturally exist embedded within). "
            f"CAMERA & SPATIAL RULES: "
            f"Straight frontal camera angle only. "
            f"The environment plane must be perpendicular to the viewer. "
            f"No perspective skew, no diagonal surfaces, no dramatic depth of field. "
            f"Horizon line stable and horizontal. "
            f"MATERIAL & LIGHTING RULES: "
            f"Realistic material interaction between Object B and Object A's environment. "
            f"Soft, neutral, commercial lighting. "
            f"No harsh shadows, no silhouette blackouts. "
            f"Object B must retain visible surface detail. "
            f"Composition: Object B centered within Object A's environment, empty clean space at TOP (minimum 15%) for headline. "
            f"No text, no logos, no labels, no symbols anywhere in the image. "
            f"ABSOLUTE PROHIBITIONS: "
            f"No abstract environments, no studio voids unless physically justified, no floating objects, no symbolic or metaphorical backgrounds, no surreal or conceptual scenery. "
            f"Style: Professional product photography, real materials, real textures. "
            f"No illustration, no CGI, no 3D render, no vector look. "
            f"Environment always overrides aesthetics. If environment logic fails, the image is invalid."
        )
    
    # Use gpt-image-1.5 model (NO FALLBACK - fail if it doesn't work)
    model = "gpt-image-1.5"
    logger.info(f"HYBRID_ENV_MODE ENV_FIXED=A COMPOSITION_MODE=ENV_EMBED SUBJECT=B A_ENV_ONLY=1 A={object_a.name} B={object_b.name} env={environment} size={size} model={model}")
    logger.info(f"IMAGE_GEN_START model={model} size={size} object_a={object_a.name} object_b={object_b.name} hybrid_type={hybrid_type.value}")
    
    try:
        response = client.images.generate(
            model=model,
            prompt=prompt,
            size=size,
            quality="medium"
        )
    except Exception as e:
        # NO FALLBACK - fail immediately with clear error
        logger.error(f"IMAGE_GEN_FAIL error={str(e)}")
        raise ValueError(f"OpenAI image generation failed: {str(e)}") from e
    
    # Extract image data
    if not response.data or len(response.data) == 0:
        logger.error("IMAGE_GEN_FAIL error=OpenAI returned empty image data")
        raise ValueError("OpenAI returned empty image data")
    
    # Try to get base64 from response (if available)
    image_b64 = getattr(response.data[0], 'b64_json', None)
    if image_b64:
        # Decode base64 to bytes
        try:
            image_bytes = base64.b64decode(image_b64)
            logger.info(f"IMAGE_GEN_OK bytes={len(image_bytes)}")
            return image_bytes
        except Exception as e:
            logger.error(f"IMAGE_GEN_FAIL error=Failed to decode base64: {str(e)}")
            raise ValueError(f"Failed to decode base64 image: {str(e)}") from e
    
    # If no base64, try to get URL and download
    image_url = getattr(response.data[0], 'url', None)
    if image_url:
        import requests
        try:
            img_response = requests.get(image_url)
            img_response.raise_for_status()
            image_bytes = img_response.content
            logger.info(f"IMAGE_GEN_OK bytes={len(image_bytes)} (from URL)")
            return image_bytes
        except Exception as e:
            logger.error(f"IMAGE_GEN_FAIL error=Failed to download from URL: {str(e)}")
            raise ValueError(f"Failed to download image from URL: {str(e)}") from e
    
    logger.error("IMAGE_GEN_FAIL error=OpenAI returned invalid image data (no base64 or URL)")
    raise ValueError("OpenAI returned invalid image data (no base64 or URL)")


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
        "armor", "shield", "barrier", "rampart", "bulwark", "anvil", "workbench", "steel cable",
        "chain link", "truck tire", "welding seam", "reinforced hinge", "protective case", "hardhat", "steel rivet", "carbon fiber sheet",
        "rebar", "toolbox", "shock absorber", "industrial clamp", "steel plate", "concrete pillar", "iron girder", "steel anchor",
        "reinforced concrete", "steel frame", "iron chain", "steel bolt", "welding joint", "steel bracket", "heavy-duty latch", "steel rod"
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
        "pipeline", "conduit", "channel", "duct", "tube", "pipe", "hose", "valve",
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
        "familiarity", "home", "hearth", "fireplace", "carpet", "light", "glow", "radiance",
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
        "calibration", "adjustment", "alignment", "positioning", "placement", "location", "dial indicator", "setting",
        "accuracy", "exactness", "correctness", "rightness", "truth", "veracity", "validity", "authenticity"
    ]
}


def validate_associations_library():
    """
    Validate ASSOCIATIONS_LIBRARY for correctness at startup.
    
    Checks for each goal:
    - Exactly 80 items
    - No duplicates (exact string match)
    
    Raises:
        ValueError: If any goal has incorrect count or duplicates, with detailed report
    """
    all_issues = []
    
    for goal, associations in ASSOCIATIONS_LIBRARY.items():
        issues = []
        
        # Check total count
        total_items = len(associations)
        if total_items != 80:
            issues.append(f"  total_items: {total_items} (expected 80)")
        
        # Check for duplicates (exact string match, case-sensitive)
        item_counts = Counter(associations)
        duplicates = {item: count for item, count in item_counts.items() if count > 1}
        
        if duplicates:
            dup_list = [f"    '{item}' -> {count}" for item, count in sorted(duplicates.items())]
            issues.append(f"  duplicates:\n" + "\n".join(dup_list))
        
        # If issues found, add to report
        if issues:
            all_issues.append({
                'goal': goal,
                'total_items': total_items,
                'duplicates': duplicates,
                'issues': issues
            })
    
    # If any issues found, build report and raise exception
    if all_issues:
        report_lines = ["ASSOCIATIONS_LIBRARY validation failed:"]
        for issue_info in all_issues:
            report_lines.append(f"  goal: {issue_info['goal']}")
            for issue_line in issue_info['issues']:
                report_lines.append(issue_line)
            report_lines.append("")  # Empty line between goals
        
        report = "\n".join(report_lines)
        logger.error(report)
        
        # Build error message for exception
        error_parts = []
        for issue_info in all_issues:
            goal = issue_info['goal']
            total = issue_info['total_items']
            dup_count = len(issue_info['duplicates'])
            
            parts = [f"{goal}: "]
            if total != 80:
                parts.append(f"count={total} (expected 80)")
            if dup_count > 0:
                if total != 80:
                    parts.append(", ")
                parts.append(f"{dup_count} duplicate(s)")
            error_parts.append("".join(parts))
        
        error_msg = "ASSOCIATIONS_LIBRARY validation failed for: " + "; ".join(error_parts)
        raise ValueError(error_msg)


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
        ValueError: (removed) If no keywords match, deterministic fallback selects goal from ALLOWED_GOALS
    """
    # Combine product name and description for keyword matching
    text = (product_name + " " + product_description).lower()
    
    # Count keyword matches for each goal
    goal_scores = {}
    for goal, keywords in GOAL_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in text)
        if score > 0:
            goal_scores[goal] = score
    
    # If no matches found, use deterministic fallback to prevent hard-fail
    if not goal_scores:
        # Deterministic fallback: hash product_name + product_description to select goal
        # This ensures consistent goal selection even when no keywords match
        fallback_text = product_name + " " + product_description
        fallback_hash = hashlib.sha256(fallback_text.encode('utf-8')).hexdigest()
        fallback_int = int(fallback_hash[:8], 16)  # Use first 8 hex chars as integer
        fallback_goal_idx = fallback_int % len(ALLOWED_GOALS)
        return ALLOWED_GOALS[fallback_goal_idx]
    
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


def build_goals_for_batch(product_name: str, product_description: str, session_seed: Optional[str] = None) -> List[str]:
    """
    Build a list of 3 different advertising goals for a 3-ad batch.
    
    SESSION DIVERSITY RULE:
    - Goals MUST vary across sessions for the same product
    - session_seed ensures each session gets different goals
    - Even if product strongly suggests one goal, it MUST NOT appear in every session
    
    Rules:
    - All 3 goals must be different from each other
    - Selection is deterministic based on hash of (productName + productDescription + session_seed + "goal" + index)
    - If session_seed is None, use product hash as fallback (still varies by session context)
    
    Args:
        product_name: Name of the product
        product_description: Description of the product
        session_seed: Optional session seed for diversity (if None, uses product hash)
    
    Returns:
        List of 3 goal strings from ALLOWED_GOALS (different across sessions)
    """
    # Step 1: Calculate session-aware primary goal
    # Include session_seed to ensure different primary goal per session
    if session_seed:
        # Use session_seed to rotate primary goal selection
        primary_hash_text = product_name + product_description + session_seed + "primary"
        primary_hash = hashlib.sha256(primary_hash_text.encode('utf-8')).hexdigest()
        primary_hash_int = int(primary_hash[:8], 16)
        
        # Get candidate goals (all goals that match product keywords, or all if none match)
        text = (product_name + " " + product_description).lower()
        goal_scores = {}
        for goal, keywords in GOAL_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                goal_scores[goal] = score
        
        # If no keyword matches, use all goals as candidates
        if not goal_scores:
            candidate_goals = list(ALLOWED_GOALS)
        else:
            # Use top-scoring goals as candidates (at least top 3, or all if < 3)
            max_score = max(goal_scores.values())
            candidate_goals = [goal for goal, score in goal_scores.items() if score == max_score]
            # If only 1-2 candidates, add more from high-scoring goals
            if len(candidate_goals) < 3:
                sorted_goals = sorted(goal_scores.items(), key=lambda x: x[1], reverse=True)
                for goal, score in sorted_goals:
                    if goal not in candidate_goals:
                        candidate_goals.append(goal)
                        if len(candidate_goals) >= 4:  # Allow some diversity
                            break
            # If still too few, add all goals
            if len(candidate_goals) < 3:
                candidate_goals = list(ALLOWED_GOALS)
        
        # Select primary goal from candidates using session hash
        primary_goal_idx = primary_hash_int % len(candidate_goals)
        primary_goal = candidate_goals[primary_goal_idx]
    else:
        # Fallback: use original derivation (still deterministic per product)
        primary_goal = derive_advertising_goal(product_name, product_description)
    
    # Step 2: Build list of available goals (excluding primary_goal)
    available_goals = [g for g in ALLOWED_GOALS if g != primary_goal]
    
    # Step 3: Select goals_for_batch[1] and goals_for_batch[2] deterministically with session_seed
    goals_for_batch = [primary_goal]
    
    # Use session_seed in hash to ensure different selection per session
    seed_suffix = session_seed if session_seed else product_name + product_description
    
    for index in [1, 2]:
        # Create deterministic hash for this index (includes session_seed)
        hash_text = product_name + product_description + seed_suffix + "goal" + str(index)
        hash_value = hashlib.sha256(hash_text.encode('utf-8')).hexdigest()
        hash_int = int(hash_value[:8], 16)  # Use first 8 hex chars as integer
        
        # Select goal from available goals (excluding already selected ones)
        goal_idx = hash_int % len(available_goals)
        selected_goal = available_goals[goal_idx]
        goals_for_batch.append(selected_goal)
        
        # Remove selected goal from available list to ensure uniqueness
        available_goals.remove(selected_goal)
    
    logger.info(f"GOAL_SELECTION product={product_name[:30]} session_seed={session_seed[:8] if session_seed else 'none'} goals={goals_for_batch}")
    
    return goals_for_batch


def map_association_to_world(association: str) -> str:
    """
    Map an association to its conceptual world/domain.
    
    Each association belongs to ONE world. Only ONE association per world is allowed.
    
    Worlds:
    - mechanical: gears, engines, machines, mechanisms
    - industrial: factories, tools, manufacturing, construction
    - biological: animals, plants, living organisms
    - geological: rocks, minerals, earth, natural formations
    - architectural: buildings, structures, construction elements
    - domestic: household items, furniture, home objects (MAX ONE)
    - natural: natural phenomena, weather, elements
    - scientific: instruments, lab equipment, measurement tools
    - human_body: physical body parts (physical only)
    - transportation: vehicles, movement devices
    - protective: safety equipment, barriers, shields
    - fluid: liquids, water, fluids, containers
    - optical: light, vision, transparency, lenses
    - abstract: concepts that don't fit other categories (minimize)
    
    Args:
        association: Association string
    
    Returns:
        World name string
    """
    assoc_lower = association.lower()
    
    # Mechanical world
    if any(kw in assoc_lower for kw in ["gear", "cog", "wheel", "pulley", "lever", "mechanism", "machine", "engine", "motor", "turbine", "rotor", "sprocket", "chain", "cable"]):
        return "mechanical"
    
    # Industrial world
    if any(kw in assoc_lower for kw in ["factory", "tool", "workshop", "construction", "steel", "concrete", "beam", "girder", "rivet", "welding", "clamp", "bracket", "anchor", "rebar"]):
        return "industrial"
    
    # Transportation world
    if any(kw in assoc_lower for kw in ["car", "vehicle", "train", "plane", "airplane", "helicopter", "boat", "ship", "submarine", "rocket", "satellite", "drone", "bicycle", "motorcycle", "skateboard", "roller"]):
        return "transportation"
    
    # Biological world
    if any(kw in assoc_lower for kw in ["animal", "bird", "fish", "cheetah", "falcon", "deer", "rabbit", "squirrel", "plant", "leaf", "flower", "tree", "seed", "sprout", "blossom", "fruit", "vegetable", "herb"]):
        return "biological"
    
    # Geological world
    if any(kw in assoc_lower for kw in ["rock", "stone", "boulder", "mountain", "cliff", "mineral", "crystal", "diamond", "marble", "granite", "quartz", "gem", "pearl", "coral"]):
        return "geological"
    
    # Architectural world
    if any(kw in assoc_lower for kw in ["building", "bridge", "dam", "tunnel", "arch", "column", "pillar", "wall", "fortress", "castle", "structure", "frame", "foundation"]):
        return "architectural"
    
    # Domestic world (MAX ONE allowed)
    if any(kw in assoc_lower for kw in ["pillow", "cushion", "mattress", "blanket", "quilt", "comforter", "duvet", "sheet", "sofa", "armchair", "recliner", "bed", "furniture", "home", "hearth", "fireplace", "carpet", "slippers", "robe", "pajamas"]):
        return "domestic"
    
    # Natural phenomena world
    if any(kw in assoc_lower for kw in ["wind", "rain", "snow", "ice", "frost", "mist", "fog", "dew", "storm", "hurricane", "tornado", "lightning", "thunder", "wave", "current", "stream", "river", "ocean", "sea", "lake", "waterfall", "cascade"]):
        return "natural"
    
    # Scientific/Measurement world
    if any(kw in assoc_lower for kw in ["compass", "ruler", "scale", "caliper", "micrometer", "gauge", "meter", "measure", "clock", "watch", "timer", "laser", "telescope", "microscope", "magnifying", "lens", "calibration", "dial indicator"]):
        return "scientific"
    
    # Human body world (physical only)
    if any(kw in assoc_lower for kw in ["eye", "pupil", "iris", "retina", "vision", "sight", "body", "hand", "finger", "muscle", "bone"]):
        return "human_body"
    
    # Protective world
    if any(kw in assoc_lower for kw in ["helmet", "shield", "armor", "barrier", "fence", "guardrail", "safety", "protection", "guard", "sentry", "lock", "key", "safe", "vault"]):
        return "protective"
    
    # Fluid/Container world
    if any(kw in assoc_lower for kw in ["water", "liquid", "fluid", "bottle", "jar", "can", "container", "vessel", "tube", "pipe", "hose", "valve", "fountain", "pool", "pond"]):
        return "fluid"
    
    # Optical world
    if any(kw in assoc_lower for kw in ["light", "beam", "ray", "shine", "glow", "illumination", "brightness", "transparency", "glass", "mirror", "prism", "kaleidoscope", "spectroscope", "binoculars", "periscope"]):
        return "optical"
    
    # Default: abstract (minimize these)
    return "abstract"


def filter_associations_for_diversity(associations: List[str], max_per_world: int = 1) -> List[str]:
    """
    Filter associations to ensure conceptual distance and world uniqueness.
    
    HARD LAW — CONCEPTUAL DISTANCE:
    - Only ONE association per world is allowed (by default)
    - If two associations could plausibly appear in the same photograph, share a supplier, store, room, or use case — reject one
    - Prefer 20-40 radically different associations over 80 conceptually neighboring ones
    - Distance is more important than completeness
    - Domestic world is penalized: if other worlds are available, prefer them over domestic
    
    Args:
        associations: List of association strings (ordered by strength)
        max_per_world: Maximum associations per world (default 1 for strict diversity)
    
    Returns:
        Filtered list of associations with maximum conceptual distance
    """
    world_to_associations = {}
    filtered = []
    domestic_count = 0
    
    for assoc in associations:
        world = map_association_to_world(assoc)
        
        # Track associations by world
        if world not in world_to_associations:
            world_to_associations[world] = []
        
        # HARD RULE: Domestic world gets extra penalty
        # If we already have domestic and other worlds are available, skip additional domestic
        if world == "domestic":
            if domestic_count >= max_per_world:
                # Skip if we already have domestic and there are other worlds available
                other_worlds_count = sum(1 for w, assocs in world_to_associations.items() if w != "domestic" and len(assocs) > 0)
                if other_worlds_count > 0:
                    # Prefer non-domestic worlds if available
                    continue
        
        # Only add if world hasn't reached max_per_world limit
        if len(world_to_associations[world]) < max_per_world:
            world_to_associations[world].append(assoc)
            filtered.append(assoc)
            if world == "domestic":
                domestic_count += 1
    
    logger.info(f"ASSOC_DIVERSITY_FILTER input_count={len(associations)} output_count={len(filtered)} worlds_represented={len(world_to_associations)} domestic_count={domestic_count}")
    
    return filtered


def generate_associations(goal: str, size: int = 80) -> List[str]:
    """
    Generate associations from the advertising goal (deterministic).
    
    Rules:
    - Default: EXACTLY 80 associations
    - Can be expanded to 120, 200, 300, 400, or 500 as fallback (deterministic repetition)
    - Associations must be physical, experiential, or functional (not abstract words)
    - No symbols, metaphors, or emotions as standalone items
    - Ordered list: index 1 = strongest association
    - Associations are internal only (not returned to frontend)
    - CONCEPTUAL DISTANCE LAW: Only ONE association per world (filtered for diversity)
    
    Args:
        goal: Advertising goal string
        size: Number of associations to generate (80, 120, 200, 300, 400, or 500)
    
    Returns:
        List of association strings, ordered by strength, filtered for conceptual distance
    
    Raises:
        ValueError: If goal not found in associations library or invalid size
    """
    if goal not in ASSOCIATIONS_LIBRARY:
        raise ValueError(f"Goal '{goal}' not found in associations library")
    
    base_associations = ASSOCIATIONS_LIBRARY[goal]
    
    if len(base_associations) != 80:
        raise ValueError(f"Expected exactly 80 associations for goal '{goal}', got {len(base_associations)}")
    
    # Validate no duplicates in base
    if len(set(base_associations)) != len(base_associations):
        raise ValueError(f"Duplicate associations found for goal '{goal}'")
    
    # Apply CONCEPTUAL DISTANCE filter: only ONE association per world
    # This ensures maximum diversity and prevents "obvious" pairs
    filtered_associations = filter_associations_for_diversity(base_associations, max_per_world=1)
    
    # If filtered list is too small (< 20), allow up to 2 per world for critical worlds
    if len(filtered_associations) < 20:
        logger.warning(f"ASSOC_DIVERSITY_FILTER result too small ({len(filtered_associations)}), allowing 2 per world for critical worlds")
        filtered_associations = filter_associations_for_diversity(base_associations, max_per_world=2)
    
    # Return filtered associations if size is 80
    if size == 80:
        return filtered_associations
    
    # Expand deterministically for larger sizes
    # Use filtered associations as base to maintain diversity
    expanded = list(filtered_associations)
    
    if size == 120:
        # Filtered base + additional from original (filtered again)
        additional = filter_associations_for_diversity(base_associations[len(filtered_associations):], max_per_world=1)
        expanded.extend(additional[:min(40, len(additional))])
    elif size == 200:
        # Filtered base + more from original (filtered again)
        remaining = base_associations[len(filtered_associations):]
        additional = filter_associations_for_diversity(remaining, max_per_world=1)
        expanded.extend(additional[:min(120, len(additional))])
    elif size == 300:
        # Filtered base + more from original (filtered again)
        remaining = base_associations[len(filtered_associations):]
        additional = filter_associations_for_diversity(remaining, max_per_world=1)
        expanded.extend(additional[:min(220, len(additional))])
    elif size == 400:
        # Filtered base + more from original (filtered again)
        remaining = base_associations[len(filtered_associations):]
        additional = filter_associations_for_diversity(remaining, max_per_world=1)
        expanded.extend(additional[:min(320, len(additional))])
    elif size == 500:
        # Filtered base + more from original (filtered again)
        remaining = base_associations[len(filtered_associations):]
        additional = filter_associations_for_diversity(remaining, max_per_world=1)
        expanded.extend(additional[:min(420, len(additional))])
    elif size == 700:
        # Filtered base + more from original (filtered again)
        remaining = base_associations[len(filtered_associations):]
        additional = filter_associations_for_diversity(remaining, max_per_world=1)
        expanded.extend(additional[:min(620, len(additional))])
    elif size == 1000:
        # Filtered base + more from original (filtered again)
        remaining = base_associations[len(filtered_associations):]
        additional = filter_associations_for_diversity(remaining, max_per_world=1)
        expanded.extend(additional[:min(920, len(additional))])
    else:
        raise ValueError(f"Invalid association size: {size}. Must be 80, 120, 200, 300, 400, 500, 700, or 1000")
    
    # Apply final diversity filter to expanded list to ensure world uniqueness
    # This ensures that even after expansion, we maintain conceptual distance
    final_filtered = filter_associations_for_diversity(expanded, max_per_world=1)
    
    # If we need more, allow 2 per world for critical worlds
    if len(final_filtered) < size:
        final_filtered = filter_associations_for_diversity(expanded, max_per_world=2)
    
    # Trim to exact size if needed
    return final_filtered[:size] if len(final_filtered) >= size else final_filtered


def determine_shape_family(object_name: str) -> str:
    """
    Determine shape family for an object based on its name (deterministic keyword matching).
    
    Args:
        object_name: Name of the object
    
    Returns:
        Shape family string (e.g., "circle", "rectangle", "bottle", etc.) or "unknown"
    """
    name_lower = object_name.lower()
    
    # Circle family (expanded keywords)
    if any(kw in name_lower for kw in ["ball", "sphere", "orb", "globe", "bead", "pearl", "bubble", "circle", "round", "wheel", "tire", "rim", "rotor"]):
        return "circle"
    
    # Disk family (must come before circle to catch specific disk items)
    if any(kw in name_lower for kw in ["disc", "disk", "plate", "coin", "frisbee", "puck"]):
        return "disk"
    
    # Gear disk family (gears, cogs, sprockets)
    if any(kw in name_lower for kw in ["gear", "cog", "sprocket"]):
        return "gear_disk"
    
    # Oval family
    if any(kw in name_lower for kw in ["oval", "ellipse", "egg", "almond", "teardrop"]):
        return "oval"
    
    # Rectangle family (expanded keywords)
    if any(kw in name_lower for kw in ["rectangle", "rectangular", "block", "brick", "slab", "tile", "book", "notebook"]):
        return "rectangle"
    
    # Square family
    if any(kw in name_lower for kw in ["square", "cube", "box", "crate", "package"]):
        return "square"
    
    # Triangle family
    if any(kw in name_lower for kw in ["triangle", "triangular", "pyramid", "cone", "arrowhead"]):
        return "triangle"
    
    # Leaf family (expanded keywords, must come before blade/bolt)
    if any(kw in name_lower for kw in ["leaf", "petal", "feather", "wing"]):
        return "leaf"
    
    # Teardrop family
    if any(kw in name_lower for kw in ["teardrop", "drop", "raindrop", "waterdrop"]):
        return "teardrop"
    
    # Bottle family (expanded keywords)
    if any(kw in name_lower for kw in ["bottle", "jar", "flask", "vial"]):
        return "bottle"
    
    # Cylinder family (expanded keywords)
    if any(kw in name_lower for kw in ["cylinder", "tube", "pipe", "rod", "pole", "column", "pillar"]):
        return "cylinder"
    
    # Box family (separate from square for compatibility)
    if any(kw in name_lower for kw in ["box", "crate", "case", "chest", "container"]):
        return "box"
    
    # Book family (handled by rectangle above, but keep for compatibility)
    if any(kw in name_lower for kw in ["book", "notebook", "journal", "tome", "volume"]):
        return "rectangle"  # Books are rectangles
    
    # Fish family
    if any(kw in name_lower for kw in ["fish", "shark", "dolphin", "whale"]):
        return "fish"
    
    # Wing family (handled by leaf above, but keep for compatibility)
    if any(kw in name_lower for kw in ["wing", "airfoil", "sail"]):
        return "leaf"  # Wings are leaf-like
    
    # Blade family (must come after leaf to avoid conflicts)
    if any(kw in name_lower for kw in ["blade", "knife", "sword", "razor", "cutting"]):
        return "blade"
    
    # Bolt family (expanded keywords)
    if any(kw in name_lower for kw in ["bolt", "lightning", "flash", "zigzag", "arrow"]):
        return "bolt"
    
    # Helmet family
    if any(kw in name_lower for kw in ["helmet", "cap", "hat", "crown"]):
        return "helmet"
    
    # Shield family (must come after disk to avoid conflicts)
    if any(kw in name_lower for kw in ["shield", "discus"]):
        return "shield"
    
    # Super-families (expanded keyword matching for better overlap detection)
    
    # Soft rectangle family (expanded keywords, must come before rectangle)
    if any(kw in name_lower for kw in ["pillow", "cushion", "mattress", "blanket", "quilt", "comforter", "duvet", "sheet"]):
        return "soft_rectangle"
    
    # Long rectangle family (plank, board, beam, ruler, ruler-like, knife, blade, sword, razor, scalpel)
    if any(kw in name_lower for kw in ["plank", "board", "beam", "ruler", "knife", "blade", "sword", "razor", "scalpel", "chisel", "awl"]):
        return "long_rectangle"
    
    # Round soft family (balloon, bubble, orb, pearl, bead, donut, bagel, ring, hoop)
    # Note: wheel/tire are handled by circle above, orb/pearl/bubble are handled by circle above
    if any(kw in name_lower for kw in ["balloon", "donut", "bagel", "ring", "hoop"]):
        return "round_soft"
    
    # Container family (bottle, jar, can, tube, vial, flask)
    if any(kw in name_lower for kw in ["bottle", "jar", "can", "tube", "vial", "flask", "container", "vessel", "pot", "jug"]):
        return "container"
    
    # Shield-like family (shield, helmet, mask, goggles) - general, not text
    if any(kw in name_lower for kw in ["helmet", "mask", "goggles", "visor", "faceplate"]):
        return "shield_like"
    
    # Disk family (coin, plate, disc, discus)
    if any(kw in name_lower for kw in ["coin", "plate", "disc", "discus", "frisbee", "puck"]):
        return "disk"
    
    # Spring-like family (spring, coil)
    if any(kw in name_lower for kw in ["spring", "coil", "spiral", "helix", "corkscrew"]):
        return "spring_like"
    
    # Wire-like family (cable, rope, chain, cord, thread)
    if any(kw in name_lower for kw in ["cable", "rope", "chain", "cord", "thread", "wire", "string", "line", "fiber", "strand"]):
        return "wire_like"
    
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
            shape_family=shape_family,
            association_key=association  # All objects from this association share the same key
        )
        candidates.append(candidate)
    
    return candidates


def build_object_pool(product_name: str, product_description: str, goal: Optional[str] = None, association_size: int = 80) -> List[ObjectCandidate]:
    """
    Build sanitized object pool from product information.
    
    Pipeline:
    1) goal = derive_advertising_goal(...) or use provided goal
    2) associations = generate_associations(goal, size=association_size)
    3) For each association (rank 1..N):
       - objects = map_association_to_objects(...)
       - add to pool with correct association_rank
    4) Apply sanitize_candidate_pool(...) from STEP 2.5
    5) If sanitized pool is empty → FAIL explicitly
    
    Args:
        product_name: Name of the product
        product_description: Description of the product
        goal: Optional advertising goal (if None, will be derived from product info)
        association_size: Number of associations to use (80, 120, or 200)
    
    Returns:
        Sanitized list of ObjectCandidate instances
    
    Raises:
        ValueError: If any step fails or sanitized pool is empty
    """
    # Step 1: Derive advertising goal (hidden, not returned to frontend) or use provided goal
    if goal is None:
        try:
            goal = derive_advertising_goal(product_name, product_description)
        except Exception as e:
            raise ValueError(f"Failed to derive advertising goal: {str(e)}") from e
    
    # Step 2: Generate associations (default 80, can be expanded to 120 or 200)
    try:
        associations = generate_associations(goal, size=association_size)
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


def generate_candidate_pairs(
    objs: List[ObjectCandidate],
    max_pairs: int = 300,
    ad_index: int = 0,
    session_seed: Optional[str] = None,
    assoc_count: int = 80,
    top_n: int = 80
) -> List[Tuple[ObjectCandidate, ObjectCandidate]]:
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
    - Window size scales with assoc_count (via top_n parameter)
    
    Strategy:
    - Use top_n objects (scaled with assoc_count, capped at 600)
    - Window position is determined by ad_index and optional session_seed
    - Create pairs in nested order i<j until max_pairs reached
    
    Args:
        objs: List of ObjectCandidate objects (should be ranked)
        max_pairs: Maximum number of pairs to generate (scaled with assoc_count, capped at 3000)
        ad_index: Index of ad in batch (0, 1, or 2) to select different window
        session_seed: Optional session seed string to vary windows between sessions
        assoc_count: Number of associations (used for scaling, passed for logging)
        top_n: Number of top objects to use for pairing (scaled with assoc_count, capped at 600)
    
    Returns:
        List of (object_a, object_b) tuples, deterministically ordered
    """
    if len(objs) < 2:
        return []
    
    # Use top_n objects (scaled with assoc_count)
    # This ensures expansion actually increases search space
    window_size = min(top_n, len(objs))
    base_offset = ad_index * 40
    
    # If session_seed provided, add deterministic offset based on hash
    if session_seed is not None:
        seed_hash = hashlib.sha256(session_seed.encode('utf-8')).hexdigest()
        seed_int = int(seed_hash[:8], 16)
        base_offset += (seed_int % 60)  # Small offset to vary between sessions
    
    # Calculate window bounds, ensuring we don't go out of range
    # For large top_n, start from beginning (base_offset becomes less relevant)
    if window_size >= len(objs):
        # Use all objects if top_n >= total objects
        candidate_pool = objs
    else:
        # Use sliding window approach for smaller windows
        start = min(base_offset, max(0, len(objs) - window_size))
        end = min(start + window_size, len(objs))
        candidate_pool = objs[start:end]
    
    if len(candidate_pool) < 2:
        return []
    
    # Generate pairs in nested order: i < j to avoid (A,B) and (B,A) duplicates
    # Also ensures no object pairs with itself
    pairs = []
    for i in range(len(candidate_pool)):
        for j in range(i + 1, len(candidate_pool)):
            if len(pairs) >= max_pairs:
                break
            pairs.append((candidate_pool[i], candidate_pool[j]))
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
    - Same super-family -> 0.80 (high overlap)
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
    
    # If either is unknown, return 0.55 (improved from 0.40)
    if shape_a == "unknown" or shape_b == "unknown":
        overlap = 0.55
        logger.info(f"OVERLAP_DEBUG A={object_a_silhouette.name if hasattr(object_a_silhouette, 'name') else shape_a}(shape={shape_a}) B={object_b_silhouette.name if hasattr(object_b_silhouette, 'name') else shape_b}(shape={shape_b}) overlap={overlap:.2f}")
        return overlap
    
    # Same shape family -> high overlap
    if shape_a == shape_b:
        overlap = 0.80
        logger.info(f"OVERLAP_DEBUG A={object_a_silhouette.name if hasattr(object_a_silhouette, 'name') else shape_a}(shape={shape_a}) B={object_b_silhouette.name if hasattr(object_b_silhouette, 'name') else shape_b}(shape={shape_b}) overlap={overlap:.2f}")
        return overlap
    
    # Compatible super-families -> 0.70 (for HYBRID eligibility at threshold >= 0.6)
    # Circle/disk/gear_disk together
    circle_disk_family = {"circle", "disk", "gear_disk"}
    if shape_a in circle_disk_family and shape_b in circle_disk_family:
        overlap = 0.70
        logger.info(f"OVERLAP_DEBUG A={object_a_silhouette.name if hasattr(object_a_silhouette, 'name') else shape_a}(shape={shape_a}) B={object_b_silhouette.name if hasattr(object_b_silhouette, 'name') else shape_b}(shape={shape_b}) overlap={overlap:.2f} (circle/disk/gear_disk compatible)")
        return overlap
    
    # Rectangle/soft_rectangle together
    rectangle_family = {"rectangle", "soft_rectangle"}
    if shape_a in rectangle_family and shape_b in rectangle_family:
        overlap = 0.70
        logger.info(f"OVERLAP_DEBUG A={object_a_silhouette.name if hasattr(object_a_silhouette, 'name') else shape_a}(shape={shape_a}) B={object_b_silhouette.name if hasattr(object_b_silhouette, 'name') else shape_b}(shape={shape_b}) overlap={overlap:.2f} (rectangle/soft_rectangle compatible)")
        return overlap
    
    # Bottle/cylinder together
    bottle_cylinder_family = {"bottle", "cylinder"}
    if shape_a in bottle_cylinder_family and shape_b in bottle_cylinder_family:
        overlap = 0.70
        logger.info(f"OVERLAP_DEBUG A={object_a_silhouette.name if hasattr(object_a_silhouette, 'name') else shape_a}(shape={shape_a}) B={object_b_silhouette.name if hasattr(object_b_silhouette, 'name') else shape_b}(shape={shape_b}) overlap={overlap:.2f} (bottle/cylinder compatible)")
        return overlap
    
    # Super-family compatibility table (same super-family -> 0.80)
    # If both shapes are the same super-family, return 0.80
    super_families = ["soft_rectangle", "long_rectangle", "round_soft", "container", "wire_like", "disk", "shield_like", "spring_like"]
    
    if shape_a in super_families and shape_b in super_families and shape_a == shape_b:
        overlap = 0.80
        logger.info(f"OVERLAP_DEBUG A={object_a_silhouette.name if hasattr(object_a_silhouette, 'name') else shape_a}(shape={shape_a}) B={object_b_silhouette.name if hasattr(object_b_silhouette, 'name') else shape_b}(shape={shape_b}) overlap={overlap:.2f} (super-family match)")
        return overlap
    
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
        overlap = 0.55
        logger.info(f"OVERLAP_DEBUG A={object_a_silhouette.name if hasattr(object_a_silhouette, 'name') else shape_a}(shape={shape_a}) B={object_b_silhouette.name if hasattr(object_b_silhouette, 'name') else shape_b}(shape={shape_b}) overlap={overlap:.2f}")
        return overlap
    
    # Different families -> low overlap
    overlap = 0.40
    logger.info(f"OVERLAP_DEBUG A={object_a_silhouette.name if hasattr(object_a_silhouette, 'name') else shape_a}(shape={shape_a}) B={object_b_silhouette.name if hasattr(object_b_silhouette, 'name') else shape_b}(shape={shape_b}) overlap={overlap:.2f}")
    return overlap


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


def evaluate_geometric_overlap_threshold(overlap_percentage: float, hybrid_threshold: float) -> Tuple[bool, bool]:
    """
    Evaluate if geometric overlap meets HYBRID eligibility threshold.
    
    HARD GATE: All conditions must be met for HYBRID to be allowed.
    
    Args:
        overlap_percentage: Calculated overlap (0.0 to 1.0)
        hybrid_threshold: Dynamic HYBRID threshold (FIDELITY) for this session
    
    Returns:
        Tuple of (hybrid_allowed, side_by_side_forced):
        - hybrid_allowed: True if overlap >= threshold (HYBRID eligible)
        - side_by_side_forced: True if overlap < threshold (HYBRID forbidden, but SIDE_BY_SIDE is disabled)
    """
    if overlap_percentage >= hybrid_threshold:
        # Overlap >= threshold: HYBRID eligible (subject to other conditions)
        return (True, False)
    else:
        # Overlap < threshold: HYBRID forbidden (SIDE_BY_SIDE is disabled)
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
    - CORE Geometric Hybrid: outer silhouette overlap meets threshold, no material/structure logic
    - Material Analogy: requires material transformation
    - Structural Morphology: biological/architectural structure similarity
    - Structural Pattern: micro-structure/repeated units
    
    Note: Threshold gating is done before this function is called, so overlap is guaranteed to meet threshold.
    
    Args:
        overlap_percentage: Geometric overlap (0.0 to 1.0) - guaranteed to meet threshold
        similarity_basis: "geometric", "material", "structural", "pattern"
        requires_material_transformation: True if material analogy is needed
        requires_structural_similarity: True if structural morphology is needed
        requires_micro_structure: True if structural pattern exception is needed
    
    Returns:
        HybridType classification (always HYBRID type, never SIDE_BY_SIDE)
    """
    # Threshold gating is done before this function, so we never return SIDE_BY_SIDE here
    
    # Check if similarity relies on material/structure (Exception types)
    if requires_material_transformation:
        return HybridType.MATERIAL_ANALOGY
    
    if requires_structural_similarity:
        return HybridType.STRUCTURAL_MORPHOLOGY
    
    if requires_micro_structure:
        return HybridType.STRUCTURAL_PATTERN
    
    # If similarity is purely geometric (outer silhouette)
    # Note: Threshold gating is done before this function, so overlap is guaranteed to meet threshold
    if similarity_basis == "geometric":
        return HybridType.CORE_GEOMETRIC
    
    # Default: CORE_GEOMETRIC (should not reach here, but safe fallback)
    return HybridType.CORE_GEOMETRIC


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
    
    # SIDE_BY_SIDE is permanently disabled - should never reach here
    if hybrid_type == HybridType.SIDE_BY_SIDE:
        return False  # Reject SIDE_BY_SIDE
    
    return False


def evaluate_ab_pair(
    object_a: Any,
    object_b: Any,
    quota_state: BatchQuotaState,
    similarity_basis: str = "geometric",
    requires_material_transformation: bool = False,
    requires_structural_similarity: bool = False,
    requires_micro_structure: bool = False,
    hybrid_threshold: float = 0.70
) -> Tuple[bool, HybridType, Optional[str]]:
    """
    Evaluate an A/B pair against all core decision rules.
    
    This is the main decision function that applies all canonical rules:
    1. Select max-projection views
    2. Calculate geometric overlap
    3. Evaluate overlap threshold (dynamic threshold gate)
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
        hybrid_threshold: Dynamic HYBRID threshold (FIDELITY) for this session
    
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
    
    # Step 3: Evaluate geometric overlap threshold (HARD GATE with dynamic threshold)
    hybrid_allowed, side_by_side_forced = evaluate_geometric_overlap_threshold(overlap_percentage, hybrid_threshold)
    
    # Step 4: If overlap < threshold, HYBRID is strictly forbidden
    # SIDE_BY_SIDE is PERMANENTLY DISABLED - reject pair if overlap < threshold
    if not hybrid_allowed:
        # HYBRID forbidden, SIDE_BY_SIDE is disabled - reject this pair
        return (False, HybridType.CORE_GEOMETRIC, f"Overlap {overlap_percentage:.2f} < HYBRID threshold {hybrid_threshold:.2f} (SIDE_BY_SIDE disabled)")
    
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


def normalize_token_set(name: str) -> set:
    """
    Normalize object name to a set of tokens for conceptual comparison.
    
    Steps:
    - lowercase
    - split into words
    - remove stopwords ("of", "the", "and", "a", "an")
    - simple singularization (remove trailing 's')
    - remove punctuation
    
    Args:
        name: Object name string
    
    Returns:
        Set of normalized tokens
    """
    # Basic stopwords
    stopwords = {"of", "the", "and", "a", "an", "in", "on", "at", "to", "for", "with", "by"}
    
    # Lowercase and remove punctuation
    name_clean = re.sub(r'[^\w\s]', ' ', name.lower())
    
    # Split into words
    words = name_clean.split()
    
    # Remove stopwords and singularize (simple: remove trailing 's')
    tokens = set()
    for word in words:
        if word not in stopwords and len(word) > 0:
            # Simple singularization: remove trailing 's' if word is longer than 3 chars
            if len(word) > 3 and word.endswith('s'):
                word = word[:-1]
            tokens.add(word)
    
    return tokens


def is_conceptually_too_close(a_name: str, b_name: str) -> bool:
    """
    Check if two object names are conceptually too close (same idea or conceptual neighbors).
    
    ABSOLUTE LAW: If two objects could share a supplier, store, room, or use case — they are too close.
    Similar shape does NOT equal similar meaning, but shared context does.
    
    Returns True if:
    a) One name contains the other (substring) after normalization
    b) They share significant tokens (intersection size >= 1)
    c) They belong to the same conceptual cluster (bedding, water, machinery, etc.)
    d) They could plausibly appear in the same photograph, room, store, or use case
    
    Args:
        a_name: First object name
        b_name: Second object name
    
    Returns:
        True if objects are conceptually too close, False otherwise
    """
    # Normalize both names
    tokens_a = normalize_token_set(a_name)
    tokens_b = normalize_token_set(b_name)
    
    # Check substring containment (after normalization)
    a_normalized = ' '.join(sorted(tokens_a))
    b_normalized = ' '.join(sorted(tokens_b))
    
    if a_normalized in b_normalized or b_normalized in a_normalized:
        return True
    
    # Check token intersection
    if len(tokens_a & tokens_b) >= 1:
        return True
    
    # Check conceptual clusters (expanded for known problematic areas)
    # ABSOLUTE LAW: If two objects could share a supplier, store, room, or use case — they are too close
    bedding_cluster = {"pillow", "cushion", "mattress", "blanket", "duvet", "sheet", "bed", "sofa", "armchair", "recliner", "quilt", "comforter", "furniture", "home", "hearth"}
    water_cluster = {"water", "dew", "rain", "mist", "fog", "ice", "frost", "bottle", "river", "lake", "ocean", "sea", "stream", "pond", "pool", "wet", "moisture", "droplet", "liquid", "fluid"}
    machinery_cluster = {"gear", "cog", "wheel", "turbine", "engine", "motor", "rotor", "pulley", "sprocket", "mechanism", "machine"}
    tool_cluster = {"tool", "wrench", "hammer", "screwdriver", "pliers", "drill", "saw", "chisel", "toolbox", "workshop"}
    office_cluster = {"desk", "chair", "lamp", "pen", "paper", "notebook", "computer", "keyboard", "office", "workspace"}
    kitchen_cluster = {"knife", "fork", "spoon", "plate", "bowl", "cup", "pot", "pan", "kitchen", "cooking", "utensil"}
    
    # Check if both names belong to the same cluster
    # If they could plausibly appear in the same photograph, room, store, or use case — reject
    clusters = [
        ("bedding", bedding_cluster),
        ("water", water_cluster),
        ("machinery", machinery_cluster),
        ("tool", tool_cluster),
        ("office", office_cluster),
        ("kitchen", kitchen_cluster)
    ]
    
    for cluster_name, cluster_tokens in clusters:
        a_in_cluster = any(token in cluster_tokens for token in tokens_a)
        b_in_cluster = any(token in cluster_tokens for token in tokens_b)
        if a_in_cluster and b_in_cluster:
            # Both objects could plausibly appear in the same context — too close
            return True
    
    return False


def find_valid_hybrid_pair(
    candidate_pairs: List[Tuple[ObjectCandidate, ObjectCandidate]],
    quota_state: BatchQuotaState,
    ranked_associations: List[ObjectCandidate],
    ad_index: int = 0,
    hybrid_threshold: float = 0.70,
    max_evaluations: int = 700
) -> Tuple[Optional[ObjectCandidate], Optional[ObjectCandidate], HybridType, Optional[str], Dict[str, int]]:
    """
    Find a valid HYBRID pair from candidates that passes all rules.
    SIDE_BY_SIDE is PERMANENTLY DISABLED.
    
    Candidate Resolution Principle:
    - Geometry is a pass/fail gate, not a ranking tool
    - Only HYBRID-eligible pairs (overlap >= 0.70) are considered
    - If multiple HYBRID candidates, select based on ad_index for variation
    - NEVER use geometric overlap to choose between candidates (only pass/fail)
    
    Args:
        candidate_pairs: List of (object_a, object_b) candidate pairs (already sanitized)
        quota_state: Batch quota state
        ranked_associations: Ranked list of associations for resolution
        ad_index: Index of ad in batch (0, 1, or 2) to select different valid pair
        hybrid_threshold: Fixed HYBRID overlap threshold (FIDELITY) for this session
        max_evaluations: Maximum number of pairs to evaluate (hard cap, default 700)
    
    Returns:
        Tuple of (object_a, object_b, hybrid_type, error_message, search_stats):
        - object_a, object_b: Selected HYBRID pair if valid, None if none found
        - hybrid_type: Classification of the valid pair (always HYBRID type, never SIDE_BY_SIDE)
        - error_message: None if valid pair found, error if none found
        - search_stats: Dict with 'evaluated_pairs' and 'passed_overlap' counts
    """
    # Note: All candidates in candidate_pairs have already passed hard sanitation filters
    
    # Save original quota_state to restore after evaluation (evaluate_ab_pair modifies it)
    original_quota_dict = quota_state_to_dict(quota_state)
    
    # Step 1: Collect valid HYBRID pairs only (SIDE_BY_SIDE is permanently disabled)
    # Hard rule: Reject same_family pairs and too-close name pairs (HARD REJECT, no fallback)
    
    def are_names_too_close(name_a: str, name_b: str) -> bool:
        """
        Check if two object names are too similar (one contains the other or shares key tokens).
        This prevents nesting scenarios like "pillow" inside "pillow" or "cushion" inside "pillow".
        """
        name_a_lower = name_a.lower().strip()
        name_b_lower = name_b.lower().strip()
        
        # If one name is contained in the other, they are too close
        if name_a_lower in name_b_lower or name_b_lower in name_a_lower:
            return True
        
        # Check for shared key tokens (simple word-based check)
        # Common soft rectangle family tokens
        soft_rect_tokens = ["pillow", "cushion", "blanket", "duvet", "mattress", "sheet", "quilt", "comforter"]
        # Common container family tokens
        container_tokens = ["bottle", "jar", "can", "vial", "flask", "container", "vessel"]
        # Common disk family tokens
        disk_tokens = ["coin", "plate", "disc", "discus", "frisbee", "puck"]
        
        # Split names into words
        words_a = set(name_a_lower.split())
        words_b = set(name_b_lower.split())
        
        # Check if they share any key token from the same family
        for token_list in [soft_rect_tokens, container_tokens, disk_tokens]:
            tokens_in_a = [t for t in token_list if t in words_a]
            tokens_in_b = [t for t in token_list if t in words_b]
            if tokens_in_a and tokens_in_b:
                # Both contain tokens from the same family - too close
                return True
        
        return False
    
    hybrid_candidates_different_family = []
    
    # Track statistics clearly
    evaluated_pairs = 0
    passed_overlap = 0  # Count of pairs where overlap >= threshold
    rejected_after_overlap = 0  # Count of pairs that passed overlap but were rejected by later gates
    rejected_too_close = 0  # Count of pairs rejected due to same_family or too-close names
    rejected_same_association = 0  # Count of pairs rejected due to same association_key
    rejected_conceptual_neighbors = 0  # Count of pairs rejected due to conceptual closeness
    selected_pair = None  # First pair that passes all gates
    
    for object_a, object_b in candidate_pairs:
        # Hard cap: stop after max_evaluations
        if evaluated_pairs >= max_evaluations:
            break
        
        evaluated_pairs += 1
        # Restore quota_state before each evaluation (evaluate_ab_pair modifies it)
        quota_state.material_analogy_used = original_quota_dict['material_analogy_used']
        quota_state.structural_morphology_used = original_quota_dict['structural_morphology_used']
        quota_state.structural_exception_used = original_quota_dict['structural_exception_used']
        
        # Gate 1: HARD REJECT same association_key (same association source)
        if hasattr(object_a, 'association_key') and hasattr(object_b, 'association_key'):
            if object_a.association_key == object_b.association_key and object_a.association_key:
                rejected_same_association += 1
                logger.info(f"ASSOC_REJECT reason=SAME_ASSOCIATION A={object_a.name} B={object_b.name} assoc={object_a.association_key}")
                continue  # HARD REJECT - skip this pair
        
        # Gate 2: HARD REJECT conceptually too close (conceptual neighbors)
        if is_conceptually_too_close(object_a.name, object_b.name):
            rejected_conceptual_neighbors += 1
            logger.info(f"ASSOC_REJECT reason=CONCEPTUAL_NEIGHBORS A={object_a.name} B={object_b.name}")
            continue  # HARD REJECT - skip this pair
        
        # Gate 3: HARD REJECT same_family pairs and too-close name pairs (no fallback)
        same_family = (object_a.shape_family == object_b.shape_family)
        names_too_close = are_names_too_close(object_a.name, object_b.name)
        
        if same_family or names_too_close:
            rejected_too_close += 1
            logger.info(f"ENV_EMBED_REJECT reason=TOO_CLOSE_OR_SAME_FAMILY A={object_a.name} B={object_b.name} shapeA={object_a.shape_family} shapeB={object_b.shape_family} same_family={same_family} names_too_close={names_too_close}")
            continue  # HARD REJECT - skip this pair
        
        # Calculate overlap to check if HYBRID is allowed
        view_a = select_max_projection_view(object_a)
        view_b = select_max_projection_view(object_b)
        overlap_percentage = calculate_geometric_overlap(view_a, view_b)
        
        # Only consider pairs with overlap >= HYBRID threshold (SIDE_BY_SIDE is disabled)
        # Use the fixed threshold (FIDELITY) for this session
        if overlap_percentage < hybrid_threshold:
            continue  # Skip pairs that don't meet HYBRID threshold
        
        # This pair passed overlap threshold
        passed_overlap += 1
        
        # Evaluate pair to get hybrid_type and check quota
        is_valid, hybrid_type, error_msg = evaluate_ab_pair(
            object_a,
            object_b,
            quota_state,
            similarity_basis="geometric",
            hybrid_threshold=hybrid_threshold
        )
        
        if not is_valid:
            # Pair passed overlap but failed later gate (quota exhausted, etc.)
            rejected_after_overlap += 1
            continue
        
        # Only accept HYBRID types (SIDE_BY_SIDE is permanently disabled)
        if hybrid_type == HybridType.SIDE_BY_SIDE:
            # Pair passed overlap but is SIDE_BY_SIDE (should not happen, but track it)
            rejected_after_overlap += 1
            continue
        
        # This is a valid HYBRID-eligible pair that passed all gates
        # Select the FIRST such pair immediately (no need to collect all candidates)
        if selected_pair is None:
            selected_pair = (object_a, object_b, hybrid_type, overlap_percentage)
        
        # Store for selection logic (all pairs here are different_family and not too-close)
        hybrid_candidates_different_family.append((object_a, object_b, hybrid_type, overlap_percentage))
    
    # Step 2: Select pair from valid candidates (all are different_family and not too-close)
    # Sanity check: if all pairs passed overlap but none selected, this is a logic error
    if passed_overlap == evaluated_pairs and passed_overlap > 0 and selected_pair is None:
        error_msg = f"SEARCH_LOGIC_BROKEN: all {passed_overlap} pairs pass overlap but none selected"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Select from valid candidates (all are different_family and not too-close)
    if len(hybrid_candidates_different_family) > 0:
        # Select the (ad_index+1)-th HYBRID candidate with different families (0-indexed)
        target_rank = ad_index
        if target_rank >= len(hybrid_candidates_different_family):
            # Fallback to first HYBRID candidate if not enough
            logger.warning(f"find_valid_hybrid_pair: ad_index={ad_index} but only {len(hybrid_candidates_different_family)} HYBRID candidates with different families, using first")
            target_rank = 0
        
        object_a, object_b, hybrid_type, overlap = hybrid_candidates_different_family[target_rank]
        
        # Re-evaluate the selected pair to update quota_state correctly
        quota_state.material_analogy_used = original_quota_dict['material_analogy_used']
        quota_state.structural_morphology_used = original_quota_dict['structural_morphology_used']
        quota_state.structural_exception_used = original_quota_dict['structural_exception_used']
        is_valid, final_hybrid_type, error_msg = evaluate_ab_pair(
            object_a,
            object_b,
            quota_state,
            similarity_basis="geometric",
            hybrid_threshold=hybrid_threshold
        )
        
        search_stats = {
            'evaluated_pairs': evaluated_pairs, 
            'passed_overlap': passed_overlap,
            'rejected_after_overlap': rejected_after_overlap,
            'rejected_too_close': rejected_too_close,
            'rejected_same_association': rejected_same_association,
            'rejected_conceptual_neighbors': rejected_conceptual_neighbors
        }
        return (object_a, object_b, final_hybrid_type, None, search_stats)
    
    # No valid HYBRID pairs found
    search_stats = {
        'evaluated_pairs': evaluated_pairs, 
        'passed_overlap': passed_overlap,
        'rejected_after_overlap': rejected_after_overlap,
        'rejected_too_close': rejected_too_close,
        'rejected_same_association': rejected_same_association,
        'rejected_conceptual_neighbors': rejected_conceptual_neighbors
    }
    return (None, None, HybridType.CORE_GEOMETRIC, f"No valid HYBRID pair found (overlap >= {hybrid_threshold} required)", search_stats)


def generate_ad(
    product_name: str,
    product_description: str,
    size: str,
    quota_state: Optional[BatchQuotaState] = None,
    ad_index: int = 0,
    session_seed: Optional[str] = None
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
    # STEP 3: GOAL + ASSOCIATIONS → OBJECT POOL
    # ========================================================================
    # Build goals for batch (3 different goals for 3-ad batch)
    # Build goals for batch (3 different goals for 3-ad batch)
    # Include session_seed to ensure goals vary across sessions (SESSION DIVERSITY RULE)
    goals_for_batch = build_goals_for_batch(product_name, product_description, session_seed=session_seed)
    goal = goals_for_batch[ad_index] if ad_index < len(goals_for_batch) else goals_for_batch[0]
    
    # ========================================================================
    # RELAXATION LOOP: Guaranteed HYBRID search with threshold and pool expansion
    # ========================================================================
    # Side-by-side is PERMANENTLY DISABLED.
    # The engine will try multiple attempts with decreasing threshold and increasing pool size.
    # Each attempt is limited to 700 comparisons.
    # ========================================================================
    
    # Read initial HYBRID threshold from environment variable
    threshold_str = os.environ.get('ACE_HYBRID_THRESHOLD')
    if threshold_str:
        try:
            initial_threshold = float(threshold_str)
        except ValueError:
            logger.warning(f"Invalid ACE_HYBRID_THRESHOLD value '{threshold_str}', using default 0.70")
            initial_threshold = 0.70
    else:
        initial_threshold = 0.70  # Default threshold
    
    # Relaxation ladder: threshold decreases, association_size increases
    threshold_levels = [initial_threshold, 0.65, 0.60, 0.55, 0.50, 0.45]
    association_levels = [80, 120, 200, 300, 500, 700]
    
    object_a = None
    object_b = None
    hybrid_type = None
    search_stats = None
    final_threshold = None
    final_assoc_size = None
    
    # Try each relaxation level
    for attempt_idx in range(len(threshold_levels)):
        hybrid_threshold = threshold_levels[attempt_idx]
        association_size = association_levels[attempt_idx]
        
        logger.info(f"HYBRID_SEARCH_ATTEMPT attempt={attempt_idx+1} threshold={hybrid_threshold} assoc_size={association_size}")
        
        try:
            # Build associations and object pool for this attempt
            associations = generate_associations(goal, size=association_size)
            object_pool = build_object_pool(product_name, product_description, goal=goal, association_size=association_size)
            
            # Apply hard sanitation filters (STEP 2.5)
            sanitized_pool = sanitize_candidate_pool(object_pool)
            
            if len(sanitized_pool) == 0:
                logger.warning(f"HYBRID_SEARCH_ATTEMPT attempt={attempt_idx+1} sanitized_pool_empty")
                continue
            
            # Log shape family statistics (only on first attempt)
            if attempt_idx == 0:
                unknown_count = sum(1 for obj in sanitized_pool if obj.shape_family == "unknown")
                total_count = len(sanitized_pool)
                logger.info(f"SHAPE_FAMILY_STATS unknown_count={unknown_count} total={total_count}")
            
            # Build ranked object list
            ranked_objs = build_ranked_object_list(sanitized_pool)
            
            # Generate candidate pairs (enough to allow 700 evaluations)
            top_n = min(len(ranked_objs), 600)  # Use up to 600 objects for pairing
            max_pairs = 1000  # Generate enough pairs to allow 700 evaluations
            
            candidate_pairs = generate_candidate_pairs(
                ranked_objs, 
                max_pairs=max_pairs, 
                ad_index=ad_index, 
                session_seed=session_seed,
                assoc_count=association_size,
                top_n=top_n
            )
            
            if len(candidate_pairs) == 0:
                logger.warning(f"HYBRID_SEARCH_ATTEMPT attempt={attempt_idx+1} no_candidate_pairs")
                continue
            
            # Search for HYBRID only (SIDE_BY_SIDE is permanently disabled)
            # Budget: 700 comparisons (hard cap)
            object_a, object_b, hybrid_type, error_msg, search_stats = find_valid_hybrid_pair(
                candidate_pairs,
                quota_state,
                ranked_objs,
                ad_index=ad_index,
                hybrid_threshold=hybrid_threshold,
                max_evaluations=700
            )
            
            # Log search statistics
            rejected_after_overlap = search_stats.get('rejected_after_overlap', 0)
            rejected_too_close = search_stats.get('rejected_too_close', 0)
            rejected_same_association = search_stats.get('rejected_same_association', 0)
            rejected_conceptual_neighbors = search_stats.get('rejected_conceptual_neighbors', 0)
            logger.info(f"HYBRID_SEARCH_STATS threshold={hybrid_threshold} assoc_size={association_size} evaluated_pairs={search_stats['evaluated_pairs']} passed_overlap={search_stats['passed_overlap']} rejected_after_overlap={rejected_after_overlap} rejected_too_close={rejected_too_close} rejected_same_association={rejected_same_association} rejected_conceptual_neighbors={rejected_conceptual_neighbors}")
            
            # If valid HYBRID found, use it
            if object_a is not None and object_b is not None:
                # Calculate overlap for logging
                view_a = select_max_projection_view(object_a)
                view_b = select_max_projection_view(object_b)
                overlap_percentage = calculate_geometric_overlap(view_a, view_b)
                final_threshold = hybrid_threshold
                final_assoc_size = association_size
                logger.info(f"HYBRID_SELECTED threshold={hybrid_threshold} assoc_size={association_size} evaluated_pairs={search_stats['evaluated_pairs']} A={object_a.name} B={object_b.name} overlap={overlap_percentage:.2f}")
                break  # Success! Exit relaxation loop
            else:
                # No HYBRID found within budget for this attempt
                logger.warning(f"HYBRID_NOT_FOUND threshold={hybrid_threshold} assoc_size={association_size} evaluated_pairs={search_stats['evaluated_pairs']} passed_overlap={search_stats['passed_overlap']}")
                # Continue to next relaxation level
        except Exception as e:
            logger.warning(f"HYBRID_SEARCH_ATTEMPT attempt={attempt_idx+1} error={str(e)}")
            continue  # Try next relaxation level
    
    # If still no HYBRID found after all attempts, use fallback
    if object_a is None or object_b is None:
        logger.warning("FALLBACK_MAX_OVERLAP_PAIR_USED: No HYBRID found after all relaxation attempts, using fallback pair")
        # Fallback: use first two objects from ranked list (deterministic)
        if len(ranked_objs) >= 2:
            object_a = ranked_objs[0]
            object_b = ranked_objs[1]
            hybrid_type = HybridType.CORE_GEOMETRIC
            # Calculate overlap for logging
            view_a = select_max_projection_view(object_a)
            view_b = select_max_projection_view(object_b)
            overlap_percentage = calculate_geometric_overlap(view_a, view_b)
            logger.info(f"FALLBACK_MAX_OVERLAP_PAIR_USED A={object_a.name} B={object_b.name} overlap={overlap_percentage:.2f}")
        else:
            raise ValueError("No valid object candidates available for fallback")
    
    # ========================================================================
    # STEP 6 & 7: HEADLINE & MARKETING TEXT GENERATION
    # ========================================================================
    # Use goal based on ad_index (already calculated above) for headline and marketing text
    headline = generate_headline(product_name, goal, ad_index)
    marketing_text = generate_marketing_text(product_name, product_description, goal, ad_index)
    
    # ========================================================================
    # STEP 8: REAL IMAGE GENERATION (OpenAI GPT Image + Headline Overlay)
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
    
    # Generate real image via OpenAI (NO PLACEHOLDER FALLBACK)
    # If OpenAI fails, raise ValueError - no fallback to placeholder
    try:
        image_bytes = generate_real_image_bytes(
            product_name,
            product_description,
            size,
            object_a,
            object_b,
            hybrid_type,
            goal
        )
    except ValueError as e:
        # Re-raise ValueError with clear message (NO PLACEHOLDER FALLBACK)
        logger.error(f"IMAGE_GEN_FAIL in generate_ad: {str(e)}")
        raise ValueError(f"Image generation failed: {str(e)}") from e
    except Exception as e:
        # Wrap any other exception as ValueError (NO PLACEHOLDER FALLBACK)
        logger.error(f"IMAGE_GEN_FAIL in generate_ad: {str(e)}")
        raise ValueError(f"Image generation failed: {str(e)}") from e
    
    # Open image with PIL (from real OpenAI image, NOT placeholder)
    try:
        img = Image.open(io.BytesIO(image_bytes))
        # Convert to RGB if needed (handles PNG/WebP from OpenAI)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        logger.info(f"IMAGE_OPENED size={img.size} mode={img.mode} bytes={len(image_bytes)}")
    except Exception as e:
        logger.error(f"IMAGE_OPEN_FAIL error={str(e)}")
        raise ValueError(f"Failed to open generated image: {str(e)}") from e
    
    # Get image dimensions
    img_width, img_height = img.size
    
    # Create drawing context
    draw = ImageDraw.Draw(img)
    
    # Safe margins
    margin_x = int(img_width * 0.06)
    margin_y = int(img_height * 0.04)
    
    # Try to load a font, fallback to default
    # Start with large font size, will be reduced if needed
    font_size = 72
    font_large = None
    try:
        # Try common font paths
        font_large = ImageFont.truetype("arial.ttf", font_size)
    except (OSError, IOError):
        try:
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except (OSError, IOError):
            # Fallback to default font
            font_large = ImageFont.load_default()
    
    # Measure text width and adjust font size if needed
    max_text_width = img_width - 2 * margin_x
    min_font_size = 18
    
    while True:
        headline_bbox = draw.textbbox((0, 0), headline, font=font_large)
        text_width = headline_bbox[2] - headline_bbox[0]
        
        if text_width <= max_text_width or font_size <= min_font_size:
            break
        
        # Reduce font size
        font_size = max(min_font_size, font_size - 4)
        try:
            font_large = ImageFont.truetype("arial.ttf", font_size)
        except (OSError, IOError):
            try:
                font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except (OSError, IOError):
                # If font loading fails, use default and break
                font_large = ImageFont.load_default()
                break
    
    # Re-measure with final font size
    headline_bbox = draw.textbbox((0, 0), headline, font=font_large)
    text_width = headline_bbox[2] - headline_bbox[0]
    
    # Position headline with safe margins (clamp to avoid cutting)
    headline_x = (img_width - text_width) // 2
    headline_x = max(margin_x, min(headline_x, img_width - margin_x - text_width))
    headline_y = margin_y
    
    # Draw headline with white outline for visibility
    # Draw outline first (thicker)
    for adj in range(-2, 3):
        for adj2 in range(-2, 3):
            draw.text((headline_x + adj, headline_y + adj2), headline, fill='white', font=font_large)
    # Then draw main text
    draw.text((headline_x, headline_y), headline, fill='black', font=font_large)
    
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


# ============================================================================
# STARTUP VALIDATION
# ============================================================================
# Validate ASSOCIATIONS_LIBRARY at module import time to catch errors early
# ============================================================================

validate_associations_library()

