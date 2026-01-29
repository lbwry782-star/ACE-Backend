"""
ACE Engine Interface

This module defines the engine interface for generating ads.
STEP 2: Core Decision Logic - implements canonical decision rules for A/B pair evaluation.
STEP 2.5: Object & Input Sanitation - hard exclusions before geometry evaluation.
"""

from typing import Dict, Any, Tuple, Optional, List
from PIL import Image, ImageDraw, ImageFont
import io
from enum import Enum
from dataclasses import dataclass


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


# Global threshold constants (HARD GATES)
GEOMETRIC_OVERLAP_THRESHOLD_HYBRID = 0.70  # 70% - minimum for HYBRID eligibility
GEOMETRIC_OVERLAP_THRESHOLD_SIDE_BY_SIDE = 0.50  # 50% - below this, forced SIDE-BY-SIDE


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


def calculate_geometric_overlap(object_a_silhouette: Any, object_b_silhouette: Any) -> float:
    """
    Calculate geometric overlap percentage between two object silhouettes.
    
    Args:
        object_a_silhouette: Dominant silhouette of object A (max-projection view)
        object_b_silhouette: Dominant silhouette of object B (max-projection view)
    
    Returns:
        Overlap percentage (0.0 to 1.0) when aligned in natural orientation
    
    Note: This is a placeholder. Actual implementation will compute outer-contour overlap
    from silhouette data structures.
    """
    # TODO: Implement actual silhouette overlap calculation
    # For now, this is a placeholder that will be called by the decision logic
    # The actual calculation will use silhouette area intersection / union
    raise NotImplementedError("Silhouette overlap calculation will be implemented with actual object data")


def select_max_projection_view(object: Any) -> Any:
    """
    Select the view with maximum projected silhouette area for an object.
    
    Args:
        object: Object to evaluate
    
    Returns:
        The projection/view with maximum silhouette area
    
    Note: This is a placeholder. Actual implementation will evaluate discrete viewpoints
    and select the one with maximum projected silhouette area.
    """
    # TODO: Implement actual max-projection view selection
    # For now, this is a placeholder
    raise NotImplementedError("Max-projection view selection will be implemented with actual object data")


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
    try:
        view_a = select_max_projection_view(object_a)
        view_b = select_max_projection_view(object_b)
    except NotImplementedError:
        # Placeholder: will be implemented with actual object data
        return (False, HybridType.SIDE_BY_SIDE, "Max-projection view selection not yet implemented")
    
    # Step 2: Calculate geometric overlap
    try:
        overlap_percentage = calculate_geometric_overlap(view_a, view_b)
    except NotImplementedError:
        # Placeholder: will be implemented with actual silhouette data
        return (False, HybridType.SIDE_BY_SIDE, "Geometric overlap calculation not yet implemented")
    
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
    quota_state: Optional[BatchQuotaState] = None
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
    # STEP 2.5: OBJECT & INPUT SANITATION (Hard Gates Before Geometry)
    # ========================================================================
    # Build/receive candidate pool (placeholder until object generation is implemented)
    # For now, create a minimal placeholder pool with valid candidates
    placeholder_candidates = [
        ObjectCandidate(
            name="placeholder_object_1",
            association_rank=1,
            is_physical_object=True,
            contains_readable_text=False,
            contains_logo_or_brand=False,
            has_label_or_packaging_text=False,
            has_communicative_graphics=False,
            has_structural_graphics_only=False,
            is_symbolic_only=False
        ),
        ObjectCandidate(
            name="placeholder_object_2",
            association_rank=2,
            is_physical_object=True,
            contains_readable_text=False,
            contains_logo_or_brand=False,
            has_label_or_packaging_text=False,
            has_communicative_graphics=False,
            has_structural_graphics_only=False,
            is_symbolic_only=False
        )
    ]
    
    # Apply hard sanitation filters BEFORE any candidate pairing
    try:
        sanitized_pool = sanitize_candidate_pool(placeholder_candidates)
    except ValueError as e:
        # No valid candidates after sanitation - FAIL explicitly
        raise ValueError("No valid object candidates after sanitation") from e
    
    # ========================================================================
    # TODO: STEP 3 - Generate ranked association list (size 80) as ObjectCandidate objects
    # TODO: STEP 4 - Generate candidate A/B pairs from sanitized pool
    # TODO: STEP 5 - Call find_valid_ab_pair() with sanitized candidates
    # TODO: STEP 6 - If no valid pair found, raise ValueError("No valid A/B pair found")
    # ========================================================================
    
    # For now, placeholder implementation until object generation is added
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
    
    # Draw text: "ACE DEMO – <SIZE>" at top center
    demo_text = f"ACE DEMO – {size}"
    text_bbox = draw.textbbox((0, 0), demo_text, font=font_large)
    text_width = text_bbox[2] - text_bbox[0]
    text_x = (width - text_width) // 2
    text_y = height // 4
    draw.text((text_x, text_y), demo_text, fill='black', font=font_large)
    
    # Draw product name below (centered)
    product_text = product_name
    product_bbox = draw.textbbox((0, 0), product_text, font=font_small)
    product_width = product_bbox[2] - product_bbox[0]
    product_x = (width - product_width) // 2
    product_y = height // 2
    draw.text((product_x, product_y), product_text, fill='darkblue', font=font_small)
    
    # Convert to JPEG bytes
    jpeg_buffer = io.BytesIO()
    img.save(jpeg_buffer, format='JPEG', quality=95)
    image_bytes_jpg = jpeg_buffer.getvalue()
    jpeg_buffer.close()
    
    # Generate placeholder marketing text
    marketing_text = f"Demo marketing text for {product_name}. This is a placeholder implementation. The actual ACE engine will generate creative content based on: {product_description[:100]}..."
    
    # Return result with updated batch state
    return {
        "image_bytes_jpg": image_bytes_jpg,
        "marketing_text": marketing_text,
        "batch_state": quota_state_to_dict(quota_state)
    }

