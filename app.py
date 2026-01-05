from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all origins (Phase 1)

# Valid ad sizes
VALID_AD_SIZES = {"1024x1024", "1024x1536", "1536x1024"}


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"ok": True}), 200


@app.route('/generate', methods=['POST'])
def generate():
    """Generate a single mock ad based on product information."""
    # Get JSON body
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400
    
    # Validate required fields
    product_name = data.get("product_name")
    product_description = data.get("product_description")
    ad_size = data.get("ad_size")
    
    # Check all fields are present
    if not product_name:
        return jsonify({"error": "product_name is required and must be a non-empty string"}), 400
    
    if not product_description:
        return jsonify({"error": "product_description is required and must be a non-empty string"}), 400
    
    if not ad_size:
        return jsonify({"error": "ad_size is required and must be a non-empty string"}), 400
    
    # Check all fields are strings
    if not isinstance(product_name, str) or not product_name.strip():
        return jsonify({"error": "product_name must be a non-empty string"}), 400
    
    if not isinstance(product_description, str) or not product_description.strip():
        return jsonify({"error": "product_description must be a non-empty string"}), 400
    
    if not isinstance(ad_size, str) or not ad_size.strip():
        return jsonify({"error": "ad_size must be a non-empty string"}), 400
    
    # Validate ad_size is one of the allowed values
    if ad_size not in VALID_AD_SIZES:
        return jsonify({
            "error": f"ad_size must be exactly one of: {', '.join(sorted(VALID_AD_SIZES))}"
        }), 400
    
    # Generate a single mock ad (Phase 1)
    # Frontend controls sequencing across attempts
    ad_id = 1
    
    # Generate headline: 3-7 words, includes product_name, original
    headline = f"{product_name} transforms your daily experience"
    
    # Generate exactly 50 words of marketing text
    # Base template with product-specific content
    base_text = (
        f"Experience the innovation of {product_name}. "
        f"Our product delivers exceptional quality and performance for your needs. "
        f"Designed with precision and care, {product_name} offers unmatched reliability. "
        f"Join thousands of satisfied customers who trust our solution. "
        f"Transform your workflow with cutting-edge technology. "
        f"Discover why {product_name} stands out from the competition. "
        f"Get started today and see the difference. "
        f"Quality craftsmanship meets modern design principles. "
        f"Trusted by professionals worldwide for consistent results. "
        f"Elevate your standards with proven excellence."
    )
    
    # Split into words and ensure exactly 50 words
    words = base_text.split()
    if len(words) > 50:
        marketing_text_50_words = " ".join(words[:50])
    elif len(words) < 50:
        # Pad with generic marketing phrases to reach exactly 50 words
        padding_words = [
            "Quality", "craftsmanship", "meets", "modern", "design", "principles", "every",
            "time", "you", "use", "it", "Trusted", "by", "professionals", "worldwide",
            "for", "consistent", "results", "that", "exceed", "expectations", "Elevate",
            "your", "standards", "with", "proven", "excellence", "and", "innovation"
        ]
        all_words = words + padding_words
        marketing_text_50_words = " ".join(all_words[:50])
    else:
        marketing_text_50_words = " ".join(words)
    
    ad = {
        "ad_id": ad_id,
        "headline": headline,
        "marketing_text_50_words": marketing_text_50_words,
        "image_url": "https://placeholder.com/1024x1024",
        "zip_url": "https://placeholder.com/download/ad.zip"
    }
    
    return jsonify(ad), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=False)

