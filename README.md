# ACE Backend - Phase 1

Minimal Flask backend for ACE ad generation system.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running Locally (Windows)

### Option 1: Using Flask development server
```bash
python app.py
```

### Option 2: Using gunicorn (production-like)
```bash
gunicorn app:app --timeout 600 --bind 0.0.0.0:10000
```

The server will start on `http://localhost:10000`

## Endpoints

### GET /health
Returns health status.

**Response:**
```json
{"ok": true}
```

### POST /generate
Generates a single mock ad based on product information. The frontend controls sequencing across multiple attempts.

**Request Body:**
```json
{
  "product_name": "string",
  "product_description": "string",
  "ad_size": "1024x1024" | "1024x1536" | "1536x1024"
}
```

**Response:**
```json
{
  "ad_id": 1,
  "headline": "string (3-7 words)",
  "marketing_text_50_words": "string (exactly 50 words)",
  "image_url": "string",
  "zip_url": "string"
}
```

## Validation

- All three fields (`product_name`, `product_description`, `ad_size`) are required
- All fields must be non-empty strings
- `ad_size` must be exactly one of: `1024x1024`, `1024x1536`, `1536x1024`
- Invalid requests return 400 with a clear error message

## Deployment

Compatible with Render. Use:
```bash
gunicorn app:app --timeout 600
```

