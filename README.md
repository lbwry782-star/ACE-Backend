# ACE Backend

Flask backend for ACE ad generation system.

## Local Development

### Windows

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

The server will start on `http://localhost:5000`

## API Endpoints

### GET /health
Health check endpoint.
Returns: `{"ok": true}`

### POST /api/generate
Generate 3 ads for a product.

**Request Body:**
```json
{
  "product_name": "string",
  "product_description": "string",
  "size": "1024x1024" | "1024x1536" | "1536x1024",
  "run_index": 1
}
```

**Response:**
```json
{
  "ads": [
    {
      "ad_index": 1,
      "marketing_text": "...",
      "image_jpg": "data:image/jpeg;base64,..."
    },
    ...
  ],
  "run_index": 1
}
```

### GET /api/download
Download ZIP file for a specific ad.

**Query Parameters:**
- `ad_index`: 1-3
- `run_index`: integer
- `product_name`: string

**Response:** ZIP file with `ad_X.jpg` and `ad_X.txt`

## Render Deployment

The app is configured to run on Render with:
- Gunicorn server
- 600 second timeout
- 2 workers, 2 threads

No environment variables required for basic operation.

