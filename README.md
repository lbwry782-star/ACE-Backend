# ACE Backend

Backend API built with Flask.

## Requirements

- Python 3.8+
- pip

## Instalation

1. Create a virtual environment (recommended):

```bash
python -m venv venv
```

2. Activate the virtual environment:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Server

```bash
python app.py
```

The server will start on `http://localhost:5000`

## API Endpoints

### POST /api/generate

Generate an ad based on product information.

**Request:**
```json
{
  "productName": "string",
  "productDescription": "string"
}
```

**Success Response (200):**
```json
{
  "requestId": "uuid",
  "status": "success",
  "result": {
    "title": "string",
    "summary": "string",
    "files": []
  }
}
```

**Error Response (400/500):**
```json
{
  "status": "error",
  "message": "string"
}
```

### GET /health

Health check endpoint.

**Response (200):**
```json
{
  "status": "ok"
}
```

## CORS

CORS is enabled for localhost development (Vite frontend).

