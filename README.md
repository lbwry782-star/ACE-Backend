# ACE Backend

Flask backend for ACE image + text generation.

## Endpoints

- `GET /health` → `{ "status": "ok" }`
- `POST /generate` → JSON with 3 ads (image_base64, headline, copy)

Configure these environment variables on Render:

- `OPENAI_API_KEY`
- `OPENAI_IMAGE_MODEL` (default: gpt-image-1)
- `OPENAI_TEXT_MODEL` (default: gpt-4.1-mini)
- `FRONTEND_URL` (optional CORS allow origin)
