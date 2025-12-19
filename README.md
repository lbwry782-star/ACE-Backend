# ACE Backend (Render-ready)

## Endpoints
- `GET /health` -> `{ "status": "ok" }`
- `POST /generate` -> generates 3 ads sequentially; handles 429 with retries (2s, 5s, 10s)
- `GET /file/<filename>` -> serves generated files
- `GET /zip/<job_id>/<ad_number>` -> downloads ZIP with `ad_<n>.jpg` + `ad_<n>.txt`

## Allowed sizes
- 1024x1024
- 1024x1536
- 1536x1024

## ENV
- OPENAI_API_KEY (required)
- OPENAI_IMAGE_MODEL (default: gpt-image-1)
- OPENAI_TEXT_MODEL (default: gpt-4.1-mini)
- FRONTEND_URL (recommended for CORS)
- PORT (Render sets automatically)

## Render start command
gunicorn app:app --timeout 600
