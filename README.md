# ACE Backend (FINAL)

## Deploy (Render)
- Build: `pip install -r requirements.txt`
- Start: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 180`

## Environment variables
- `OPENAI_API_KEY` (required for real generation)
- Optional:
  - `OPENAI_TEXT_MODEL` (default: gpt-4.1-mini)
  - `OPENAI_IMAGE_MODEL` (default: gpt-image-1)
  - `MIN_SECONDS_BETWEEN_OPENAI_CALLS` (default: 2.5)
  - `MAX_RETRIES` (default: 6)

## API
- `GET /health`
- `POST /api/generate` JSON: {client_name, prompt, size}
  -> {job_id}
- `GET /api/job/<job_id>` -> status + ad payload when ready
