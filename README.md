# ACE Backend (NEW)

## Render settings
- Build: `pip install -r requirements.txt`
- Start: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4`
- Health check path: `/health`

## Env vars
- `OPENAI_API_KEY` (optional)
- `USE_MOCK` (default `1`): keep `1` until you deposit money / want real OpenAI calls.
- `OPENAI_IMAGES` (default `0`): turn on only when ready (image calls are heavy and trigger 429 easily).
- `MAX_ADS_PER_SESSION` (default `3`)
- `DEFAULT_RETRY_AFTER` (default `30`)
