# ACE Backend — V5 (Single Ad + Stable ZIP)

Why V4 got 502:
- Embedding 3 images + 3 ZIPs as base64 made the /generate response huge and heavy.

What V5 does:
- Generates ONLY 1 ad (as requested).
- Returns image as `image_data_url` (base64).
- Returns `zip_url` that downloads ZIP from an in-memory cache (TTL 15 minutes).
- Includes robust 429 handling with retries/backoff.

Render start command:
gunicorn app:app --timeout 600

Env:
FRONTEND_URL=https://ace-advertising.agency
WEB_CONCURRENCY=1
