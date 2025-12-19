# ACE Backend — V2

Fixes:
- CORS: uses FRONTEND_URL if set; otherwise allows all origins.
- OpenAI text uses chat.completions (no responses API).
- Sizes: 1024x1024 / 1024x1792 / 1792x1024 (legacy sizes mapped).

Render start command:
gunicorn app:app --timeout 600

Recommended Render env:
FRONTEND_URL=https://ace-advertising.agency
WEB_CONCURRENCY=1
