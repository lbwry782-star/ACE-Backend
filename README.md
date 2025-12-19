# ACE Backend — V3

Fixes in this version:
- Image sizes aligned to OpenAI image endpoint supported values for gpt-image-1:
  1024x1024, 1024x1536, 1536x1024
- Frontend dropdown updated accordingly.
- Backend maps any incoming 1024x1792/1792x1024 to supported equivalents.

Render start command:
gunicorn app:app --timeout 600

Recommended Render env:
FRONTEND_URL=https://ace-advertising.agency
WEB_CONCURRENCY=1
