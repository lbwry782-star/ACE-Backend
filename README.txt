ACE Backend v2 (no deprecated OpenAI parameters)

Endpoints:
- GET  /health
- POST /start-session  { sid }
- POST /generate       { product, size, sid? }
- GET  /download?request_id=...&index=1..3

Token system:
- One attempt per sid (unless product == "4242" developer mode).
- Developer mode bypasses sid and attempt limits.

OpenAI:
- Uses OPENAI_IMAGE_MODEL (gpt-image-1) via client.images.generate without 'n'.
- Uses OPENAI_TEXT_MODEL (gpt-4.1-mini) via client.responses.create.
