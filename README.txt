ACE BACKEND V3 — TOKEN v3.2 with IPN

Endpoints:
- GET  /health
- POST /ipn           -> IPN from iCount, creates/refreshes token for a given cp when status=paid
- GET  /token-status  -> Frontend can query if token is valid for a given cp
- POST /generate      -> Generates 3 ads if token/override is valid

Environment variables:
- OPENAI_API_KEY     (required)
- OPENAI_IMAGE_MODEL (default: gpt-image-1)
- OPENAI_TEXT_MODEL  (default: gpt-4.1-mini)
- IPN_SECRET         (optional, HMAC secret used to verify iCount IPN signatures)
- ACE_DEV_SECRET     (optional, shared secret for internal override mode)

Notes:
- Token data is stored in-memory (Python dict). For production you should replace this
  with a persistent store such as Redis or a database.
