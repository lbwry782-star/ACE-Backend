ACE Backend (Prototype)

Endpoints:
- GET /health      → simple health check
- POST /generate   → create 3 ads (image + headline + 50-word copy)

TOKEN & OVERRIDE (summary):
- Product == "4242" in the request → does NOT generate ads, but activates OVERRIDE mode.
- OVERRIDE mode (strongest): allows unlimited creations, survives page refresh as long as the server stays up.
- If OVERRIDE is not active, a request must include "token": true in the JSON body to be allowed.
- Otherwise, /generate returns 403 (blocked).

Environment variables:
- OPENAI_API_KEY      → required
- OPENAI_IMAGE_MODEL  → default: gpt-image-1
- OPENAI_TEXT_MODEL   → default: gpt-4.1-mini
