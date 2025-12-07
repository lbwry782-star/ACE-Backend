ACE Backend (Token + Override) – Updated

Key behaviors:
- /generate always returns HTTP 200 for generation attempts (except for bad input / no token).
- If generation fails (OpenAI error, timeout, JSON issue), response is:
  { "mode": "...", "error": "Generation failed: ...", "ads": [] }
  so the frontend can show an error and allow another attempt.
- Successful TOKEN generation does not fail the HTTP layer and the frontend
  will mark the token as used.
