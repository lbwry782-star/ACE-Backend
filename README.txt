ACE Backend (Render, OpenAI image version)

Endpoints:
- GET  /health    -> simple health check (also shows engine + model)
- POST /generate  -> returns JSON with 3 ads, each including base64 image + copy text.

Environment variables to configure on Render:

- OPENAI_API_KEY      (required)  Your OpenAI API key for this backend.
- FRONTEND_URL        (optional)  e.g. https://lbwry782-star.github.io/ACE-Frontend
- OPENAI_IMAGE_MODEL  (optional)  default: gpt-image-1
- OPENAI_TEXT_MODEL   (optional)  default: gpt-4.1-mini

No Stripe is used in this version. Payment is handled by iCount and the user
is redirected to builder.html according to your iCount configuration.
