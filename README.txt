ACE Backend – ENGINE V0.7 (Hybrid Object Engine, OpenAI new SDK)

This backend implements the ENGINE V0.7 logic for ACE / ADesk using the new OpenAI Python SDK.

• 3 ads per request (JPG + 50-word TXT each).
• Hybrid-object photos generated via OpenAI Images (model gpt-image-1) when OPENAI_API_KEY is set.
• COPY (headline) is rendered onto the image at the bottom in a dark translucent strip.
• No other text, no logos, no brands, no people, no CGI.

Endpoints
---------
GET /health
  → { "status": "ok", "engine": "ENGINE_V0.7", "openai_configured": true/false }

POST /generate
  Input JSON (any combination):
    {
      "product": "...",
      "description": "...",
      "size": "1080x1350"
    }

  Output:
    ZIP containing:
      ad_1.jpg, ad_1.txt,
      ad_2.jpg, ad_2.txt,
      ad_3.jpg, ad_3.txt

Notes
-----
• If OPENAI_API_KEY is not configured, the backend falls back to a soft gradient background
  so the endpoint still works (but without a real hybrid photo).
• To get full ENGINE V0.7 behaviour, set OPENAI_API_KEY in Render.
