ACE BACKEND (Flask)

Start command on Render:
    python app.py

Environment variables required:
    OPENAI_API_KEY      = your OpenAI API key
    OPENAI_IMAGE_MODEL  = gpt-image-1
    OPENAI_TEXT_MODEL   = gpt-4.1-mini
    FRONTEND_URL        = https://lbwry782-star.github.io

Endpoints:
    GET /health   -> {"status":"ok"}
    POST /generate
        JSON body:
            {
              "product": "Product name in English",
              "description": "Optional description in English",
              "size": "1024x1024" | "1024x1536" | "1536x1024"
            }

        Returns:
            {
              "success": true,
              "ads": [
                {
                  "headline": "...",
                  "copy": "...",
                  "image_b64": "base64-encoded PNG"
                },
                ...
              ]
            }
