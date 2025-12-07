
# ACE Backend — OpenAI Engine Version

This backend is ready for **real** image + text generation using the OpenAI API.

## Endpoints

- `GET /health` → `{"status": "ok"}`
- `POST /generate`

### /generate request body

```json
{
  "product": "Product name (string)",
  "description": "One–two lines describing the product",
  "size": "1024x1024" | "1024x1792" | "1792x1024",
  "token": true
}
```

Notes:

- `size` must be exactly one of the three values above (from the MASTER 1.4 document).
- `token` must be `true` for normal users after payment.
- If `product == "4242"` → OVERRIDE MODE for unlimited testing (ignores token).

### /generate response

```json
{
  "ads": [
    {
      "image_base64": "...",   // JPEG, ready to display or save as ad_image.jpg
      "headline": "...",       // appears on the image itself
      "copy": "...",           // 50-word marketing text
      "objective": "...",
      "mode": "fusion" | "placement"
    },
    { "... ad 2 ..." },
    { "... ad 3 ..." }
  ]
}
```

The FRONTEND is responsible for:
- Converting `image_base64` to a visible image.
- Saving `ad_image.jpg` + `ad_text.txt` inside ZIP files for download.

## OpenAI models

The backend uses:

- Text model (ACE Engine logic): `OPENAI_TEXT_MODEL` (default `gpt-4.1-mini`)
- Image model: `OPENAI_IMAGE_MODEL` (default `gpt-image-1`)

It calls the **Responses API** for text and the **Images API** for images.
The ACE Engine prompt is implemented in `_call_ace_engine()` and follows the
Engine 3.2 document: Audience → 3 Targets → A/B objects → Fusion/Placement.

## Environment variables (Render)

Set these in your Render service:

- `OPENAI_API_KEY`      = your secret OpenAI key
- `OPENAI_TEXT_MODEL`   = gpt-4.1-mini   (or another text model)
- `OPENAI_IMAGE_MODEL`  = gpt-image-1    (or dall-e-3 if you prefer)
- `PORT`                = 10000 (Render sets this automatically, but keep the default code)

You **do not** need to expose any secrets to the FRONTEND. The FRONTEND only calls this backend.

## Deploy on Render

1. Create a new Web Service from this folder (GitHub repo or direct upload).
2. Set **Start Command** to:

   ```bash
   gunicorn app:app
   ```

3. Add the environment variables above.
4. Deploy. When the service URL is ready, point your FRONTEND `/generate` calls to it.

This backend is fully compatible with the existing ACE FRONTEND you already use.
