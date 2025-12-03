
# ACE Backend (Fixed)

This backend is designed to work with the latest ACE Builder frontend.

## Endpoints

- `GET /health` → `{ "status": "ok" }`
- `POST /generate`
  - Request JSON:
    - `product` (string, required)
    - `description` (string, optional)
    - `size` (string, e.g. `"1024x1024"`)
  - Response JSON:
    - `{ "ads": [ { "headline", "marketing_text", "image_base64", "size" }, ... ] }`

The token / 4242 logic is implemented **only in the frontend**. The backend simply
receives a request and returns 3 ad variations.
