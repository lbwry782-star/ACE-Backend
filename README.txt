ACE Backend (Render, iCount version – CORS fixed)

Endpoints:
- GET /health  -> simple health check
- POST /generate -> returns ZIP file with 3 JPG placeholders + copy.txt

CORS is configured to allow all origins ("*") to avoid header issues caused
by malformed FRONTEND_URL values in environment variables.
