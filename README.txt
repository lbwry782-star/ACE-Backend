ACE Backend (Render, iCount version – robust /generate)

Endpoints:
- GET /health  -> simple health check + 'pil_available' flag
- POST /generate -> returns ZIP file with either:
    * 3 JPG placeholders + copy.txt (if Pillow is available)
    * OR 3 tiny text placeholders + copy.txt (if Pillow is not available)

CORS is configured to allow all origins ("*") so that GitHub Pages frontend
can always call this backend without header issues.
