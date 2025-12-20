# ACE Backend — V4 (Embedded ZIP + Image)

This version fixes 'ZIP content not found' by **removing filesystem dependency**.
The `/generate` response includes:
- `image_data_url` (data:image/jpeg;base64,....)
- `zip_data_url` (data:application/zip;base64,....)

So the frontend can display images and download ZIPs immediately without calling `/zip/...`.

Render start command:
gunicorn app:app --timeout 600

Recommended Render env:
FRONTEND_URL=https://ace-advertising.agency
WEB_CONCURRENCY=1
