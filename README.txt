ACE BACKEND (Render / Flask)

Endpoints
- GET /health
- POST /generate  JSON: { product, description, size, ad_index }
- GET /file/<name>.jpg
- GET /file/<name>.zip

Environment (Render)
OPENAI_API_KEY=...
OPENAI_IMAGE_MODEL=gpt-image-1
OPENAI_TEXT_MODEL=gpt-4.1-mini
FRONTEND_URL=https://ace-advertising.agency
BACKEND_URL=https://ace-backend-k1p6.onrender.com  (optional, used to build absolute URLs)
PORT=10000

Start command (Render):
gunicorn app:app --timeout 600
