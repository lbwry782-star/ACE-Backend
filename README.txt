ACE Backend – Amir Gottlieb Demo (Render, iCount version)

This backend is a fixed demo for the ACE website.

Endpoints:
- GET /health
  -> {"status":"ok","time":"...","demo":"amir_gottlieb_v1"}

- POST /generate
  -> Always returns the same ZIP file with:
     * ad_1.jpg, ad_2.jpg, ad_3.jpg   (static 1080x1080 ads)
     * ad_1.txt, ad_2.txt, ad_3.txt   (English marketing texts)
     * copy.txt                       (short description of the package)

The /generate endpoint ignores incoming JSON. It is meant as a stable demo
for the website, not as a generic engine. CORS is fully open ("*") so the
GitHub Pages frontend can call this backend without issues.
