ACE Backend v6

Changes in this version:

1) DOWNLOAD FIX
   • /download now uses request_id when available, but if request_id is unknown
     it falls back to the most recent generation (last_request).
   • This prevents the "Unknown request_id" error after a successful generate.

2) NO HYBRID, NO LOGOS
   • Image prompt does NOT use the word "hybrid" and does NOT mention ACE rules.
   • Image prompt explicitly forbids logos, brand marks, app UI, watermarks.
   • Only one short English headline (3–7 words) is allowed inside the image,
     and no other text.

3) TEXT GENERATION
   • Text prompt forbids brand names and trademarks in the copy.
   • On any error, three safe placeholder variants are used.

Endpoints:
   GET  /health
   POST /start-session
   POST /generate   { product, size, sid? }
   GET  /download?request_id=...&index=1..3\n