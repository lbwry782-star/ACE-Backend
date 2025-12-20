# ACE — V6

Two fixes:
- H01 + H10: 3 ads are generated as 3 sequential attempts (Generate / Generate Again).
- Engine rules are enforced in the prompt: audience -> intent -> 80 physical objects -> A/B -> projection -> HYBRID or side-by-side -> classic background of A -> photorealistic -> no text on image.

429 handling:
- retries + backoff
- sequential attempts reduce burst load

Render start command:
gunicorn app:app --timeout 600
