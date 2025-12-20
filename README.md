# ACE — V7

Changes:
- After ad #3, the Generate button becomes disabled and shows **CONSUMED** (no restart).
- Headline enforcement: 3–7 words, includes product name, regenerated if too similar to description.
- UI shows headline prominently.

Render start command:
gunicorn app:app --timeout 600
