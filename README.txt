ACE Backend (Render, iCount version)

Endpoints:
- GET /health  -> simple health check
- POST /generate -> returns ZIP file with 3 JPG placeholders + copy.txt

Environment variables to configure on Render:

- FRONTEND_URL          (e.g. https://lbwry782-star.github.io/ACE-Frontend)

No Stripe is used in this version. Payment is handled by iCount and the user
returns to builder.html?paid=1 according to your iCount configuration.
