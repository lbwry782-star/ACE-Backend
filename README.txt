ACE Backend (Render, ultra-stable text-only demo)

Endpoints:
- GET /health  -> simple health check
- POST /generate -> returns ZIP file with:
    * 3 small text files (ad_1.txt, ad_2.txt, ad_3.txt)
    * copy.txt with summary

No Pillow / image generation is used in this demo, to guarantee that the
endpoint never crashes due to missing native libraries. This is only for
testing the full payment → builder → ZIP flow.
