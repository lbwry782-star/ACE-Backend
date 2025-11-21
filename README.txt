ACE Backend (Render, iCount version – marketing text demo)

Endpoints:
- GET /health  -> simple health check
- POST /generate -> returns ZIP file with:
    * ad_1.txt
    * ad_2.txt
    * ad_3.txt
    * copy.txt

Each ad_*.txt contains a different English marketing text variation based on the
product and description received from the Builder. No images are generated in
this demo version.
