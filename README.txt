ACE Backend — ENGINE V0.5 (Render, iCount version)

This backend implements the ENGINE V0.5 logic for the ACE / ADesk demo,
with a focus on structure and copy rather than final photography.

Endpoints
---------
GET /health
    → { "status": "ok", "time": "...", "engine": "ENGINE_V0.5" }

POST /generate
    Input JSON (any of these keys):
        {
          "product": "...",
          "description": "...",
          "size": "1080x1350"
        }

    Output:
        A ZIP file containing:
            ad_1.jpg
            ad_1.txt
            ad_2.jpg
            ad_2.txt
            ad_3.jpg
            ad_3.txt

    Current behaviour:
        • Derives three simple persona states from the product/description.
        • For each persona, assigns a psychological goal (clarity, confidence, calm).
        • Builds a COPY line (headline, 3–7 words) for each variation.
        • Creates a minimal photographic-style placeholder image:
              - Soft colour background (no hard geometric primitives).
              - COPY placed on the image.
        • Generates a 50-word English marketing paragraph per variation,
          addressing the persona and goal (never describing shapes).
        • Packs all files into a single ZIP and returns it.

IMPORTANT
---------
• This version does NOT yet implement the full hybrid-object visual engine.
  Visuals are placeholders only, intended to be replaced later with real
  photographic hybrids selected from a 100-object library.

• COPY (headline) appears ONLY on the image.
  The 50-word marketing text is saved in the corresponding .txt file.
