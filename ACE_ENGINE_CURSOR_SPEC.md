0) Non-Negotiable Rules

Follow H01–H08 exactly. No interpretation, no shortcuts. 



Decisions are based on strict visual/geometric shape similarity only. No conceptual closeness. 


App is English only. 



Builder + Preview: black background, no extra text beyond what is specified. 



External errors (Network/OpenAI): handle cleanly without special user messaging. 



1) Environment (reference)

OPENAI_IMAGE_MODEL = gpt-image-1

OPENAI_TEXT_MODEL = gpt-4.1-mini

PORT = 10000

Start: gunicorn app:app --timeout 600 



2) Input Contract

User provides only:

product_name

product_description

ad_size

Allowed sizes (OpenAI standard only): 



1024x1024

1024x1536

1536x1024

Invalid size → reject (400).

3) Output Contract

Return exactly 3 ads, sequentially (3 attempts, not parallel). 




Each ad contains:

generated JPG image

marketing TXT of exactly 50 words (headline excluded) 



ZIP download containing both files (consistent naming: ad_1.jpg + ad_1.txt, etc.) 



4) H01 — Audience Inference

Infer target audience strictly from product name + description only.
Derive: age, lifestyle, needs, knowledge level, pains. No external info. 



5) H02 — Three Advertising Goals

Derive 3 distinct advertising goals from the audience.
Generate 3 ads one after another; each ad uses a different goal; all share the same size. 



6) H03 — Generate Flow (UI State Rules)

Clicking GENERATE starts progress bar and disables button (gray), label becomes GENERATE AGAIN. 



Button becomes active again only when generation ends. 



Progress bar moves at constant speed; reaches 100% at 4 minutes. 


If generation finishes early → bar moves quickly to 100%.

If bar reaches 100% early → wait at 100% until generation ends.

After the 3rd ad (i.e., after 2 times “GENERATE AGAIN”), button becomes disabled/gray with label CONSUMED. 


7) H04 — Core Object Engine (UPDATED)

For each ad:

7.1 Generate object list

Generate a list of 80 physical, real, associative objects based on the ad goal (derived from audience). 



Forbidden: ideas, symbols, abstract/illustrative concepts. Objects must be simple, everyday, familiar. 



Objects should be non-functional / not functionally linked. 


7.2 Pick A and B

A = object with central meaning to the ad goal. 


B = object used for conceptual emphasis (but pairing is still only by shape). 


7.3 No text/logos

Do not pick objects containing text/logos/letters/numbers/external graphics. 



Allowed only if the mark is an inherent physical part of the object (examples: playing cards, dice dots, engraved compass letters). 



If object requires textual interpretation → forbidden. 


7.4 Choose projections (C and D)

Select one projection per object that highlights the dominant projection area:

Projections must be clean silhouettes, no confusing details. 


Dominant projection shape covers most of the projection area. 


If multiple projections exist, choose the one with the largest visible dominant area. 


C (A’s projection) is always chosen first. 


7.5 Compare simplified shapes (E and F)

To compare E and F, simplify C and D while preserving recognizability to an average human eye. 


7.6 Alignment (UPDATED)

You may adjust:

size and angle of C and D,

placement on the surface,

and even proportions,
only as long as the projection shapes remain classic and not distorted. 


7.7 HYBRID vs SIDE BY SIDE

If E and F can reach almost geometric overlap → create HYBRID between C and D. 


HYBRID occurs when F (from B’s D) is perfectly embedded into E (from A’s C). 


Present the HYBRID at an angle that maximizes E+F area visibility while keeping full photographic realism. 


If not almost overlapping but similar → create SIDE BY SIDE: C and D side-by-side at the same angle, highlighting maximal similar area, placed close together. 


7.8 Rejection rules

If there is no similarity between E and F → reject the pair and move to next A/B pair. 


If similarity is not immediate and clear to an average human eye → reject automatically. 


Never force similarity; do not allow pairs only “the engine sees” as similar. 


Do not consider idea/meaning/association/context — only visual shape similarity. 


C and D must not be parts of the same natural object. 


If repeated failures: increase object list size by +10 (90, 100, …). 


8) H05 — Background

Background must always be the classic background of projection C.
C controls lighting/texture/composition/background; D never changes C’s background or lighting. 


9) H06 — Headline inside the image

Headline rules: 


Inside the image on A’s background

Must include product name

3–7 words

Original promise

Not a quote/variation of product description

Appears above/below/next-to projections (never on them), as part of composition

Headline size similar to projections
Ad contains only:

projections (C and D: HYBRID or SIDE BY SIDE)

headline only (no other text)

10) H07 — Full realism

Projections must be realistic photography.
Forbidden: vector/painting/illustration/3D/AI-art effect. 


11) H08 — Marketing text (50 words)

Never on image; always next to it.

Provided for landing pages/social; downloadable with image via ZIP.

Derived from the ad goal.

Exactly 50 words (headline excluded). 

