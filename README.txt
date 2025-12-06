ACE Backend v3 – sid is optional (ICOUNT friendly)

• /health        → status ok
• /start-session → optional helper when sid is used
• /generate      → { product, size, sid? }

Rules:
- 'product' is required.
- Developer mode: product == "4242" → unlimited, ignores sid.
- If sid is supplied (Stripe-style future flow) → 1 attempt per sid.
- If NO sid (ICOUNT/manual flow) → no server-side limit, request is allowed.

OpenAI:
- Uses client.images.generate (gpt-image-1) without 'n'.
- Uses client.responses.create (gpt-4.1-mini) for 3 variants of headline+copy.
