ACE Backend v4 – robust against text-model errors

• /generate:
  - 'product' required, 'size' optional (default 1024x1024), 'sid' optional.
  - If sid is provided → 1 attempt per sid. If not → unlimited (ICOUNT flow).
  - Developer mode: product == "4242" → ignores sid and limits.

• Image generation:
  - Uses client.images.generate(model=OPENAI_IMAGE_MODEL, size=...).
  - If images fail → 500 with clear error.

• Text generation:
  - Tries client.responses.create(model=OPENAI_TEXT_MODEL, input=...).
  - If anything fails (API error, parsing etc.) → falls back to safe placeholders.
  - This guarantees that a text error will NOT cause HTTP 500 for the user.
