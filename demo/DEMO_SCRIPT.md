# Demo GIF storyboard — 3 beats, ~30 seconds

Setup: `docker compose up -d`, corpus ingested, Ollama running,
`uv run --group demo streamlit run demo/app.py`. Sidebar visible.

## Beat 1 — Cited answer (pipeline engine)

Type: **"Does insurance pay if I hit a deer?"**
Show: tokens streaming live, then the latency caption, then click open a
citation expander to reveal the actual ISO policy text ("contact with bird
or animal" under other-than-collision). The point on screen: the answer
and its evidence, side by side, citation mechanically verified.

## Beat 2 — ~1s agent refusal (engine toggle → agent)

Flip the sidebar engine toggle to **agent**.
Type: **"What does boat insurance cover?"**
Show: near-instant refusal — caption reads well under 2s and "refused".
The point: the router saw out-of-scope and never spent retrieval or a
14-second generation on it. (Contrast is the whole beat: deer question
took ~15s of honest work; boat question took ~1s of honest refusal.)

## Beat 3 — State filter (AZ question)

Keep agent engine; set the state dropdown to **AZ** (or leave auto-detect
— the question names Arizona anyway).
Type: **"What penalties can an Arizona driver face for driving without
insurance?"**
Show: cited answer ($500–$1,000 fine, suspension, impoundment) with the
AZ New Driver's Guide expander open. The point: state-scoped retrieval
grounds in the right state's document.

Recording notes: 1280×800 window, hide browser chrome, ~2s pause on each
opened expander, no cuts inside token streaming (the stream IS the demo).
