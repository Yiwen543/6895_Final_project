# Cathey — Final Presentation Materials

EECS 6895 Final Project · Columbia University · May 5, 2026.

This folder contains the slide deck (LaTeX/Beamer) and the speaker scripts for the final onsite presentation.

## Files

| File | Purpose |
|---|---|
| `slides.tex` | 15-slide Beamer deck. Compile on Overleaf with **pdfLaTeX**. Aspect ratio is 16:9. |
| `speaker_scripts.md` | Three speaker scripts (~3 min 30 s each) plus a Q\&A buffer. |

## Time budget

| Section | Slides | Speaker | Time |
|---|---|---|---|
| Motivation, novelty, system, stack | 1–5 | Speaker 1 | 3 min 30 s |
| Data and methods | 6–10 | Speaker 2 | 3 min 30 s |
| Experiments, demo, conclusion | 11–15 | Speaker 3 | 3 min 30 s + demo |
| Q&A | — | All | ~1 min |

Total: ~12 minutes, including demo and group Q&A.

## How to use

1. Open `slides.tex` on Overleaf. Replace the three `Team Member A/B/C` placeholders with the real names and unis.
2. Add your group photo, university logo, or Columbia EECS logo to the title slide if you want.
3. Each speaker reads their part of `speaker_scripts.md`. The bracketed `[Slide N — ...]` cues match the slide numbers in `slides.tex`.
4. The demo slide (Slide 14) is intentionally short. Plan for the live demo to take ~1 min 30 s. Keep a fallback recorded video in case the Wi-Fi or Bluetooth speaker fails.

## Customization tips

- If the Beamer theme `Madrid` does not match your taste, try `Berlin`, `Frankfurt`, or `metropolis`. Just change the `\usetheme` line.
- The block diagram on Slide 4 is currently text-only inside an `\fbox`. If you have time, draw a real diagram in TikZ or Figma and replace it with `\includegraphics`.
- For the demo slide, you can also add screenshots of `cathey_memory/skills.json` or the `benchmark_results.md` table.
