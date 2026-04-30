# Nova — Final Project Report (LaTeX)

EECS 6895 · Columbia University · IEEE conference style.

## Files

| File | Purpose |
|---|---|
| `main.tex` | The full IEEE double-column report. Compile on Overleaf with **pdfLaTeX**. |

## How to compile on Overleaf

1. Create a new project on Overleaf.
2. Upload `main.tex`.
3. Set the compiler to **pdfLaTeX** (Project > Settings > Compiler).
4. Click **Recompile**.

No external `.bib` file is needed — the bibliography is embedded inline using `thebibliography`.

## Customization checklist

Before submitting, please update:

- [ ] **Title block (line ~30)**: replace `Team Member A / B / C` with the real names and unis.
- [ ] **Email addresses**: replace `uni1@columbia.edu`, `uni2@columbia.edu`, `uni3@columbia.edu` with the real ones.
- [ ] **Acknowledgment section**: confirm the professor's name and any TAs you want to thank.
- [ ] **Numbers in the experiments**: if you re-run the benchmark on different hardware or with new training data, update Tables I and II in Section V.
- [ ] **Reference [1]**: the FTC press release on Ring is illustrative. Swap it for a more recent privacy reference if you have one.
- [ ] **Page count**: the rubric says less than 10 pages. The current draft should be under 10 with the IEEE conference template; verify after first compile.

## Grading rubric mapping

The report is structured to match the rubric weights from the assignment:

| Rubric section | Weight | Where in `main.tex` |
|---|---|---|
| Abstract | — | line ~58 |
| Introduction | 5% | Section I |
| Related Work | 5% | Section II |
| Data | 10% | Section III |
| Methods | 25% | Section IV |
| System Overview | 25% | Section V |
| Experiments | 20% | Section VI |
| Conclusion | 5% | Section VII |
| Writing / Formatting | 5% | applies to whole document |

## Style notes

- Sentences are kept short and use simple vocabulary by design — the report is meant to read like a clearly written student paper, not like a polished research paper. Please do not over-edit it into more complex sentences.
- Technical terms (LoRA, GGUF, RAG, VAD, etc.) are used when needed; everything else is plain English.
- All numbers are taken from the actual files in the repo (`benchmark_results.md`, `finetune/train_data.py`, `config.py`).
