# syntactic_complexity_in_dialogue

Code base for the accepted EMNLP paper.

## Syntactic-complexity convergence pipeline

The [`syntactic_complexity_convergence/`](syntactic_complexity_convergence/)
subproject provides dialogue preprocessing, dependency-based syntactic
complexity metrics, 8- and 16-phase assignment, permutation tests for
convergence, linear CKA analysis, and FourierGPT-style human/LLM detection
with surprisal and length-only baselines.

Start with the
[subproject README](syntactic_complexity_convergence/README.md) for
installation, supported datasets (A2, Hazumi, BTSJ, Candor, Maptask, LLM_EN,
and LLM_DE), input formats, and reproducible command examples.

The repository contains analysis code only; corpus data, language-model
weights, and generated results are not included.
