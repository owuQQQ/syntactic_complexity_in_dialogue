# syntactic_complexity_in_dialogue

Code base for the accepted EMNLP paper.

**Paper:** Yu Wang, Yanran Chen, Yifan Wang, Steffen Eger, and Hendrik
Buschmeier. 2026. [Syntactic Complexity Convergence in Dialogue: Analysis of
the Phenomenon and Application to LLM Detection](https://aclanthology.org/2026.emnlp-main.1239/).
In *Proceedings of the 2026 Conference on Empirical Methods in Natural
Language Processing*, pages 28244–28269. Association for Computational
Linguistics. [PDF](https://aclanthology.org/2026.emnlp-main.1239.pdf) ·
[DOI](https://doi.org/10.18653/2026.emnlp-main.1239).

## Syntactic-complexity convergence pipeline

The [`syntactic_complexity_convergence/`](syntactic_complexity_convergence/)
project provides dialogue preprocessing, dependency-based syntactic
complexity metrics, 8- and 16-phase assignment, permutation tests for
convergence, linear CKA analysis, and FourierGPT-style human/LLM detection
with surprisal and length-only baselines.

Start with the
[subproject README](syntactic_complexity_convergence/README.md) for
installation, supported datasets (A2, Hazumi, BTSJ, Candor, Maptask, LLM_EN,
and LLM_DE), input formats, and reproducible command examples.

The repository contains analysis code only; corpus data, language-model
weights are not included.
