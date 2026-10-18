# Syntactic-complexity convergence

This subproject provides one reproducible path from dialogue turns to
dependency-based complexity, speaker convergence tests, linear CKA, and
length-controlled human/LLM detection. It is self-contained; generated data
and model weights are not stored in this folder.

## Datasets

The intended corpus set and Stanza language codes are:

| Canonical name | Common aliases | Language |
|---|---|---|
| `A2` | `a2` | German (`de`) |
| `Hazumi` | `hazumi` | Japanese (`ja`) |
| `BTSJ` | `btsj` | Japanese (`ja`) |
| `Candor` | `combined_candor`, `candor` | English (`en`) |
| `Maptask` | `maptask` | English (`en`) |
| `LLM_EN` | `llm_english` | English (`en`) |
| `LLM_DE` | `LLM_GERMAN`, `llm_german` | German (`de`) |

Use the canonical name in output filenames and analysis notes. The aliases
above only normalize corpus naming; point each command at the actual file in
your workspace. Keep human and LLM dialogue splits and IDs separate.

## Setup

From the project root:

```bash
python3 -m pip install -r syntactic_complexity_convergence/requirements.txt
python3 -c "import stanza; stanza.download('en')"
python3 -c "import stanza; stanza.download('de')"
python3 -c "import stanza; stanza.download('ja')"
```

Use Python 3.10 or newer. The scripts use the built-in modern type syntax,
without postponed-annotation imports.

Stanza models are large and are downloaded separately. Download only the
language models needed for a run. The metric calculation supports CPU by
default; use `--gpu` only when Stanza can access a configured GPU.

## 1. Preprocess dialogue turns

The canonical CSV columns are `dialogue_id,speaker,turn_index,text`.
CSV columns are inferred for common forms (`Dialogue_id`/`id`, `speaker`,
`turn_id`, `utterance`/`text`); override them for corpus-specific schemas.
Within each dialogue, the first observed participant is relabeled
`initiator`, the second `follower`. If input contains one role per file, pass
`--role` to each file and combine the canonical CSVs before metric calculation.

Example for a CSV with `id`, `speaker`, `turn_id`, and `utterance`:

```bash
python3 -m syntactic_complexity_convergence.preprocess \
  --input English_data/combined_candor.csv \
  --output data/Candor_turns.csv \
  --id-column id --speaker-column speaker --turn-column turn_id --text-column utterance
```

Example for a role-specific legacy JSON mapping dialogue IDs to lists of
utterances:

```bash
python3 -m syntactic_complexity_convergence.preprocess \
  --input json_combined_con/Candor_initiator.json --role initiator \
  --output data/Candor_initiator_turns.csv
```

When combining role-specific outputs, concatenate their rows into one CSV
before the next step. A JSON input may also be a list of records or an object
mapping IDs to lists of `{speaker, text}` turn objects. Empty/incomplete
records are dropped and counted in the command output.

## 2. Calculate syntactic complexity

```bash
python3 syntactic_complexity_convergence/calculate_syntactic_complexity.py \
  --input data/Candor_turns.csv --output data/Candor_metrics.csv --language en
```

The executable [`calculate_syntactic_complexity.py`](calculate_syntactic_complexity.py)
imports `stanza` directly, creates `stanza.Pipeline` with tokenization,
part-of-speech tagging, lemmatization, and dependency parsing, then calculates
the metrics for every turn. The metric functions in
[`calculate_metrics.py`](calculate_metrics.py) retain the formulas and public
helper names from the existing project implementation (`tree_depth`,
`dependency_length`, `calculate_complexity`, `long_dependency_ratio`,
`sentence_syncom_metrics`, `calculate_syncom`, and `summarize_syncom`).
The command-line script uses these shared functions rather than duplicating
the calculations.

Each utterance is parsed with Stanza. The result includes `word_count`,
`syn_value`, `dependency_length`, `tree_depth`, and
`complex_dependency_ratio`. `syn_value` retains the prototype definition:
one half whitespace-token count per distinct head ID (including the root) plus
one half maximum dependency-tree depth:

```text
syn_value = 0.5 * (whitespace_token_count / number_of_distinct_head_IDs)
          + 0.5 * maximum_dependency_tree_depth
```

For example, with 6 whitespace tokens, 4 distinct head IDs (including root
ID 0), and a tree depth of 3, `syn_value = 0.5 * (6 / 4) + 0.5 * 3 = 2.25`.
`dependency_length` is the mean absolute token-index distance to the head,
excluding root arcs. `complex_dependency_ratio` is the number of non-root
arcs with distance greater than 2 divided by all non-root arcs. The
implementation is in [`calculate_metrics.py`](calculate_metrics.py), in
`dependency_metrics`.

## 3. Label dialogue phases

Phase labels are assigned independently within each dialogue and speaker, in
turn order, using contiguous near-equal-sized splits. The phase step adds both
8- and 16-phase categories by default:

```bash
python3 -m syntactic_complexity_convergence.phase_labels \
  --input data/Candor_metrics.csv --output data/Candor_metrics_phased.csv
```

The output adds integer `phase_8` and `phase_16` columns. For a different
resolution, pass for example `--phases 8,16,32`. Analyses below also split
turns internally if a matching phase column is not present.

## 4. Test syntactic convergence against chance

```bash
python3 -m syntactic_complexity_convergence.convergence \
  --input data/Candor_metrics.csv --output-prefix results/Candor_convergence \
  --phases 8 --min-turns 16 --permutations 9999 --seed 42
```

For each paired dialogue, each speaker's ordered turns are split into equal
contiguous phases. The analysis computes the absolute difference between
speaker phase medians for each metric, then fits a slope to the mean gap over
phases. If the input has a matching `phase_8` or `phase_16` column, that
category is used. A negative slope is convergence. The one-sided Monte Carlo test
independently shuffles phase order within each dialogue, preserving each
dialogue's phase-gap values while removing their chronological ordering.
Reported p-values use the plus-one correction and are also adjusted across
the requested metrics with Benjamini-Hochberg.

The test output also retains the earlier perturbation-test method: strict
and loose ratios count dialogues whose absolute speaker-gap sequence ends
lower than it starts, with the strict ratio additionally requiring at least
two phase-to-phase decreases. Its null independently
shuffles each speaker's turn order within each dialogue before recalculating
those ratios. The output includes the null mean/standard deviation, a
plus-one-corrected one-sided permutation p-value, Cohen's *d*, and a
dialogue-bootstrap 95% interval for the observed-minus-null strict ratio.
These strict-ratio p-values are adjusted separately across metrics. Use
`--bootstrap-samples` to set the number of bootstrap draws and `--n-jobs` to
select parallel workers (`1` disables parallelism).

The output is `*_phases.csv` (mean gaps) and `*_tests.csv` (slope and
permutation tests, strict/loose ratios, effect sizes, and intervals).

This randomization test addresses whether the observed *temporal ordering*
of gaps is more convergent than a randomized ordering; it does not establish
causality or generalize beyond the sampled dialogues.

Set `--phases 16 --min-turns 32` to repeat the analysis at 16 phases.

## Reuse of the existing analysis code

The active analysis logic from the earlier `test_syntactic_complexity`
research code is integrated into one reusable workflow: the Stanza metric
functions and their original formulas are in `calculate_metrics.py`; turn
normalization is in `preprocess.py`; 8/16-phase labeling is in
`phase_labels.py`; convergence and perturbation tests are in `convergence.py`
and `statistics.py`; paired-speaker linear CKA is in `cka.py`; and
FourierGPT/NLL/surprisal controls are in `fouriergpt.py`. The original research
folder is left unchanged. Its hard-coded experiment loops, scratch outputs,
checkpoint copies, and unrelated demos are not duplicated inside this clean
subproject.

## 5. Linear CKA analysis

```bash
python3 -m syntactic_complexity_convergence.cka \
  --input data/Candor_metrics.csv --output-prefix results/Candor \
  --phases 8 --min-turns 16 --points-per-phase 20 --permutations 9999
```

Turn-level metrics are standardized within the input corpus. Per dialogue and
phase, each speaker's feature trajectory is linearly interpolated to the same
number of points; centered linear CKA is calculated across the four metrics.
The output is CKA **distance** (`1 - similarity`), so a negative distance
slope indicates increasing alignment. The phase-order permutation test uses
the same null as the convergence analysis. The minimum is two turns per
speaker per phase; raise `--min-turns` to match the intended phase resolution.
For 16 phases, use at least `--min-turns 32`.

## 6. FourierGPT, surprisal, and length controls

The detection module has no import-time dependency on the separate FourierGPT
scripts: it can either consume their long-form Fourier CSV (`sid,freq,power`
or `dialogue_id,freq,power`) or derive one-sided Fourier magnitude spectra
directly from ID-keyed per-token NLL series. NLL scoring itself must be
performed by a causal language model; the upstream FourierGPT `run_nll.py`
is one option. Its output rows are positional, so preserve the input dialogue
order and attach each dialogue ID before using this module. The internal
`rfft` transform matches FourierGPT's positive-frequency magnitude spectrum.

Generate length sidecars from each metrics file
using Stanza token and turn counts:

```bash
python3 -m syntactic_complexity_convergence.dialogue_lengths \
  --input data/Candor_human_metrics.csv --output data/Candor_human_lengths.csv
python3 -m syntactic_complexity_convergence.dialogue_lengths \
  --input data/LLM_EN_metrics.csv --output data/LLM_EN_lengths.csv
```

Supply separate length sidecars with an ID and
one of `length`, `word_count`, `token_count`, or `turn_count`. These should
contain the actual dialogue length, not the number of Fourier bins.

Surprisal baselines accept either ID-tab-series files or the existing semcom
text format (`index:<dialogue_id>`, followed by `surprisal`, then token values).
The parser combines utterance-level values by dialogue and ignores the
`shannon` and `renyi` sections. Raw NLL series use the ID-tab-series format:
`dialogue_id<TAB>nll_1 nll_2 ...`.
The script compares fixed-width, linearly resampled feature vectors using a
stratified logistic-regression split. It reports Fourier/NLL/surprisal
classifiers and their length-only controls. By default, the data are balanced
within pooled length-quantile bins before splitting, so the feature model is
length-controlled; the length-only row is a control on that same matched
sample. The surprisal baseline uses scored-sequence length as its control.

```bash
# Existing FourierGPT spectra:
python3 -m syntactic_complexity_convergence.fouriergpt \
  --human-fft path/to/human_fft.csv --llm-fft path/to/llm_fft.csv \
  --human-lengths path/to/human_lengths.csv --llm-lengths path/to/llm_lengths.csv \
  --human-semcom path/to/human_semcom.txt \
  --llm-semcom path/to/llm_semcom.txt \
  --output results/LLM_EN_detection.csv --seed 42

# Or derive spectra from ID-keyed language-model NLL series:
python3 -m syntactic_complexity_convergence.fouriergpt \
  --human-nll path/to/human_nll.txt --llm-nll path/to/llm_nll.txt \
  --human-lengths path/to/human_lengths.csv --llm-lengths path/to/llm_lengths.csv \
  --output results/LLM_EN_nll_detection.csv --seed 42
```

Choose either an existing Fourier CSV pair or an NLL pair, and optionally add
a surprisal pair. At least one feature pair is required. Reduce
`--length-bins` if length matching leaves too few samples.
The output contains accuracy, precision, recall, F1, and ROC-AUC. This is a
supervised detector; evaluate on a held-out corpus or model family before
making claims about generalization.

## Testing

Run the lightweight tests (they do not need Stanza model downloads):

```bash
python3 -m unittest discover -s syntactic_complexity_convergence/tests
```
