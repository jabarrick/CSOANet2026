# Clarifications of the registration (https://osf.io/8st47) implemented in the code

The registration text was not reachable while this code was written; the code follows the
summary in CLAUDE.md. Yu answered the points below on 2026-11-01. They belong in the freeze
amendment together with the commit hash.

## Data

| Point | Decision |
|---|---|
| Lee2019-MI runs | A session is both of its runs (offline `train` and online `test`), 100 trials each, so 200 per session and 400 per subject. MOABB 1.7.2's `Lee2019_MI()` loads only the offline run by default, so the loader builds `Lee2019_MI(train_run=True, test_run=True)`. F1a uses all 400 trials; F2 trains on all of session 1 and tests on all of session 2. |
| Documented trials, Lee2019-MI | 400 per subject for F1a. For F2, 200 in each session; both sessions must keep at least 80 %. |
| Documented trials, Cho2017 | 2 × `n_imagery_trials` from the subject's original `.mat` file (struct `eeg`), read after unlock in the exclusion step and never derived from the epochs MOABB returns. A missing field stops the run with an error; nothing is guessed. |
| Exclusion scope | Per cell: F1a and F2 each decide on their own trials. |
| Within-subject validation split | 20 % of session-1 trials with a fresh `numpy.random.default_rng(0)` for each subject, so a subject's split does not depend on which other subjects are included. |

## Recipes

| Point | Decision |
|---|---|
| R4 | R1 without mixup, Gaussian noise, circular shift, random crop and channel dropout, **and without label smoothing** (plain cross-entropy). It keeps R1's AdamW, cosine schedule and early stopping on validation accuracy. |
| R3 batch | 16 in every cell, F2 included. E stays 100 cross / 500 within. |
| "Validation accuracy" for early stopping (R1, R4, T) | Plain accuracy on the validation trials. |
| T optimiser space | 4 choices: Adam (no weight decay), AdamW with wd 0, 1e-4 or 1e-3. |
| T schedules | constant; cosine to lr / 100 over E (as R1); step ×0.1 at 40 % and 80 % of E (as R3). |
| T selection | "by validation accuracy" = early stopping with R1's patience (15 cross / 80 within); "by validation loss" = all E epochs, lowest validation loss. |
| T draws | The same 24 configurations (one rng, seed 0) for all four architectures in a cell, drawn independently for each cell. |
| T score | Cross: train on fold 1's training subjects, score = balanced accuracy on fold 1's validation subjects at the selected checkpoint. Within: for each of the 10 lowest subject IDs, the registered 80/20 split of session 1; score = mean balanced validation accuracy over the 10. Seed 0 only. Ties go to the lowest configuration index. |
| Non-finite loss | A run fails at the first non-finite training loss. Training stops there and the run is repeated with seed + 100. If the retry fails as well, it also stops at the first non-finite loss. The checkpoint selected so far (or the current weights, if none was selected) is then evaluated and kept, and flagged in the CSV. |

## Outcomes

| Point | Decision |
|---|---|
| EEGNet int8 form | Both are reported. The standard (Lawhern) graph is the registered secondary outcome (`bacc_int8`, `agreement`, MACs, RAM). The exact spatial-first rewrite gives extra columns (`*_sf`). |

## Analysis

| Point | Decision |
|---|---|
| H3 overlap | **Equivalent wins.** A CI inside ±2 points is "equivalent", even if it excludes 0. "A better" / "B better" is used only for a CI that excludes 0 and is not inside ±2. |
| H4 | The 6 architecture pairs under T, at the same 99.79 % level as H3, with H3's categories. Kendall's τ is computed between the four architectures' mean accuracies under R1 and under T, one τ per family. |

## Implementation choices not covered above (for Yu's review)

- H1 compares A + R (+ dataset) with A (+ dataset) (3 df). H2 compares A × R (+ dataset) with A + R (+ dataset) (9 df). Both are ML fits with a random intercept per subject. In F1, subject IDs are prefixed with the dataset.
- "Inside ±2 points" is taken strictly: −0.02 < lower and upper < 0.02. Intervals are two-sided t intervals of the per-subject paired differences of seed-averaged balanced accuracy.
- ART: residuals from the full factorial cell means (A × R, × dataset in F1); for each tested effect the residuals plus that effect's estimate are ranked, a mixed model with every fixed term is fitted to the ranks, and the effect is tested with the same LRT as H1 / H2.
- Seeds: `torch.manual_seed(seed)` before model construction for each architecture × recipe × fold or subject. The int8 calibration set (256 training trials) is drawn with `default_rng(seed)`.
- Normalisation gain: per-channel standard deviation over the fold's training trials, which excludes the validation trials.
- Un-augmented training accuracy is computed over all training trials at the selected checkpoint.
