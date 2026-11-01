# CSOANet2026: notes for Claude Code

Work on branch `csoanet-v2-deploy` (or branches cut from it). `main` is the old v1 code; do not touch it.
Everything for the deployment and recipe study lives in `deploy/` and `eeg_project/scripts/models/`.

## Preregistration: hard constraints

The study "Is a shared training recipe neutral? Recipe × architecture interactions in compact
motor-imagery EEG decoders" is registered on OSF: https://osf.io/8st47 (registered 2026-11-01).
The registration is the spec. Code must implement it exactly. If anything in it is ambiguous,
impossible or looks wrong, stop and ask Yu: a change needs an OSF amendment before the
confirmatory runs it affects.

1. **Do not use the confirmatory datasets.** Lee2019-MI and Cho2017 (MOABB `Lee2019_MI`,
   `Cho2017`) must not be downloaded, loaded, inspected, trained on or evaluated on until Yu has
   frozen the confirmatory code at a git tag and says so himself (see the end of this file). This
   holds in every permission mode, auto mode included, and for smoke tests, single folds and
   "quick checks" too.
   - Write and test their loaders against synthetic stand-ins with the documented shapes,
     events and sessions.
   - Every code path that would fetch or read either dataset must refuse to run unless the
     environment variable `CSOANET_CONFIRMATORY_UNLOCK=1` is set, and must raise a clear error
     otherwise. Never set that variable, never pass it on a command line, never write it into a
     file or config, and never work around the check. Only Yu sets it, after the freeze.
   - Do not run any MOABB call, script or test that would download these two datasets.
2. **The four architectures are fixed**: EEGNet-8,2, ShallowConvNet, MI-BMInet, CSOANet-v2 long kernel
   (`v2-longkt`). Do not change `csoanet2026_v2.py`, `mibminet.py`, `baselines.py` model code or
   their registered hyperparameters. New model ideas (a "v3") go in new files on a separate branch
   and are exploratory; they never enter this study.
3. PhysioNet-MI and BCI IV 2a were the exploratory pilot datasets. They are **not** part of the
   registered plan. Keep their code paths working, but do not add them to the confirmatory runs.

### Registered design (summary; the OSF text is authoritative)

- **Datasets and cells**, two-class left vs right hand, MOABB 1.7.2:
  - F1a: Lee2019-MI, 54 subjects, 62 ch, 1000 Hz, cross-subject, 5 subject folds.
  - F1b: Cho2017, 52 subjects, 64 ch, 512 Hz, cross-subject, 5 subject folds.
  - F2: Lee2019-MI, within-subject, session 1 trains, session 2 tests.
- **Preprocessing**: average reference, 1-40 Hz band-pass, resample to 250 Hz, window 0-4 s after
  the cue (Cho2017: 0-3 s), all EEG channels, no artefact rejection beyond MOABB defaults.
  Normalisation: one gain per channel from the training trials of each fold.
- **Splits**: cross-subject, subjects shuffled with seed 0 into 5 folds, 10 % of each fold's training
  subjects held out for validation. Within-subject, a random 20 % of session-1 trials (seed 0) for
  validation.
- **Architectures** (own weight constraints kept under every recipe; kernels per own rule at 250 Hz):
  - EEGNet-8,2: temporal kernel 125, separable kernel 31; dropout 0.25 cross / 0.5 within;
    max-norm 1 on spatial filters, 0.25 on classifier.
  - ShallowConvNet: 40 temporal filters (kernel 25), 40 spatial filters, square, mean-pool 75
    stride 15, log; dropout 0.5; no constraints. (`eeg_project/scripts/models/convnets.py` has a
    `ShallowConvNet`; check it matches before using it.)
  - MI-BMInet: Nk 32, Nf 64; dropout 0.2 cross / 0.5 within; no constraints.
  - CSOANet-v2 long kernel: temporal kernel 125; static gate, bounded terms, ReLU; max-norm 0.25.
- **Recipes**. E and B: cross-subject 100 epochs, batch 64; within-subject 500 epochs, batch 32.
  - R1: AdamW lr 1e-3, wd 1e-3, cosine to 1e-5 over E; mixup α 0.2, label smoothing 0.1,
    Gaussian noise σ 0.02, circular shift ±5 samples, 90 % random crop with zero padding,
    10 % channel dropout; early stopping on validation accuracy, patience 15 (cross) / 80 (within).
  - R2: Adam lr 1e-3 constant; cross-entropy, no augmentation; all E epochs, lowest validation loss.
  - R3: Adam lr 1e-2, eps 1e-7, ×0.1 at 40 % and 80 % of E; batch 16; no augmentation;
    all E epochs, lowest validation loss.
  - R4: R1 without the augmentation package (same optimiser, schedule, selection).
  - Condition T (for H4 only): per architecture and cell, 24 random configurations from the
    registered space (optimiser Adam / AdamW with wd 0, 1e-4, 1e-3; lr log-uniform [1e-4, 1e-2];
    schedule constant / cosine / step; batch 16 / 32 / 64; each augmentation on/off; label smoothing
    0 / 0.1; selection by validation loss or accuracy), scored on the tuning split only (cross: train
    and validation subjects of fold 1; within: session 1 of the 10 lowest subject IDs). The best
    configuration is then run on all folds or subjects with the same seeds. Test data never used.
- **Seeds** 0, 1, 2 for every architecture × recipe × fold or subject.
- **Outcomes**: primary, balanced accuracy per subject on test trials, float32, averaged over seeds.
  Secondary: int8 balanced accuracy (ONNX Runtime static QDQ, per-channel symmetric weights,
  MinMax calibration on 256 training trials); float32-int8 prediction agreement; un-augmented
  training accuracy and validation accuracy at the selected checkpoint; params, MACs, peak int8 RAM.
- **Exclusions**: a subject only if MOABB cannot load it or < 80 % of documented trials remain,
  decided before any training and listed. A run with non-finite loss is repeated once with
  seed + 100; if it fails again its accuracy is kept as is.
- **Analysis** per family (F1 pooled with dataset as fixed effect; F2): linear mixed model,
  random intercept per subject, ML fit. H1: LRT of recipe main effect (3 df). H2: LRT of A×R (9 df).
  H2b: per-subject c = (R2 − R1)EEGNet − (R2 − R1)CSOANet-v2, two-sided one-sample t-test,
  supported only if significant and positive. Holm over H1, H2, H2b at α 0.05 per family.
  H3: 6 pairs × 4 recipes, Bonferroni-adjusted 99.79 % CIs of paired differences; categories
  A better / B better / equivalent (inside ±2 points) / inconclusive. H4: same, R1 vs T, plus
  Kendall's τ. Sensitivity: int8 accuracy; aligned-rank-transform ANOVA; F1a and F1b separately.

### Open question to raise with Yu before freezing

H3's categories overlap: an interval can lie above 0 and inside ±2 at once. The registration does
not set a precedence. Do not pick one silently; ask Yu, and the choice goes into the freeze amendment.

## Work before the freeze

1. ShallowConvNet variant wired into `train_physionet_mi.py` (or a new confirmatory script).
2. R4 and condition T.
3. Lee2019-MI and Cho2017 loaders, the registered preprocessing, splits and exclusion check,
   tested on synthetic stand-ins only.
4. Per-protocol dropout, balanced accuracy, seeds, per-subject per-seed CSV output, resumable runs.
5. Analysis script for H1-H4 and the sensitivity analyses, tested on synthetic results.
6. Tests in `deploy/test_v2.py` (or a new test file) for every new piece; all tests pass.
7. Yu reviews, then tags the commit (e.g. `prereg-freeze-v1`) and files the OSF amendment with
   the commit hash. Only after that does Yu set `CSOANET_CONFIRMATORY_UNLOCK=1` and download the
   confirmatory datasets.

## Engineering

- Run `python deploy/test_v2.py` after every change.
- Keep everything runnable on CPU; GPU optional.
- Small, focused commits with clear messages.
