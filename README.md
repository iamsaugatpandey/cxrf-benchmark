# Detecting AI-generated chest radiographs across hospitals and generators

Code for the preregistered benchmark *"Detecting AI-generated chest radiographs across hospitals and generators: a multi-institution benchmark"* (Saugat Pandey, Furman University).

- **Preregistration:** Open Science Framework, https://doi.org/10.17605/OSF.IO/5VWKJ (registered 9 November 2026)
- **Paper:** under review. A citation will be added on publication.

## What the study does

We trained 585 detectors of AI-generated chest radiographs. They come from seven detector families: frozen-encoder probes with and without latent blending, kNN, one-class SVM, spectrum and file-format baselines, and a fine-tuned ResNet-18. The detectors were trained on 5 hospital configurations × 3 generator configurations. We then measured two things:

1. **Hospital shift (H1):** the false-positive rate on genuine radiographs from hospitals the detector never saw.
2. **Generator shift (H2–H4):** the true-positive rate on fakes from generators the detector never saw, including text-prompted and image-conditioned fakes from gpt-image-2 and Gemini.

Each threshold was fixed in advance at the 99th percentile of validation real images from the training hospitals. Outcomes were analysed with Bayesian multilevel beta-binomial models (brms/Stan). Six frontier multimodal LLMs (Claude, GPT and Gemini models) were scored on a frozen 2,000-image subset.

## Repository layout

```
src/cxrf/            Python package: preprocessing, splits, encoders, detectors, metrics, API logging
scripts/             Pipeline steps (data download → preprocessing → generation → embeddings → detectors → LLMs)
analysis/            Analysis tables, Bayesian models (R), bootstrap cross-check, LLM metrics, Supplementary Information builder
figures/             Figure scripts (R/ggplot2 and Python) and the shared theme
tests/               Unit tests (pytest)
results/models/      Posterior summaries, contrasts and posterior predictive checks for every fitted model (CSV/TXT)
results/tables/      Bootstrap and LLM summary tables (CSV)
manifest/            Image identifier list with source, hospital, class, generator, view and split (images.csv.gz)
docs/                LLM evaluation protocol
```

## Data

No images are included. The real radiographs come from public datasets. Download them from their providers and accept each provider's terms:

| Hospital / source | Dataset | Access |
|---|---|---|
| NIH Clinical Center (US) | ChestX-ray14 | https://nihcc.app.box.com/v/ChestXray-NIHCC |
| Indiana University (US) | Open-I | https://openi.nlm.nih.gov/faq |
| Montgomery County (US), Shenzhen (CN) | NLM tuberculosis sets | https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#tuberculosis-image-data-sets |
| JSRT (JP), PadChest (ES) | via NODE21 | https://zenodo.org/records/5548363 |
| Stanford (US) | CheXpert Plus | https://aimi.stanford.edu/datasets/chexpert-plus (Stanford research-use agreement) |
| Open generator G1 | SynthCheX-75K | Hugging Face |
| Open generator G2 | MONAI CXR latent diffusion | Hugging Face |

`manifest/images.csv.gz` lists every image's identifier, source, hospital, class, generator, view, split and exclusion flag, so the splits can be reproduced exactly. CheXpert, NODE21 and PadChest images are referenced by identifier only, as their licences require. CheXpert images were never sent to any third-party API.

## Large files on Zenodo

Files too large for GitHub are archived on Zenodo: **DOI: 10.5281/zenodo.24144501** (https://doi.org/10.5281/zenodo.24144501; this DOI always resolves to the latest version). The archive contains:

| File | Contents |
|---|---|
| `synthetic_images.zip` | every generated radiograph used in the study (MONAI, gpt-image-2, Gemini; text-prompted and image-conditioned), with prompts and QC flags |
| `llm_responses.zip` | raw responses of the six multimodal LLMs on the frozen 2,000-image subset |
| `detector_scores.zip` | per-image scores, thresholds and flags of all 585 fitted detectors |
| `embeddings.zip` | frozen-encoder embeddings and pixel features |
| `model_posteriors.zip` | cell tables and posterior draws of every Bayesian model |

Files are released under restricted access: request them on the Zenodo page. Per-image material derived from CheXpert Plus and NODE21 (JSRT, PadChest) is not included, because their licences do not permit redistribution.

## Images and outputs from GPT, Claude and Gemini

**This repository does not include the synthetic radiographs generated with gpt-image-2 (OpenAI) or Gemini (Google), or the per-image responses of the GPT, Claude and Gemini models used as detectors.** They are in the Zenodo archive above under restricted access. Researchers who need these images or raw results can request them there or contact the author:

**Saugat Pandey, Furman University: spandey2@furman.edu**

They will be shared where the provider terms and the source datasets' licences allow.

## Setup

Python 3.11 (developed on macOS with Apple-silicon MPS):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock
pip install -e .
pytest                      # unit tests
```

R 4.3 with `brms`, `cmdstanr` (CmdStan 2.40), `arrow`, `dplyr`, `tidyr`, `tidybayes`, `ggdist`, `ggplot2`, `patchwork`, `ggrepel` and `forcats`.

API keys are only needed to regenerate commercial fakes or re-run the LLM evaluation. Copy `.env.example` to `.env` (git-ignored) and fill in the keys.

## Reproducing the analysis

Run every command from the repository root. The pipeline expects the data under `data/` (see each script's docstring).

| Step | Command |
|---|---|
| 1. Inventory, preprocessing, splits | `scripts/build_inventory.py`, `scripts/preprocess_all.py`, `scripts/build_metadata.py` |
| 2. Open and commercial fakes | `scripts/gen_open.py`, `scripts/gen_api.py` (dry run by default), `scripts/qc_fakes.py` |
| 3. Embeddings and pixel features | `scripts/embed_all.py`, `scripts/compute_pixel_features.py` |
| 4. Detectors | `scripts/run_embed_grid.py`, `scripts/train_resnet.py` |
| 5. Sensitivity variants | `scripts/make_variants.py`, `scripts/run_sensitivity_embed.sh`, `scripts/run_s4_512.sh`, `analysis/sensitivity_cells.py` |
| 6. LLM evaluation | `scripts/freeze_llm_subset.py`, `scripts/run_llm.py`, `analysis/llm_metrics.py` |
| 7. Analysis tables | `analysis/build_tidy.py`, `analysis/bootstrap_h1.py` |
| 8. Bayesian models | `analysis/run_all_fits.sh` (calls `fit_h1_cells.R` and `fit_fakes_cells.R`), then `analysis/run_refits.sh` |
| 9. Figures | `Rscript figures/fig2_h1.R`, … `figures/fig7_multiverse.R`; `python figures/fig1a_design.py` |
| 10. Supplementary Information | `python analysis/make_si.py` |

The full model fits take several hours each on a laptop. Their summaries are already in `results/models/`.

## Licence

Code: MIT licence (see `LICENSE`). The datasets keep their own licences.

## Contact

Saugat Pandey, Department of Computer Science, Furman University: spandey2@furman.edu
