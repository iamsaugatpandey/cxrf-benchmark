# LLM evaluation protocol (H5) — design rationale with sources

This is the reference document for the LLM arm. Methods text and the TRIPOD-LLM checklist are drafted from it. It implements preregistration OSF 5vwkj ("Measured variables → LLM evaluation"). Code: `scripts/run_llm.py` (prompt version `detect-v3`, frozen 2026-11-09 before any test image was sent). BibTeX keys are in `manuscript/references.bib`.

## 1. Task framing
| Choice | What we do | Why / source |
| --- | --- | --- |
| Task | Estimate the probability that a single radiograph is AI-generated or AI-edited | Same authenticity task as Tordjman et al. 2026 [`tordjman2026`] and Lee et al. 2026 [`lee2026`]. "Or edited" covers our image-conditioned arm. |
| Prevalence disclosure | The model is told the set contains both kinds of image, but not the proportion | Matches Lee et al.'s reader instructions ("told the set contained both real and AI-generated images but were not told the true class prevalence"). Stating a prevalence would anchor the probabilities. |
| Research framing (system prompt) | "image-forensics research study… not a diagnostic task" | Reduces refusals on medical images (pilot: 0 refusals with this framing). Refusals are still counted, never imputed in the primary analysis (preregistered). |
| Format disclosure | The prompt states that all images share one format (256 px, grayscale, cropped, no metadata) | True statement. It stops the model from reasoning about file properties we deliberately neutralised. The trained detectors face the same constraint. |

## 2. Output and elicitation
| Choice | What we do | Why / source |
| --- | --- | --- |
| Primary output | Verbalised probability, integer 0–100 | Verbalised confidence is the standard way to elicit probabilities from instruction-tuned LLMs. It is often better calibrated than token probabilities for RLHF models [`tian2023just`; `xiong2024can`]. A continuous score is required for AUC, ECE and Brier. |
| Anchors | 0 = certainly authentic, 100 = certainly AI, 50 = cannot tell | Explicit scale anchors reduce scale-interpretation variance [`xiong2024can`]. |
| Secondary output | One-sentence main visual cue | Enables the preregistered exploratory coding into Lee et al.'s T1–T9 taxonomy (2 coders on 10%, Cohen's κ [`cohen1960coefficient`]). Free text, not taxonomy labels, so the model is not primed with our categories. |
| Format enforcement | JSON schema through each provider's native structured-output API (Anthropic `output_config.format`, OpenAI `text.format` strict, Gemini `responseJsonSchema`) | Same approach as Lee et al. ("Structured JSON output was enforced through model-native mechanisms"). It removes parsing failures as a confound. The 0–100 range is checked in the parser, because Anthropic schemas do not accept integer bounds. Out-of-range values are flagged `unparseable`. |
| Decision threshold | Primary: 50. Secondary: calibrated to 1% FPR on validation reals | Preregistered. |

## 3. Inference settings
| Choice | What we do | Why / source |
| --- | --- | --- |
| Sampling | Provider defaults; no temperature or top-p anywhere | Preregistered ("provider default sampling"). Claude 5.x does not accept temperature. Test–retest κ then measures real run-to-run variability. |
| Reasoning effort | `low` on all six models (Anthropic `effort`, OpenAI `reasoning.effort`, Gemini `thinkingLevel`) | Lee et al. listed their mixed inference settings as a limitation ("this inference control heterogeneity is acknowledged as a limitation"). We use one uniform setting. "Low" also keeps the preregistered OpenAI budget ($25) feasible: default effort would roughly double GPT-6 output tokens. Reported as an implementation detail. |
| Max output tokens | 4,000 | Room for reasoning plus the JSON. |
| Refusal fallbacks | None | A server-side fallback would answer with a different model than the one being evaluated. Refusals are outcomes and are reported as such. |
| Batch APIs | All three providers, 50% price | Same model and outputs as synchronous calls. Batch results are asynchronous and order-free, so they are keyed by `custom_id`. |

## 4. Images and design
| Choice | What we do | Why / source |
| --- | --- | --- |
| Image | The 256-px preprocessed PNG that the trained detectors score | A true same-image head-to-head. 188/300 sampled Open-I images do not exist at 512 px, so 512 would make resolution a hospital cue. |
| Subset | 2,000 test images, stratified (1,000 real / 1,000 fake), seed 20261109; CheXpert excluded (data-use agreement) | Preregistered. The prior LLM studies used 60–264 images; per-stratum n here supports hierarchical estimates. |
| Order | Pre-randomised, stratum-balanced run order; preregistered prefix rule if a budget cap stops a run | Preregistered. Keeps any truncated run representative. |
| Test–retest | 2 extra zero-shot runs on 403 images | Preregistered. Reported as Cohen's κ at threshold 50 and the SD of probabilities. |
| 4-shot | 4 fixed labelled training-split examples (2 real: NIH, Open-I; 2 fake: SynthCheX, MONAI), alternating authentic/AI order, balanced labels; 503 test images | In-context examples per Brown et al. [`brown2020language`]. Few-shot predictions are sensitive to label balance and order (majority-label and recency bias, Zhao et al. [`zhao2021calibrate`]), so the examples are label-balanced, alternating, fixed and identical for every model. Examples come only from training-split images and open generators, so no test image or test-only generator is shown. |

## 5. Metrics (preregistered)
- Discrimination: AUC with cluster-bootstrap CIs.
- At threshold: TPR, FPR, per hospital and generator, in the same multilevel models as the trained detectors.
- Calibration: ECE with 10 equal-mass bins [`naeini2015obtaining`; `nixon2019measuring`; `guo2017calibration`], Brier score [`brier1950verification`], and reliability diagrams.
- Repeatability: Cohen's κ across runs [`cohen1960coefficient`].
- Cost per 1,000 images, from returned token usage.

## 6. Reporting
- TRIPOD-LLM [`gallifant2025tripod`] and CLAIM 2024 [`tejani2024`] checklists.
- Release: prompts verbatim (this file + `run_llm.py`), exact request bodies (`data/llm/requests/`, images excluded from release where licences forbid), raw responses (`data/llm/raw/`), returned model snapshot IDs, run dates.

## 7. Deviations from the pilot (not from the preregistration)
- Pilot used 512-px images, a Gemini temperature of 0 and no schema enforcement. These are replaced by 256-px images, defaults with uniform low effort, and native JSON schemas.
- Prompt v3 adds the prevalence-undisclosed sentence and the format sentence, both noted above.
