#!/bin/zsh
# Sensitivity analysis S4 (OSF 5vwkj 4.6): 512-px images, Open-I excluded (fewer than 512 px for most Open-I images).
# Encoder inputs: DINOv3 512, RAD-DINO 518 (native), BiomedCLIP 224 (fixed). Same detector grid (no ResNet), then
# tidy tables (data/analysis/sens/S4_*) and the H1/H2 refits with the primary model code.
set -e
cd "$(dirname "$0")/.."
echo "[$(date)] S4 embeddings"
.venv/bin/python scripts/embed_all.py dinov3 biomedclip rad-dino --variant 512 --batch 16
.venv/bin/python scripts/compute_pixel_features.py 512
echo "[$(date)] S4 detector grid"
.venv/bin/python scripts/run_embed_grid.py --encoders dinov3__512,rad-dino__512,biomedclip__512
.venv/bin/python scripts/run_embed_grid.py --encoders pix_fft__512,pix_format__512 --detectors logit
.venv/bin/python analysis/build_tidy.py 512
echo "[$(date)] S4 fits"
Rscript analysis/fit_h1_cells.R S4 maximal data/analysis/sens/S4_reals_cells.parquet > logs/sens_h1_S4.log 2>&1; echo "h1_S4 exit $?"
Rscript analysis/fit_fakes_cells.R h2 S4 data/analysis/sens/S4_fakes_cells.parquet > logs/sens_h2_S4.log 2>&1; echo "h2_S4 exit $?"
echo "[$(date)] S4 done"
