#!/bin/zsh
# Sensitivity analyses S2 (mask) and S3 (jpeg): variant images -> embeddings (3 encoders) -> pixel features
# -> the same detector grid as the primary analysis (all trained families except ResNet-18).
# Outputs: data/processed_variants/<v>/, data/embeddings/<enc>__<v>/, results/scores/embed/<enc>__<v>/...
set -e
cd "$(dirname "$0")/.."
for v in mask jpeg; do
  echo "[$(date)] variant $v: images"
  .venv/bin/python scripts/make_variants.py $v --workers 4
  echo "[$(date)] variant $v: embeddings"
  .venv/bin/python scripts/embed_all.py dinov3 biomedclip rad-dino --variant $v
  .venv/bin/python scripts/compute_pixel_features.py $v
  echo "[$(date)] variant $v: detector grid"
  .venv/bin/python scripts/run_embed_grid.py --encoders dinov3__$v,rad-dino__$v,biomedclip__$v
  .venv/bin/python scripts/run_embed_grid.py --encoders pix_fft__$v,pix_format__$v --detectors logit
done
echo "[$(date)] sensitivity S2/S3 done"
