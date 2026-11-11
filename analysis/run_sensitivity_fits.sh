#!/bin/zsh
# Queue of preregistered sensitivity refits (OSF 5vwkj 4.6) + the per-image robustness model, run 2 at a time
# after the primary fits finish. Each refit reuses the primary model code with an alternative cells table.
# Logs: logs/sens_<tag>.log ; outputs: results/models/<tag>_* (same files as the primary fits)
cd "$(dirname "$0")/.."
S=data/analysis/sens
waitfor() { while pgrep -f "$1" >/dev/null; do sleep 120; done; }
echo "[$(date)] waiting for primary fits"; waitfor "fit_h1_cells.R --args cells maximal"; waitfor "fit_fakes_cells.R --args h[24]"
run2() {  # run the given commands two at a time
  local -a cmds; cmds=("$@"); local i=1
  while (( i <= ${#cmds} )); do
    eval "${cmds[i]}" & ; (( i++ ))
    if (( i <= ${#cmds} )); then eval "${cmds[i]}" & ; (( i++ )); fi
    wait
  done
}
echo "[$(date)] batch 1: S1, S5, S6 (H1) and S5, S6, S7, S9 (H2), S6 (H4a)"
run2 \
 "Rscript analysis/fit_h1_cells.R S1 maximal $S/S1_reals_cells.parquet > logs/sens_h1_S1.log 2>&1" \
 "Rscript analysis/fit_h1_cells.R S5 maximal data/analysis/reals_cells.parquet 2 > logs/sens_h1_S5.log 2>&1" \
 "Rscript analysis/fit_h1_cells.R S6 maximal $S/S6_reals_cells.parquet > logs/sens_h1_S6.log 2>&1" \
 "Rscript analysis/fit_fakes_cells.R h2 S5 data/analysis/fakes_cells.parquet 2 > logs/sens_h2_S5.log 2>&1" \
 "Rscript analysis/fit_fakes_cells.R h2 S6 $S/S6_fakes_cells.parquet > logs/sens_h2_S6.log 2>&1" \
 "Rscript analysis/fit_fakes_cells.R h2 S7 $S/S7_fakes_cells.parquet > logs/sens_h2_S7.log 2>&1" \
 "Rscript analysis/fit_fakes_cells.R h2 S9 $S/S9_fakes_cells.parquet > logs/sens_h2_S9.log 2>&1" \
 "Rscript analysis/fit_fakes_cells.R h4a S6 $S/S6_fakes_cells.parquet > logs/sens_h4a_S6.log 2>&1"
echo "[$(date)] waiting for the S2/S3 detector grids"; waitfor "run_sensitivity_embed.sh"
.venv/bin/python analysis/build_tidy.py mask && .venv/bin/python analysis/build_tidy.py jpeg
echo "[$(date)] batch 2: S2, S3"
run2 \
 "Rscript analysis/fit_h1_cells.R S2 maximal $S/S2_reals_cells.parquet > logs/sens_h1_S2.log 2>&1" \
 "Rscript analysis/fit_h1_cells.R S3 maximal $S/S3_reals_cells.parquet > logs/sens_h1_S3.log 2>&1" \
 "Rscript analysis/fit_fakes_cells.R h2 S2 $S/S2_fakes_cells.parquet > logs/sens_h2_S2.log 2>&1" \
 "Rscript analysis/fit_fakes_cells.R h2 S3 $S/S3_fakes_cells.parquet > logs/sens_h2_S3.log 2>&1" \
 "Rscript analysis/fit_fakes_cells.R h4a S2 $S/S2_fakes_cells.parquet > logs/sens_h4a_S2.log 2>&1" \
 "Rscript analysis/fit_fakes_cells.R h4a S3 $S/S3_fakes_cells.parquet > logs/sens_h4a_S3.log 2>&1"
echo "[$(date)] robustness: per-image binomial model with (1 | patient), full data"
Rscript analysis/fit_h1.R 1 perimage_full maximal > logs/fit_h1_perimage_full.log 2>&1
echo "[$(date)] all sensitivity fits done"
