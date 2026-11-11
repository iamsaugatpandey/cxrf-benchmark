#!/bin/zsh
# h1_S4 had 3 divergences at adapt_delta 0.95 -> preregistered procedure: refit at 0.99.
# Waits for the S4 pipeline (run_s4_512.sh) to finish, archives the first fit, refits. Log: logs/sens_h1_S4_ad099.log
cd "$(dirname "$0")/.."
echo "[$(date)] waiting for run_s4_512.sh"   # marker for progress.sh: start h1_S4_ad099
while pgrep -f run_s4_512.sh >/dev/null; do sleep 120; done
mkdir -p results/models/superseded
for f in results/models/h1_S4{.rds,_summary.txt,_contrasts.csv,_draws.rds,_ppc.csv,_site_contrasts.csv,_site_draws.rds,_sd_site.rds}(N); do mv $f results/models/superseded/; done
echo "[$(date)] start h1_S4_ad099"
ADAPT_DELTA=0.99 Rscript analysis/fit_h1_cells.R S4 maximal data/analysis/sens/S4_reals_cells.parquet > logs/sens_h1_S4_ad099.log 2>&1
echo "[$(date)] end h1_S4_ad099 (exit $?)"
