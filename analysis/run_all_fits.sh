#!/bin/zsh
# Driver for all Bayesian fits (Phase 8). Primary fits first (H1 || H2 || H4a->H4b), then the sensitivity queue.
# Each step logs to logs/<name>.log and prints start/end with exit codes here.
cd "$(dirname "$0")/.."
step() { local name=$1; shift; echo "[$(date)] start $name"; "$@" > logs/$name.log 2>&1; echo "[$(date)] end $name (exit $?)"; }
step fit_h1_cells Rscript analysis/fit_h1_cells.R cells maximal &
step fit_h2 Rscript analysis/fit_fakes_cells.R h2 &
{ step fit_h4a Rscript analysis/fit_fakes_cells.R h4a; step fit_h4b Rscript analysis/fit_fakes_cells.R h4b; } &
wait
echo "[$(date)] primary fits done; starting sensitivity queue"
zsh analysis/run_sensitivity_fits.sh
