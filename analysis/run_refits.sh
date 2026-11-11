#!/bin/zsh
# Refits after the overnight queue (preregistered procedure: divergences -> adapt_delta 0.99; low ESS -> more draws).
cd "$(dirname "$0")/.."
S=data/analysis/sens
step() { local name=$1; shift; echo "[$(date)] start $name"; env "$@" > logs/refit_$name.log 2>&1; echo "[$(date)] end $name (exit $?)"; }
mv_old() { mkdir -p results/models/superseded; for f in results/models/$1{.rds,_summary.txt,_contrasts.csv,_draws.rds,_ppc.csv,_site_contrasts.csv,_site_draws.rds,_sd_site.rds,_gen_draws.rds,_sd_generator.rds}(N); do mv $f results/models/superseded/; done; }
for t in h4b h1_S3 h1_S5 h2_S5 h4a_S2 h4a_S3 h4a_S6; do mv_old $t; done
{ step h4b ADAPT_DELTA=0.99 DRAWS=8000 Rscript analysis/fit_fakes_cells.R h4b;
  step h1_S3 ADAPT_DELTA=0.99 Rscript analysis/fit_h1_cells.R S3 maximal $S/S3_reals_cells.parquet;
  step h1_S5 ADAPT_DELTA=0.99 Rscript analysis/fit_h1_cells.R S5 maximal data/analysis/reals_cells.parquet 2;
  step h2_S5 ADAPT_DELTA=0.99 Rscript analysis/fit_fakes_cells.R h2 S5 data/analysis/fakes_cells.parquet 2; } &
{ step h4a_S2 ADAPT_DELTA=0.99 Rscript analysis/fit_fakes_cells.R h4a S2 $S/S2_fakes_cells.parquet;
  step h4a_S3 ADAPT_DELTA=0.99 Rscript analysis/fit_fakes_cells.R h4a S3 $S/S3_fakes_cells.parquet;
  step h4a_S6 ADAPT_DELTA=0.99 Rscript analysis/fit_fakes_cells.R h4a S6 $S/S6_fakes_cells.parquet;
  step h4a_S7 ADAPT_DELTA=0.99 Rscript analysis/fit_fakes_cells.R h4a S7 $S/S7_fakes_cells.parquet; } &
wait; echo "[$(date)] refits done"
