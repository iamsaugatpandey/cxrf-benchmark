# H1 PRIMARY FIT (OSF 5vwkj), cell-level beta-binomial formulation.
#
# Preregistered Bernoulli model:
#   flag ~ 1 + family * novelty_site + view + (1 + novelty_site | site) + (1 | patient) + (1 | config:seed)
# Fitted on cells (site x view x fitted detector x seed x novelty: k flags of n images; 5,265 rows) with a
# beta-binomial likelihood (custom family, analysis/betabinom2.R). Fixed effects, (1 + novelty_site | site)
# and (1 | config:seed) are kept exactly as preregistered. (1 | patient) is replaced by beta-binomial
# overdispersion within cells: images of one patient fall in one cell per fit. Correlation of the same images
# across fits is ignored, which makes seen-vs-unseen intervals conservative. The per-image binomial model with
# (1 | patient) is fitted as a robustness check (fit_h1.R).
# History (all logged in WORKLOG): per-image model on the full data needs > 30 h and did not converge on a 5%
# sample; built-in beta_binomial failed at initialisation; a binomial + observation-level RE version
# (5,265 extra parameters) mixed badly (Rhat 1.24, ESS 13) and is archived in results/models/olre_nonconverged/.
#
# Priors (preregistered): Intercept ~ N(logit 0.02, 1.5); b ~ N(0, 1); sd ~ half-N(0, 1); cor ~ LKJ(2).
# phi (not preregistered): gamma(0.01, 0.01), brms' default for beta-binomial precision.
# Env: ADAPT_DELTA (default 0.95; 0.99 after divergences, per the preregistered procedure), DRAWS per chain (default 3000)
# Usage: Rscript analysis/fit_h1_cells.R [tag=cells] [model=maximal|nocor|noslope] [cells.parquet] [prior_scale=1]
# Output: results/models/h1_<tag>{.rds,_summary.txt,_contrasts.csv,_site_contrasts.csv,_draws.rds,_site_draws.rds,_ppc.csv,_sd_site.rds}
suppressPackageStartupMessages({
  library(arrow); library(dplyr); library(tidyr); library(brms); library(cmdstanr)
})
set_cmdstan_path(file.path(Sys.getenv("HOME"), ".cmdstan/cmdstan-2.40.0"))
source("analysis/betabinom2.R")
args <- commandArgs(trailingOnly = TRUE)
tag    <- if (length(args) >= 1) args[1] else "cells"
mform  <- if (length(args) >= 2) args[2] else "maximal"
infile <- if (length(args) >= 3) args[3] else "data/analysis/reals_cells.parquet"  # sensitivity cells (S1, S2, S3, S6)
pscale <- if (length(args) >= 4) as.numeric(args[4]) else 1                        # S5: 2 = priors twice as wide
out <- "results/models"; dir.create(out, showWarnings = FALSE, recursive = TRUE)

d <- read_parquet(infile) |>
  mutate(family = relevel(factor(family), ref = "mlp"),
         view = factor(ifelse(view == "AP", "AP", "PA"), levels = c("PA", "AP")),
         site = factor(site), config_seed = factor(paste(config, seed, sep = "#")))
cat(sprintf("%s: cells %d, image-fits %d, fits x seeds %d, sites %d\n", tag, nrow(d), sum(d$n),
            n_distinct(d$config_seed), n_distinct(d$site)))

re_site <- switch(mform, maximal = "(1 + novelty_site | site)", nocor = "(1 + novelty_site || site)",
                  noslope = "(1 | site)",
                  # EXPLORATORY (not preregistered): hospital effects that differ by detector family, added because the
                  # preregistered model's posterior predictive check fitted only 49/91 site x family x novelty cells
                  sitefam = "(1 + novelty_site | site) + (1 + novelty_site | site:family)")
vterm <- if (n_distinct(d$view) > 1) "+ view" else ""  # S1 (PA-only) has a single view
f <- bf(as.formula(paste("k | vint(n) ~ 1 + family * novelty_site", vterm, "+", re_site, "+ (1 | config_seed)")))
pri <- c(prior_string(sprintf("normal(-3.89, %g)", 1.5 * pscale), class = "Intercept"),
         prior_string(sprintf("normal(0, %g)", pscale), class = "b"),
         prior_string(sprintf("normal(0, %g)", pscale), class = "sd"),
         prior(gamma(0.01, 0.01), class = phi))
if (mform %in% c("maximal", "sitefam")) pri <- c(pri, prior(lkj(2), class = cor))

t0 <- Sys.time()
fit <- brm(f, data = d, family = betabinom2, stanvars = betabinom2_stanvars, prior = pri, backend = "cmdstanr",
           chains = 4, cores = 4, iter = 1000 + as.integer(Sys.getenv("DRAWS", "3000")), warmup = 1000,
           control = list(adapt_delta = as.numeric(Sys.getenv("ADAPT_DELTA", "0.95"))), init = 0,
           seed = 20261109, refresh = 500, file = file.path(out, paste0("h1_", tag)), file_refit = "on_change")
cat("fit time:", format(Sys.time() - t0), "\n")

# ---- diagnostics (preregistered criteria), on population-level, SD, correlation, phi and site effects
ess <- posterior::summarise_draws(posterior::as_draws_df(fit), "rhat", "ess_bulk", "ess_tail")
ess <- ess[!grepl("^(lp__|lprior|r_config_seed\\[)", ess$variable), ]
np <- nuts_params(fit); ndiv <- sum(np$Value[np$Parameter == "divergent__"])
ok <- max(ess$rhat, na.rm = TRUE) < 1.01 && min(ess$ess_bulk, na.rm = TRUE) > 1000 &&
      min(ess$ess_tail, na.rm = TRUE) > 1000 && ndiv == 0
sink(file.path(out, paste0("h1_", tag, "_summary.txt")))
print(summary(fit))
cat(sprintf("\nmax Rhat %.4f | min bulk ESS %.0f | min tail ESS %.0f | divergences %d\n",
            max(ess$rhat, na.rm = TRUE), min(ess$ess_bulk, na.rm = TRUE), min(ess$ess_tail, na.rm = TRUE), ndiv))
cat("criteria met:", ok, "\n")
sink()
cat("criteria met:", ok, "\n")

# ---- posterior predictive check: observed vs predicted FPR per site x family x novelty
yrep <- posterior_predict(fit, ndraws = 500)
ppc <- d |> mutate(row = row_number()) |> group_by(site, family, novelty_site) |>
  group_modify(function(g, k) {
    pr <- rowSums(yrep[, g$row, drop = FALSE]) / sum(g$n)
    tibble(obs = sum(g$k) / sum(g$n), pred_med = median(pr), pred_lo = quantile(pr, .025), pred_hi = quantile(pr, .975))
  }) |> ungroup() |> mutate(inside = obs >= pred_lo & obs <= pred_hi)
write.csv(ppc, file.path(out, paste0("h1_", tag, "_ppc.csv")), row.names = FALSE)
cat(sprintf("PPC: %d / %d site x family x novelty cells inside 95%% predictive interval\n", sum(ppc$inside), nrow(ppc)))

# ---- population-level contrasts (typical site and fit: site and config effects at 0; PA), pp.
# Beta-binomial mean = mu, so this is the expected FPR at the typical site and fit.
nd <- expand_grid(family = levels(d$family), novelty_site = 0:1, view = "PA", n = 1)
ep <- posterior_epred(fit, newdata = nd, re_formula = NA)
dr <- tibble(draw = rep(seq_len(nrow(ep)), ncol(ep)), p = as.vector(ep),
             family = rep(nd$family, each = nrow(ep)), nov = rep(nd$novelty_site, each = nrow(ep))) |>
  pivot_wider(names_from = nov, values_from = p, names_prefix = "nov") |>
  mutate(fpr_seen = 100 * nov0, fpr_unseen = 100 * nov1, delta_pp = fpr_unseen - fpr_seen)
summ <- dr |> group_by(family) |> summarise(
  across(c(fpr_seen, fpr_unseen, delta_pp), list(med = median, lo95 = ~quantile(.x, .025), hi95 = ~quantile(.x, .975),
                                                 lo66 = ~quantile(.x, .17), hi66 = ~quantile(.x, .83))),
  pr_pos = mean(delta_pp > 0), .groups = "drop") |>
  mutate(verdict = case_when(delta_pp_lo95 > 0 & delta_pp_med >= 1 ~ "supported",
                             delta_pp_lo95 > -1 & delta_pp_hi95 < 1 ~ "not supported (negligible)",
                             TRUE ~ "inconclusive"))
write.csv(summ, file.path(out, paste0("h1_", tag, "_contrasts.csv")), row.names = FALSE)
saveRDS(dr, file.path(out, paste0("h1_", tag, "_draws.rds")))

# ---- site-specific FPR (observed sites; config effect at 0; PA)
nd2 <- expand_grid(family = levels(d$family), site = levels(d$site), novelty_site = 0:1, view = "PA", n = 1)
ep2 <- posterior_epred(fit, newdata = nd2, re_formula = if (mform == "sitefam") ~ (1 + novelty_site | site) + (1 + novelty_site | site:family) else ~ (1 + novelty_site | site))
sc <- nd2 |> mutate(med = 100 * apply(ep2, 2, median), lo95 = 100 * apply(ep2, 2, quantile, .025),
                    hi95 = 100 * apply(ep2, 2, quantile, .975))
write.csv(sc, file.path(out, paste0("h1_", tag, "_site_contrasts.csv")), row.names = FALSE)
saveRDS(list(nd = nd2, draws = ep2), file.path(out, paste0("h1_", tag, "_site_draws.rds")))
# H3 ingredient: between-site SDs (logit scale)
saveRDS(posterior::as_draws_df(fit)[, c("sd_site__Intercept", "sd_site__novelty_site")],
        file.path(out, paste0("h1_", tag, "_sd_site.rds")))
print(summ |> select(family, fpr_seen_med, fpr_unseen_med, delta_pp_med, delta_pp_lo95, delta_pp_hi95, pr_pos, verdict))
