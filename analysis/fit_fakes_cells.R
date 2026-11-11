# H2 and H4 (secondary, OSF 5vwkj): detection of fakes (TPR), cell-level beta-binomial (analysis/betabinom2.R),
# same formulation as fit_h1_cells.R: the preregistered per-source-report / per-source-image term becomes
# beta-binomial overdispersion within cells; (1 | config:seed) is kept.
#
#  h2  : text-only fakes from G1, G2, G4, G5. novelty = generator absent from the training fakes.
#        k|vint(n) ~ family * novelty + (1 + novelty | generator) + (1 | config_seed)
#  h4a : G4/G5 fakes; novelty = image-conditioned (vs text-only). All commercial fakes are unseen generators.
#        k|vint(n) ~ family * novelty + (1 | generator) + (1 | config_seed)
#  h4b : image-conditioned fakes only; novelty = the source radiograph's hospital is absent from the detector's
#        training reals (Montgomery always unseen).
#        k|vint(n) ~ family * novelty + (1 | cond_site) + (1 | generator) + (1 | config_seed)
# Priors as preregistered (Intercept N(logit 0.02, 1.5), b N(0,1), sd half-N(0,1), cor LKJ(2)); phi gamma(0.01, 0.01).
# Env: ADAPT_DELTA (default 0.95; 0.99 after divergences, per the preregistered procedure), DRAWS per chain (default 3000)
# Usage: Rscript analysis/fit_fakes_cells.R h2|h4a|h4b [tag_suffix] [cells.parquet] [prior_scale=1]
suppressPackageStartupMessages({library(arrow); library(dplyr); library(tidyr); library(brms); library(cmdstanr)})
set_cmdstan_path(file.path(Sys.getenv("HOME"), ".cmdstan/cmdstan-2.40.0"))
source("analysis/betabinom2.R")
args <- commandArgs(trailingOnly = TRUE)
which <- args[1]; tag <- paste0(which, if (length(args) >= 2 && args[2] != "") paste0("_", args[2]) else "")
infile <- if (length(args) >= 3) args[3] else "data/analysis/fakes_cells.parquet"  # sensitivity cells (S2, S3, S6, S7, S9)
pscale <- if (length(args) >= 4) as.numeric(args[4]) else 1
out <- "results/models"; dir.create(out, showWarnings = FALSE, recursive = TRUE)
core <- c("nih", "openi", "shenzhen", "chexpert")
train_sites <- function(h) if (h == "nih_only") "nih" else setdiff(core, sub("loho_", "", h))

d <- read_parquet(infile) |>
  mutate(family = relevel(factor(family), ref = "mlp"),
         config_seed = factor(paste(config, seed, sep = "#")),
         hosp_cfg = sapply(strsplit(config, "\\|"), `[`, 3))
if (which == "h2") {
  d <- filter(d, conditioning == "text") |> mutate(novelty = novelty_gen)
  f <- k | vint(n) ~ 1 + family * novelty + (1 + novelty | generator) + (1 | config_seed)
  contrast <- "generator unseen - seen"
} else if (which == "h4a") {
  d <- filter(d, generator %in% c("G4", "G5")) |> mutate(novelty = as.integer(conditioning == "image"))
  f <- k | vint(n) ~ 1 + family * novelty + (1 | generator) + (1 | config_seed)
  contrast <- "image-conditioned - text-only"
} else {
  d <- filter(d, conditioning == "image") |>
    mutate(novelty = as.integer(!mapply(function(s, h) s %in% train_sites(h), cond_site, hosp_cfg)))
  f <- k | vint(n) ~ 1 + family * novelty + (1 | cond_site) + (1 | generator) + (1 | config_seed)
  contrast <- "conditioning hospital unseen - seen"
}
cat(sprintf("%s: cells %d, image-fits %d\n", tag, nrow(d), sum(d$n)))
pri <- c(prior_string(sprintf("normal(-3.89, %g)", 1.5 * pscale), class = "Intercept"),
         prior_string(sprintf("normal(0, %g)", pscale), class = "b"),
         prior_string(sprintf("normal(0, %g)", pscale), class = "sd"),
         prior(gamma(0.01, 0.01), class = phi))
if (which == "h2") pri <- c(pri, prior(lkj(2), class = cor))
fit <- brm(f, data = d, family = betabinom2, stanvars = betabinom2_stanvars, prior = pri, backend = "cmdstanr",
           chains = 4, cores = 4, iter = 1000 + as.integer(Sys.getenv("DRAWS", "3000")), warmup = 1000,
           control = list(adapt_delta = as.numeric(Sys.getenv("ADAPT_DELTA", "0.95"))), init = 0,
           seed = 20261109, refresh = 500, file = file.path(out, tag), file_refit = "on_change")
ess <- posterior::summarise_draws(posterior::as_draws_df(fit), "rhat", "ess_bulk", "ess_tail")
ess <- ess[!grepl("^(lp__|lprior|r_config_seed\\[)", ess$variable), ]
np <- nuts_params(fit); ndiv <- sum(np$Value[np$Parameter == "divergent__"])
ok <- max(ess$rhat, na.rm = TRUE) < 1.01 && min(ess$ess_bulk, na.rm = TRUE) > 1000 &&
      min(ess$ess_tail, na.rm = TRUE) > 1000 && ndiv == 0
sink(file.path(out, paste0(tag, "_summary.txt"))); print(summary(fit))
cat(sprintf("\nmax Rhat %.4f | min bulk ESS %.0f | min tail ESS %.0f | divergences %d\n", max(ess$rhat, na.rm = TRUE),
            min(ess$ess_bulk, na.rm = TRUE), min(ess$ess_tail, na.rm = TRUE), ndiv))
cat("criteria met:", ok, "\n"); sink()
cat("criteria met:", ok, "\n")

nd <- expand_grid(family = levels(d$family), novelty = 0:1, n = 1)
ep <- posterior_epred(fit, newdata = nd, re_formula = NA)  # typical generator / hospital / fit; = expected TPR
dr <- tibble(draw = rep(seq_len(nrow(ep)), ncol(ep)), p = as.vector(ep),
             family = rep(nd$family, each = nrow(ep)), nov = rep(nd$novelty, each = nrow(ep))) |>
  pivot_wider(names_from = nov, values_from = p, names_prefix = "nov") |>
  mutate(tpr0 = 100 * nov0, tpr1 = 100 * nov1, delta_pp = tpr1 - tpr0)
summ <- dr |> group_by(family) |> summarise(
  across(c(tpr0, tpr1, delta_pp), list(med = median, lo95 = ~quantile(.x, .025), hi95 = ~quantile(.x, .975),
                                       lo66 = ~quantile(.x, .17), hi66 = ~quantile(.x, .83))),
  pr_neg = mean(delta_pp < 0), .groups = "drop") |>
  mutate(contrast = contrast,
         verdict = case_when(delta_pp_hi95 < 0 & delta_pp_med <= -1 ~ "supported",
                             delta_pp_lo95 > -1 & delta_pp_hi95 < 1 ~ "not supported (negligible)",
                             TRUE ~ "inconclusive"))
write.csv(summ, file.path(out, paste0(tag, "_contrasts.csv")), row.names = FALSE)
saveRDS(dr, file.path(out, paste0(tag, "_draws.rds")))
if (which == "h2") {  # generator-level TPR and between-generator SD (H3)
  nd2 <- expand_grid(family = levels(d$family), generator = sort(unique(d$generator)), novelty = 0:1, n = 1)
  e2 <- posterior_epred(fit, newdata = nd2, re_formula = ~ (1 + novelty | generator))
  saveRDS(list(nd = nd2, draws = e2), file.path(out, paste0(tag, "_gen_draws.rds")))
  saveRDS(posterior::as_draws_df(fit)[, c("sd_generator__Intercept", "sd_generator__novelty")],
          file.path(out, paste0(tag, "_sd_generator.rds")))
}
print(summ |> select(family, tpr0_med, tpr1_med, delta_pp_med, delta_pp_lo95, delta_pp_hi95, verdict))
