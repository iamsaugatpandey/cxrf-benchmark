# H1 (PRIMARY, OSF 5vwkj): does FPR on real test images rise when the hospital is unseen in training?
#
# Preregistered Bernoulli model:
#   flag ~ 1 + family * novelty_site + view + (1 + novelty_site | site) + (1 | patient) + (1 | config:seed)
# Fitted here as the equivalent BINOMIAL model on per-image counts (k flags out of n fits), because the
# Bernoulli version has 4.7 M rows. Aggregation is exact for the fixed effects, site and patient terms.
# The config:seed term cannot be kept once fits are summed within an image; it is replaced by
# (1 | family:encoder), the finest configuration grouping that survives aggregation. Reported as a deviation.
#
# Priors (preregistered): Intercept ~ N(logit 0.02, 1.5); b ~ N(0, 1); sd ~ half-N(0, 1); cor ~ LKJ(2).
# Inputs : data/analysis/reals_agg.parquet (build_tidy.py)
# Outputs: results/models/h1_<tag>.rds (brmsfit), h1_<tag>_summary.txt, h1_<tag>_contrasts.csv
# Usage  : Rscript analysis/fit_h1.R [sample_frac=1] [tag=full] [model=maximal|nocor|noslope]
suppressPackageStartupMessages({
  library(arrow); library(dplyr); library(brms); library(cmdstanr); library(tidybayes); library(tidyr)
})
set_cmdstan_path(file.path(Sys.getenv("HOME"), ".cmdstan/cmdstan-2.40.0"))
args <- commandArgs(trailingOnly = TRUE)
frac  <- if (length(args) >= 1) as.numeric(args[1]) else 1
tag   <- if (length(args) >= 2) args[2] else "full"
mform <- if (length(args) >= 3) args[3] else "maximal"
out <- "results/models"; dir.create(out, showWarnings = FALSE, recursive = TRUE)

d <- read_parquet("data/analysis/reals_agg.parquet") |>
  mutate(family = relevel(factor(family), ref = "mlp"),
         view = factor(ifelse(view == "AP", "AP", "PA"), levels = c("PA", "AP")),
         site = factor(site), cfg = factor(paste(family, encoder, sep = ":")))
if (frac < 1) {  # subsample whole images (keeps the within-image seen/unseen pairing)
  set.seed(1); ids <- unique(d$image_id); d <- filter(d, image_id %in% sample(ids, round(frac * length(ids))))
}
cat(sprintf("rows %d, images %d, patients %d, sites %d\n", nrow(d), n_distinct(d$image_id),
            n_distinct(d$patient_id), n_distinct(d$site)))

re_site <- switch(mform, maximal = "(1 + novelty_site | site)", nocor = "(1 + novelty_site || site)",
                  noslope = "(1 | site)")
f <- bf(as.formula(paste("k | trials(n) ~ 1 + family * novelty_site + view +", re_site,
                         "+ (1 | patient_id) + (1 | cfg)")))
pri <- c(prior(normal(-3.89, 1.5), class = Intercept), prior(normal(0, 1), class = b),
         prior(normal(0, 1), class = sd))
if (mform == "maximal") pri <- c(pri, prior(lkj(2), class = cor))

t0 <- Sys.time()
fit <- brm(f, data = d, family = binomial(), prior = pri, backend = "cmdstanr",
           chains = 4, cores = 4, threads = threading(2), iter = 3000, warmup = 1000,
           control = list(adapt_delta = 0.95), seed = 20261109, refresh = 200,
           file = file.path(out, paste0("h1_", tag)), file_refit = "on_change")
cat("fit time:", format(Sys.time() - t0), "\n")

sink(file.path(out, paste0("h1_", tag, "_summary.txt")))
print(summary(fit)); cat("\nmax Rhat:", max(rhat(fit), na.rm = TRUE), "\n")  # criteria: < 1.01
cat("min bulk ESS ratio:", min(neff_ratio(fit), na.rm = TRUE), "\n")
np <- nuts_params(fit); cat("divergences:", sum(np$Value[np$Parameter == "divergent__"]), "\n")
sink()

# Contrasts on the probability scale (pp), population level (site/patient/config effects at 0), PA view
# Marginal over patients (the estimand is the average FPR over images, not the FPR of the median image):
# 50 new patient levels per row drawn from N(0, sd_patient) and averaged; site and config effects at 0.
nd <- expand_grid(family = levels(d$family), novelty_site = 0:1, view = "PA", n = 1)
ndr <- nd[rep(seq_len(nrow(nd)), each = 50), ] |> mutate(patient_id = paste0("new", row_number()))
epr <- posterior_epred(fit, newdata = ndr, re_formula = ~ (1 | patient_id), allow_new_levels = TRUE,
                       sample_new_levels = "gaussian")
ep <- sapply(seq_len(nrow(nd)), function(i) rowMeans(epr[, ((i - 1) * 50 + 1):(i * 50), drop = FALSE]))
dr <- tibble(draw = rep(seq_len(nrow(ep)), ncol(ep)), p = as.vector(ep),
             family = rep(nd$family, each = nrow(ep)), nov = rep(nd$novelty_site, each = nrow(ep))) |>
  pivot_wider(names_from = nov, values_from = p, names_prefix = "nov") |>
  mutate(fpr_seen = 100 * nov0, fpr_unseen = 100 * nov1, delta_pp = fpr_unseen - fpr_seen)
summ <- dr |> group_by(family) |>
  summarise(across(c(fpr_seen, fpr_unseen, delta_pp),
                   list(med = median, lo95 = ~quantile(.x, .025), hi95 = ~quantile(.x, .975),
                        lo66 = ~quantile(.x, .17), hi66 = ~quantile(.x, .83))),
            pr_pos = mean(delta_pp > 0), .groups = "drop")
write.csv(summ, file.path(out, paste0("h1_", tag, "_contrasts.csv")), row.names = FALSE)
saveRDS(dr, file.path(out, paste0("h1_", tag, "_draws.rds")))
print(summ |> select(family, fpr_seen_med, fpr_unseen_med, delta_pp_med, delta_pp_lo95, delta_pp_hi95, pr_pos))
