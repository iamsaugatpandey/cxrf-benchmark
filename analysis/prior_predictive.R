# Prior predictive check for the primary model (preregistration §4.3).
# Simulates the design (no outcome data), samples from the PRIOR ONLY, and plots the implied
# distribution of FPR per (family x site novelty). Requirement: the 95% prior range covers
# roughly 0.1%–60% without piling up at 0 or 1.
# Run from the project root:  Rscript analysis/prior_predictive.R
source("figures/theme_cxrf.R")
suppressPackageStartupMessages({ library(brms); library(tidybayes); library(cmdstanr) })
set_cmdstan_path(file.path(Sys.getenv("HOME"), ".cmdstan", "cmdstan-2.40.0"))
set.seed(20261108)

design <- expand_grid(
  family = c("frozen-encoder", "pixel", "cnn"),
  site = c("NIH", "Open-I", "Shenzhen", "CheXpert"),
  config = paste0("c", 1:4),
  patient = 1:25
) |>
  mutate(novelty_site = ifelse(site == "NIH", "seen", "unseen"),
         view = sample(c("PA", "AP"), n(), replace = TRUE, prob = c(0.7, 0.3)),
         patient = paste(site, patient),
         flag = rbinom(n(), 1, 0.05))  # placeholder outcome; ignored with sample_prior = "only"

priors <- c(
  prior(normal(-3.9, 1.5), class = Intercept),   # logit(0.02) = -3.89
  prior(normal(0, 1), class = b),
  prior(normal(0, 1), class = sd),                # half-normal via lower bound 0
  prior(lkj(2), class = cor)
)

fit <- brm(
  flag ~ 1 + family * novelty_site + view + (1 + novelty_site | site) + (1 | patient) + (1 | config),
  data = design, family = bernoulli(), prior = priors, sample_prior = "only",
  chains = 2, iter = 1500, warmup = 500, cores = 2, seed = 1, refresh = 0, backend = "cmdstanr"
)

newd <- expand_grid(family = unique(design$family), novelty_site = c("seen", "unseen"), view = "PA")
draws <- newd |>
  add_epred_draws(fit, re_formula = NA) |>
  ungroup()

summ <- draws |>
  group_by(family, novelty_site) |>
  summarise(q025 = quantile(.epred, 0.025), q50 = median(.epred), q975 = quantile(.epred, 0.975),
            p_lt_0.001 = mean(.epred < 0.001), p_gt_0.9 = mean(.epred > 0.9), .groups = "drop")
print(summ)
readr::write_csv(summ, "admin/prior_predictive_summary.csv")

p <- ggplot(draws, aes(x = .epred, y = family, fill = novelty_site, colour = novelty_site)) +
  cxrf_halfeye(height = 0.8) +
  scale_x_continuous(trans = "logit", breaks = c(0.001, 0.01, 0.05, 0.2, 0.5, 0.8),
                     labels = scales::label_percent(accuracy = 0.1)) +
  scale_fill_manual(values = c(seen = cxrf_pal$seen, unseen = cxrf_pal$unseen)) +
  scale_colour_manual(values = c(seen = cxrf_pal$seen, unseen = cxrf_pal$unseen)) +
  labs(x = "Prior predictive false-positive rate (logit axis)", y = NULL,
       title = "Priors imply plausible false-positive rates",
       subtitle = "Primary model sampled from the prior only; blue = seen hospital, red = unseen") +
  theme_cxrf()
save_cxrf(p, "admin/prior_predictive", width_mm = 120, height_mm = 70)
cat("done\n")
