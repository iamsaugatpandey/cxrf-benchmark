# Frequentist cross-check for H1 (OSF 5vwkj): lme4 binomial GLMM with the same structure as fit_h1.R.
# Output: results/models/h1_glmm.txt (+ .rds). Usage: Rscript analysis/glmm_h1.R
suppressPackageStartupMessages({library(arrow); library(dplyr); library(lme4)})
d <- read_parquet("data/analysis/reals_agg.parquet") |>
  mutate(family = relevel(factor(family), ref = "mlp"),
         view = factor(ifelse(view == "AP", "AP", "PA"), levels = c("PA", "AP")),
         site = factor(site), cfg = factor(paste(family, encoder, sep = ":")))
t0 <- Sys.time()
g <- glmer(cbind(k, n - k) ~ 1 + family * novelty_site + view + (1 + novelty_site | site) + (1 | patient_id) + (1 | cfg),
           data = d, family = binomial, nAGQ = 0, control = glmerControl(optimizer = "nloptwrap"))
saveRDS(g, "results/models/h1_glmm.rds")
# population-level FPR per family x novelty at RE = 0, PA view
nd <- expand.grid(family = levels(d$family), novelty_site = 0:1, view = factor("PA", levels = c("PA","AP")))
nd$fpr <- 100 * predict(g, newdata = nd, re.form = NA, type = "response")
sink("results/models/h1_glmm.txt"); print(summary(g)); print(Sys.time() - t0)
print(reshape(nd[, c("family","novelty_site","fpr")], idvar = "family", timevar = "novelty_site", direction = "wide")); sink()
print(reshape(nd[, c("family","novelty_site","fpr")], idvar = "family", timevar = "novelty_site", direction = "wide"))
