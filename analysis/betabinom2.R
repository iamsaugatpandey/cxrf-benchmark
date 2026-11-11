# Beta-binomial likelihood as a brms custom family (mean mu on the logit scale, precision phi on the log scale).
# Why custom: brms' built-in beta_binomial failed at initialisation with Stan 2.40 on our cell data, although each
# row evaluated finitely (bisected 2026-11-09). This writes the log-pmf out explicitly:
#   log p(y | T, mu, phi) = log C(T, y) + lbeta(y + mu*phi, T - y + (1 - mu)*phi) - lbeta(mu*phi, (1 - mu)*phi)
# E[y / T] = mu, so posterior_epred with trials = 1 returns the population rate directly (no averaging needed),
# and Var(y) = T mu (1 - mu) [1 + (T - 1) / (phi + 1)] absorbs within-cell clustering (images of one patient).
# Usage: source("analysis/betabinom2.R"); brm(k | vint(n) ~ ..., family = betabinom2, stanvars = betabinom2_stanvars)
betabinom2 <- brms::custom_family(
  "betabinom2", dpars = c("mu", "phi"), links = c("logit", "log"),
  lb = c(0, 0), ub = c(1, NA), type = "int", vars = "vint1[n]")
betabinom2_stanvars <- brms::stanvar(block = "functions", scode = "
  real betabinom2_lpmf(int y, real mu, real phi, int T) {
    return lchoose(T, y) + lbeta(y + mu * phi, T - y + (1 - mu) * phi) - lbeta(mu * phi, (1 - mu) * phi);
  }
  int betabinom2_rng(real mu, real phi, int T) {
    return beta_binomial_rng(T, mu * phi, (1 - mu) * phi);
  }
")
log_lik_betabinom2 <- function(i, prep) {
  mu <- brms::get_dpar(prep, "mu", i = i); phi <- brms::get_dpar(prep, "phi", i = i)
  y <- prep$data$Y[i]; T <- prep$data$vint1[i]
  lchoose(T, y) + lbeta(y + mu * phi, T - y + (1 - mu) * phi) - lbeta(mu * phi, (1 - mu) * phi)
}
posterior_predict_betabinom2 <- function(i, prep, ...) {
  mu <- brms::get_dpar(prep, "mu", i = i); phi <- brms::get_dpar(prep, "phi", i = i)
  rbinom(length(mu), prep$data$vint1[i], rbeta(length(mu), mu * phi, (1 - mu) * phi))
}
posterior_epred_betabinom2 <- function(prep) {
  mu <- brms::get_dpar(prep, "mu")
  mu * matrix(prep$data$vint1, nrow = nrow(mu), ncol = ncol(mu), byrow = TRUE)
}
