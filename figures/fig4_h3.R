# Fig. 4 — Which shift varies more? (H3)
# a: posterior SDs (logit scale) of the between-hospital intercept (FPR, H1 model) and between-generator intercept
#    (TPR, H2 model), and of the novelty slopes (how much the shift effect itself varies).
# b: ratio hospital SD / generator SD (intercepts); preregistered H3 predicted ratio >= 1.
source("figures/theme_cxrf.R")
M <- "results/models"
a <- readRDS(file.path(M, "h1_cells_sd_site.rds")); b <- readRDS(file.path(M, "h2_sd_generator.rds"))
n <- min(nrow(a), nrow(b))
long <- bind_rows(
  tibble(what = "Baseline level", source = "Between hospitals (false-positive rate)", sd = a$sd_site__Intercept),
  tibble(what = "Baseline level", source = "Between generators (true-positive rate)", sd = b$sd_generator__Intercept),
  tibble(what = "Size of the shift effect", source = "Between hospitals (false-positive rate)", sd = a$sd_site__novelty_site),
  tibble(what = "Size of the shift effect", source = "Between generators (true-positive rate)", sd = b$sd_generator__novelty)) |>
  mutate(source = factor(source, levels = c("Between generators (true-positive rate)", "Between hospitals (false-positive rate)")))
pa <- ggplot(long, aes(sd, source, fill = source)) +
  cxrf_halfeye(colour = cxrf_pal$ink) +
  scale_fill_manual(values = c(cxrf_pal$unseen, cxrf_pal$seen) |> setNames(levels(long$source))) +
  facet_wrap(~what, ncol = 2) +
  scale_x_continuous("Posterior standard deviation (log-odds scale)", limits = c(0, 4.5)) + scale_y_discrete(NULL) +
  labs(tag = "a", title = "Generators differ more than hospitals",
       subtitle = "Random-effect SDs from the H1 (hospital) and H2 (generator) models; 66% and 95% CrI") +
  theme_cxrf()
r <- tibble(ratio = a$sd_site__Intercept[1:n] / b$sd_generator__Intercept[1:n])
pb <- ggplot(r, aes(ratio, y = 0)) +
  annotate("rect", xmin = 1, xmax = Inf, ymin = -Inf, ymax = Inf, fill = "#F4F4F4") +
  annotate("text", x = 0.93, y = 0.95, label = "H3 predicted\nratio \u2265 1 \u2192", hjust = 1, vjust = 1, size = 2,
           colour = cxrf_pal$ink2, family = cxrf_font, lineheight = 0.9) +
  geom_vline(xintercept = 1, linetype = "22", colour = cxrf_pal$ink2, linewidth = 0.3) +
  cxrf_halfeye(fill = cxrf_pal$highlight, colour = cxrf_pal$ink) +
  scale_x_log10("Hospital SD / generator SD (log scale)", breaks = c(0.1, 0.25, 0.5, 1, 2), limits = c(0.08, 2.5)) +
  scale_y_continuous(NULL, breaks = NULL) +
  labs(tag = "b", title = "H3 not supported",
       subtitle = sprintf("Ratio %.2f [%.2f, %.2f]\nPr(ratio \u2265 1) = %.3f", median(r$ratio), quantile(r$ratio, .025),
                          quantile(r$ratio, .975), mean(r$ratio >= 1))) +
  theme_cxrf()
fig <- pa + pb + plot_layout(widths = c(2, 1.1))
save_cxrf(fig, "figures/out/fig4_h3", width_mm = 183, height_mm = 62)
cat("saved figures/out/fig4_h3.{pdf,png}\n")
