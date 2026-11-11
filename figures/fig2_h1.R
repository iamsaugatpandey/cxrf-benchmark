# Fig. 2 — How large is hospital shift? (H1, primary)
# a: posterior of dFPR (unseen - seen hospital, pp) per detector family, magnitude bands fixed in advance.
# b: FPR on real test images per hospital (rows) and family (panels): seen (blue) vs unseen (red),
#    posterior median + 66/95% CrI, log scale; dashed line at the 1% target.
# Input : results/models/h1_<tag>_draws.rds, h1_<tag>_site_draws.rds   Usage: Rscript figures/fig2_h1.R [tag]
source("figures/theme_cxrf.R")
args <- commandArgs(trailingOnly = TRUE); tag <- if (length(args)) args[1] else "cells"
M <- "results/models"
fam_lab <- c(mlp = "Encoder probe (MLP)", mlp_lbr = "MLP + latent blending", knn = "Encoder kNN",
             resnet = "ResNet-18 (fine-tuned)", format = "Format-feature logistic", ocsvm = "Encoder one-class SVM",
             fft = "Spectrum logistic")
site_lab <- c(nih = "NIH (US)", chexpert = "CheXpert (US)", openi = "Open-I (US)", montgomery = "Montgomery (US)",
              shenzhen = "Shenzhen (CN)", jsrt = "JSRT (JP)", padchest = "PadChest (ES)")

dr <- readRDS(file.path(M, paste0("h1_", tag, "_draws.rds")))
ord <- dr |> group_by(family) |> summarise(m = median(delta_pp)) |> arrange(m) |> pull(family)
dr <- dr |> mutate(fam = factor(fam_lab[family], levels = fam_lab[ord]))

bands <- tibble(xmin = c(-1, 1, 5, 15), xmax = c(1, 5, 15, Inf), lab = c("negligible", "small", "moderate", "large"),
                fill = c("#F4F4F4", "#FAFAFA", "#F4F4F4", "#FAFAFA"))
xmax_a <- max(quantile(dr$delta_pp, .999), 20)
pa <- ggplot(dr, aes(x = delta_pp, y = fam)) +
  geom_rect(data = bands, aes(xmin = xmin, xmax = pmin(xmax, xmax_a * 1.05), ymin = -Inf, ymax = Inf),
            fill = bands$fill, inherit.aes = FALSE) +
  geom_text(data = bands, aes(x = (xmin + pmin(xmax, xmax_a)) / 2, y = Inf, label = lab), vjust = 1.2,
            size = 2, colour = cxrf_pal$ink2, family = cxrf_font, inherit.aes = FALSE) +
  geom_vline(xintercept = 0, colour = cxrf_pal$ref, linewidth = 0.3) +
  cxrf_halfeye(fill = cxrf_pal$unseen, colour = cxrf_pal$ink, density = "unbounded") +
  scale_x_continuous("Change in false-positive rate, unseen − seen hospital (percentage points)",
                     expand = expansion(mult = c(0.01, 0.03))) +
  scale_y_discrete(NULL, expand = expansion(add = c(0.3, 1.1))) +
  coord_cartesian(xlim = c(-2, xmax_a)) +
  labs(tag = "a", title = "False-positive rates were higher at hospitals absent from training",
       subtitle = "Population-level posterior; point = median, bars = 66% and 95% CrI") +
  theme_cxrf()

sd <- readRDS(file.path(M, paste0("h1_", tag, "_site_draws.rds")))
long <- as_tibble(sd$nd) |> mutate(col = row_number()) |>
  filter(!(site == "montgomery" & novelty_site == 0)) |>
  rowwise() |> mutate(med = 100 * median(sd$draws[, col]),
                      lo95 = 100 * quantile(sd$draws[, col], .025), hi95 = 100 * quantile(sd$draws[, col], .975),
                      lo66 = 100 * quantile(sd$draws[, col], .17), hi66 = 100 * quantile(sd$draws[, col], .83)) |>
  ungroup() |>
  mutate(fam = factor(fam_lab[family], levels = rev(fam_lab[ord])),
         site = factor(site_lab[site], levels = rev(site_lab)),
         nov = factor(ifelse(novelty_site == 1, "unseen", "seen"), levels = c("seen", "unseen")))
pb <- ggplot(long, aes(y = site, x = med, colour = nov)) +
  geom_vline(xintercept = 1, linetype = "22", colour = cxrf_pal$ink2, linewidth = 0.3) +
  geom_linerange(aes(xmin = lo95, xmax = hi95), linewidth = 0.4, position = position_dodge(width = 0.6)) +
  geom_linerange(aes(xmin = lo66, xmax = hi66), linewidth = 1.1, position = position_dodge(width = 0.6)) +
  geom_point(size = 0.9, position = position_dodge(width = 0.6)) +
  scale_colour_manual(NULL, values = c(seen = cxrf_pal$seen, unseen = cxrf_pal$unseen)) +
  scale_x_log10("False-positive rate on real test images (%, log scale)", breaks = c(0.1, 1, 10, 100),
                labels = c("0.1", "1", "10", "100")) +
  coord_cartesian(xlim = c(0.05, 100)) +
  facet_wrap(~fam, ncol = 4) + scale_y_discrete(NULL) +
  labs(tag = "b", title = "The increase varied by hospital and detector family",
       subtitle = "Blue: hospital seen in training; red: unseen. Dashed line: the 1% target set on seen-hospital validation images") +
  theme_cxrf(grid_x = TRUE) + theme(panel.spacing.x = unit(6, "pt"))

fig <- pa / pb + plot_layout(heights = c(1, 1.55))
save_cxrf(fig, "figures/out/fig2_h1", width_mm = 183, height_mm = 175)
cat("saved figures/out/fig2_h1.{pdf,png}\n")
