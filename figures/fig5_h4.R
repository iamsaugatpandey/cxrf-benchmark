# Fig. 5 — Do image-conditioned fakes, which inherit a hospital's appearance, evade detectors? (H4)
# a: TPR of commercial fakes per detector family: text-prompted (grey reference) vs image-conditioned, split by the
#    hospital whose normal radiograph was edited; empirical with cluster-bootstrap 95% CIs over source images/reports.
# b: posterior dTPR (image-conditioned - text-only), H4a; and (if fitted) H4b (conditioning hospital unseen - seen).
source("figures/theme_cxrf.R")
suppressPackageStartupMessages(library(arrow))
M <- "results/models"
fam_lab <- c(mlp = "Encoder probe (MLP)", mlp_lbr = "MLP + latent blending", knn = "Encoder kNN",
             resnet = "ResNet-18 (fine-tuned)", format = "Format-feature logistic", ocsvm = "Encoder one-class SVM",
             fft = "Spectrum logistic")
site_lab <- c(nih = "NIH", openi = "Open-I", shenzhen = "Shenzhen", montgomery = "Montgomery")
F <- read_parquet("data/analysis/fakes_long.parquet",
                  col_select = c("image_id", "group_id", "family", "generator", "conditioning", "cond_site", "flag")) |>
  filter(generator %in% c("G4", "G5"))
boot_ci <- function(df, B = 1000) {  # cluster bootstrap over source group (report or source image)
  g <- df |> group_by(group_id) |> summarise(k = sum(flag), n = n(), .groups = "drop")
  reps <- replicate(B, { i <- sample.int(nrow(g), replace = TRUE); sum(g$k[i]) / sum(g$n[i]) })
  tibble(tpr = 100 * sum(g$k) / sum(g$n), lo = 100 * quantile(reps, .025), hi = 100 * quantile(reps, .975))
}
set.seed(20261109)
txt <- F |> filter(conditioning == "text") |> group_by(family) |> group_modify(~boot_ci(.x)) |> ungroup()
img <- F |> filter(conditioning == "image") |> group_by(family, cond_site) |> group_modify(~boot_ci(.x)) |> ungroup()
img <- img |> mutate(site = factor(site_lab[cond_site], levels = rev(site_lab)), fam = factor(fam_lab[family], levels = fam_lab))
txt <- txt |> mutate(fam = factor(fam_lab[family], levels = fam_lab))
pa <- ggplot(img, aes(y = site)) +
  geom_rect(data = txt, aes(xmin = lo, xmax = hi, ymin = -Inf, ymax = Inf), inherit.aes = FALSE, fill = cxrf_pal$grid) +
  geom_vline(data = txt, aes(xintercept = tpr), colour = cxrf_pal$ink2, linewidth = 0.4) +
  geom_linerange(aes(xmin = lo, xmax = hi), colour = cxrf_pal$unseen, linewidth = 0.5) +
  geom_point(aes(x = tpr), colour = cxrf_pal$unseen, size = 1.2) +
  facet_wrap(~fam, ncol = 4) +
  scale_x_continuous("True-positive rate on commercial fakes (%)", limits = c(0, NA)) + scale_y_discrete("Edited radiograph's hospital") +
  labs(tag = "a", title = "Edited radiographs are detected about as rarely as text-prompted fakes",
       subtitle = "Red: image-conditioned fakes by source hospital (95% bootstrap CI); grey line and band: text-prompted fakes") +
  theme_cxrf()
dr <- readRDS(file.path(M, "h4a_draws.rds")) |> mutate(h = "Image-conditioned − text-only (H4a)")
if (file.exists(file.path(M, "h4b_draws.rds")))
  dr <- bind_rows(dr, readRDS(file.path(M, "h4b_draws.rds")) |> mutate(h = "Source hospital unseen − seen (H4b)"))
dr <- dr |> mutate(fam = factor(fam_lab[family], levels = rev(fam_lab)))
pb <- ggplot(dr, aes(delta_pp, fam)) +
  annotate("rect", xmin = -1, xmax = 1, ymin = -Inf, ymax = Inf, fill = "#F4F4F4") +
  geom_vline(xintercept = 0, colour = cxrf_pal$ref, linewidth = 0.3) +
  cxrf_halfeye(fill = cxrf_pal$unseen, colour = cxrf_pal$ink, density = "unbounded") +
  facet_wrap(~h, ncol = 2) +
  scale_x_continuous("Change in true-positive rate (percentage points)") + scale_y_discrete(NULL) +
  labs(tag = "b", title = "Model-based contrasts", subtitle = "Population-level posterior; 66% and 95% CrI; grey band = negligible (±1 pp)") +
  theme_cxrf()
fig <- pa / pb + plot_layout(heights = c(1.25, 1))
save_cxrf(fig, "figures/out/fig5_h4", width_mm = 183, height_mm = 165)
cat("saved figures/out/fig5_h4.{pdf,png}\n")
