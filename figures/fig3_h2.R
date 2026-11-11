# Fig. 3 — How large is generator shift? (H2)
# a: TPR (% of fakes flagged at the frozen 1%-FPR threshold) by training generator (rows) x test set (columns),
#    per detector family; empirical, pooled over hospital configurations, encoders and seeds.
# b: posterior dTPR (unseen - seen generator, pp) per family from the H2 model (if fitted).
# Inputs: data/analysis/fakes_long.parquet, results/models/h2_draws.rds    Usage: Rscript figures/fig3_h2.R
source("figures/theme_cxrf.R")
suppressPackageStartupMessages(library(arrow))
fam_lab <- c(mlp = "Encoder probe (MLP)", mlp_lbr = "MLP + latent blending", knn = "Encoder kNN",
             resnet = "ResNet-18 (fine-tuned)", format = "Format-feature logistic", ocsvm = "Encoder one-class SVM",
             fft = "Spectrum logistic")
F <- read_parquet("data/analysis/fakes_long.parquet", col_select = c("family", "gen_cfg", "generator", "conditioning", "flag"))
cells <- F |> mutate(test = case_when(conditioning == "image" ~ paste(generator, "edit"),
                                      TRUE ~ paste(generator, ifelse(generator %in% c("G4", "G5"), "text", "")))) |>
  group_by(family, gen_cfg, test) |> summarise(tpr = 100 * mean(flag), .groups = "drop") |>
  mutate(train = factor(c(g1 = "G1", g2 = "G2", g12 = "G1 + G2")[gen_cfg], levels = c("G1 + G2", "G2", "G1")),
         test = factor(trimws(test), levels = c("G1", "G2", "G4 text", "G5 text", "G4 edit", "G5 edit")),
         fam = factor(fam_lab[family], levels = fam_lab),
         seen = (train == "G1" & test == "G1") | (train == "G2" & test == "G2") | (train == "G1 + G2" & test %in% c("G1", "G2"))) |>
  complete(fam, train, test) |>  # ResNet-18 was fitted with pooled G1 + G2 fakes only: other rows shown as not fitted
  mutate(seen = coalesce(seen, FALSE))
pa <- ggplot(cells, aes(test, train, fill = tpr)) +
  geom_tile(colour = "white", linewidth = 0.6) +
  geom_tile(data = filter(cells, seen), fill = NA, colour = cxrf_pal$seen, linewidth = 0.5) +
  geom_text(aes(label = ifelse(is.na(tpr), "–", sprintf("%.0f", tpr)), colour = !is.na(tpr) & tpr > 55), size = 2, family = cxrf_font) +
  scale_colour_manual(values = c(`TRUE` = "white", `FALSE` = cxrf_pal$ink), guide = "none") +
  scale_fill_gradient(low = "#F7F3EE", high = cxrf_pal$unseen, limits = c(0, 100), name = "TPR (%)", na.value = "#EDEDED") +
  facet_wrap(~fam, ncol = 4) +
  scale_x_discrete("Test fakes (G4 gpt-image-2, G5 Gemini: never trained on; edit = image-conditioned)", position = "bottom") +
  scale_y_discrete("Training fakes") +
  labs(tag = "a", title = "Fakes from generators absent from training were mostly missed",
       subtitle = "% of fakes flagged at the threshold fixed for 1% false positives; outlined = generator seen in training; – = not fitted") +
  theme_cxrf(grid_x = FALSE) +
  theme(legend.position = "right", legend.key.width = unit(4, "pt"), legend.key.height = unit(18, "pt"),
        axis.text.x = element_text(angle = 35, hjust = 1), panel.spacing = unit(5, "pt"))

if (file.exists("results/models/h2_draws.rds")) {
  dr <- readRDS("results/models/h2_draws.rds")
  ord <- dr |> group_by(family) |> summarise(m = median(delta_pp)) |> arrange(desc(m)) |> pull(family)
  dr <- mutate(dr, fam = factor(fam_lab[family], levels = fam_lab[ord]))
  pb <- ggplot(dr, aes(delta_pp, fam)) +
    annotate("rect", xmin = -1, xmax = 1, ymin = -Inf, ymax = Inf, fill = "#F4F4F4") +
    geom_vline(xintercept = 0, colour = cxrf_pal$ref, linewidth = 0.3) +
    cxrf_halfeye(fill = cxrf_pal$unseen, colour = cxrf_pal$ink, density = "unbounded") +
    scale_x_continuous("Change in true-positive rate, unseen − seen generator (percentage points)") +
    scale_y_discrete(NULL) +
    labs(tag = "b", title = "Model-based change for unseen generators", subtitle = "Text-prompted fakes; population-level posterior, 66% and 95% CrI") +
    theme_cxrf()
  fig <- pa / pb + plot_layout(heights = c(1.5, 1))
  save_cxrf(fig, "figures/out/fig3_h2", width_mm = 183, height_mm = 185)
} else {
  save_cxrf(pa, "figures/out/fig3_h2", width_mm = 183, height_mm = 110)
}
cat("saved figures/out/fig3_h2.{pdf,png}\n")
