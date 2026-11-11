# Fig. 7 — Are the main results robust to the preregistered sensitivity analyses? (S1–S9)
# a: H1 dFPR (unseen - seen hospital) and b: H2 dTPR (unseen - seen generator), population-level posterior
#    median + 66%/95% CrI, for the three families that supported H1 (encoder probe, latent blending, ResNet-18),
#    across the primary analysis and every sensitivity analysis that applies to that hypothesis.
# Inputs: results/models/{h1_cells,h1_S*,h2,h2_S*}_contrasts.csv. ResNet-18 is absent where it was not refitted
# (S2, S3: masked/JPEG images were scored by frozen encoders only; S4: 512 px).
source("figures/theme_cxrf.R")
M <- "results/models"
fam_lab <- c(mlp = "Encoder probe (MLP)", mlp_lbr = "MLP + latent blending", resnet = "ResNet-18 (fine-tuned)")
fam_col <- c("Encoder probe (MLP)" = "#1B7F79", "MLP + latent blending" = "#7FBFB9", "ResNet-18 (fine-tuned)" = "#C98B2B")
an_lab <- c(primary = "Primary analysis", S1 = "S1  PA views only", S2 = "S2  Markers and borders masked",
            S3 = "S3  JPEG-harmonized", S4 = "S4  512 px (Open-I excluded)", S5 = "S5  Priors twice as wide",
            S6 = "S6  Threshold on all-hospital reals", S7 = "S7  Memorized prompts removed",
            S9 = "S9  Test-split prompts only")
rd <- function(file, an, hyp) read.csv(file.path(M, file)) |> filter(family %in% names(fam_lab)) |>
  transmute(hyp, an, family, med = delta_pp_med, lo95 = delta_pp_lo95, hi95 = delta_pp_hi95,
            lo66 = delta_pp_lo66, hi66 = delta_pp_hi66)
h1 <- bind_rows(rd("h1_cells_contrasts.csv", "primary", "h1"),
                lapply(paste0("S", 1:6), \(s) rd(sprintf("h1_%s_contrasts.csv", s), s, "h1")))
h2 <- bind_rows(rd("h2_contrasts.csv", "primary", "h2"),
                lapply(paste0("S", c(2:7, 9)), \(s) rd(sprintf("h2_%s_contrasts.csv", s), s, "h2")))
prep <- function(d) d |> mutate(an = factor(an_lab[an], levels = rev(an_lab)),
                                fam = factor(fam_lab[family], levels = fam_lab)) |> droplevels()
h1 <- prep(h1); h2 <- prep(h2)

forest <- function(d, xlab, hl) {
  ggplot(d, aes(y = an, colour = fam)) +
    annotate("rect", xmin = -1, xmax = 1, ymin = -Inf, ymax = Inf, fill = "#F4F4F4") +
    { if (!is.null(hl)) annotate("rect", xmin = -Inf, xmax = Inf, ymin = hl - 0.5, ymax = hl + 0.5,
                                 fill = cxrf_pal$highlight, alpha = 0.15) } +
    geom_vline(xintercept = 0, colour = cxrf_pal$ref, linewidth = 0.3) +
    geom_linerange(aes(xmin = lo95, xmax = hi95), linewidth = 0.35, position = position_dodge(width = 0.7)) +
    geom_linerange(aes(xmin = lo66, xmax = hi66), linewidth = 1.1, position = position_dodge(width = 0.7)) +
    geom_point(aes(x = med, shape = fam), size = 1.3, fill = "white", stroke = 0.5,
               position = position_dodge(width = 0.7)) +
    scale_colour_manual(NULL, values = fam_col) + scale_shape_manual(NULL, values = c(21, 22, 24)) +
    scale_x_continuous(xlab) + scale_y_discrete(NULL) +
    theme_cxrf()
}
hl1 <- which(levels(h1$an) == an_lab[["S6"]])
pa <- forest(h1, "Change in false-positive rate,\nunseen − seen hospital (percentage points)", hl1) +
  labs(tag = "a", title = "Hospital shift (H1)",
       subtitle = "Higher false-positive rate in every analysis;\nmuch smaller when thresholds used all hospitals (S6, gold)")
pb <- forest(h2, "Change in true-positive rate,\nunseen − seen generator (percentage points)", NULL) +
  labs(tag = "b", title = "Generator shift (H2)",
       subtitle = "Lower true-positive rate in every analysis\n(text-prompted fakes)")
fig <- (pa | pb) + plot_layout(guides = "collect") & theme(legend.position = "bottom", legend.key.width = unit(10, "pt"), plot.margin = margin(4, 14, 4, 4))
save_cxrf(fig, "figures/out/fig7_multiverse", width_mm = 183, height_mm = 100)
cat("saved figures/out/fig7_multiverse.{pdf,png}\n")
