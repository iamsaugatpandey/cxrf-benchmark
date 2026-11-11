# Style demo using PILOT data (not a paper result): hospital shift and generator shift.
# Posterior for each rate: Beta(k + 0.5, n - k + 0.5) (Jeffreys prior), shown as half-eyes
# with 66% / 95% intervals. Run from the project root:
#   Rscript figures/demo_pilot_shift.R
source("figures/theme_cxrf.R")

d <- readr::read_csv("results/pilot_demo_counts.csv", show_col_types = FALSE) |>
  mutate(
    encoder = recode(encoder, dinov3 = "DINOv3", `rad-dino` = "RAD-DINO", biomedclip = "BiomedCLIP"),
    detector = paste0(encoder, ifelse(lbr == "LBR", " + LBR", "")),
    detector = factor(detector, levels = rev(c("DINOv3", "DINOv3 + LBR", "RAD-DINO", "RAD-DINO + LBR",
                                               "BiomedCLIP", "BiomedCLIP + LBR"))),
    shift = ifelse(grepl("unseen", set), "unseen", "seen")
  ) |>
  rowwise() |> mutate(draw = list(rate_draws(k, n))) |> ungroup() |>
  unnest(draw)

lab_df <- function(df, x_seen, x_unseen, txt_seen, txt_unseen) {
  top <- levels(df$detector)[nlevels(df$detector)]
  tibble(detector = factor(top, levels = levels(df$detector)),
         x = c(x_seen, x_unseen), shift = c("seen", "unseen"), label = c(txt_seen, txt_unseen))
}

panel <- function(df, xlab, xmax, ref = NULL, ref_lab = NULL, labs) {
  p <- ggplot(df, aes(y = detector, x = draw, fill = shift, colour = shift)) +
    cxrf_halfeye(height = 0.85)
  if (!is.null(ref)) {
    p <- p + geom_vline(xintercept = ref, linetype = "22", colour = cxrf_pal$ink2, linewidth = 0.35) +
      annotate("text", x = ref, y = 0.55, label = ref_lab, hjust = -0.08, vjust = 1, size = 2.2,
               family = cxrf_font, colour = cxrf_pal$ink2)
  }
  p +
    geom_text(data = labs, aes(x = x, y = detector, label = label, colour = shift),
              inherit.aes = FALSE, nudge_y = 0.62, size = 2.4, family = cxrf_font,
              fontface = "bold", hjust = 0.5) +
    scale_fill_manual(values = c(seen = cxrf_pal$seen, unseen = cxrf_pal$unseen)) +
    scale_colour_manual(values = c(seen = cxrf_pal$seen, unseen = cxrf_pal$unseen)) +
    scale_x_continuous(labels = scales::label_percent(accuracy = 1), limits = c(0, xmax),
                       expand = expansion(mult = c(0, 0.02))) +
    coord_cartesian(clip = "off") +
    labs(x = xlab, y = NULL) +
    theme_cxrf()
}

real <- filter(d, kind == "real")
fake <- filter(d, kind == "fake")

pa <- panel(real, "False-positive rate on real images", 0.42, ref = 0.01, ref_lab = "1% target",
            labs = lab_df(real, 0.03, 0.24, "seen hospitals", "Open-I (unseen)")) +
  ggtitle("   Unseen hospital: real images flagged as fake",
          subtitle = "    Threshold fixed at 1% FPR on held-out training-hospital images")
pb <- panel(fake, "True-positive rate on fake images", 1.0,
            labs = lab_df(fake, 0.90, 0.30, "seen generator", "GPT-image + Gemini")) +
  ggtitle("   Unseen generators: fakes are missed",
          subtitle = "    Same threshold; API generators never seen in training") +
  theme(axis.text.y = element_blank())

fig <- (pa | pb) + plot_layout(widths = c(1, 1)) +
  plot_annotation(tag_levels = "a",
                  caption = paste0("PILOT DATA \u2014 style demonstration only, not a paper result. ",
                                  "Points, posterior medians; thick and thin lines, 66% and 95% credible intervals.\n",
                                  "n = 250 held-out training-hospital and 200 Open-I real images; 368 SynthCheX and 20 API fakes; seed 0."),
                  theme = theme_cxrf())

save_cxrf(fig, "figures/out/demo_pilot_shift", width_mm = 183, height_mm = 88)
cat("saved figures/out/demo_pilot_shift.{pdf,png}\n")
