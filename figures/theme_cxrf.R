# theme_cxrf.R — the single figure theme for the manuscript (Style Guide §6).
# Every figure script does: source("figures/theme_cxrf.R")
# Nature Portfolio spec: Helvetica, 7 pt text, 8 pt bold lowercase panel tags,
# widths 89 mm (single) / 120–136 mm (1.5) / 183 mm (double), vector PDF output.

suppressPackageStartupMessages({
  library(ggplot2); library(ggdist); library(distributional)
  library(dplyr); library(tidyr); library(patchwork); library(forcats)
})

cxrf_font <- "Helvetica"

# ---- Palette: one meaning per color, fixed across the whole paper ----
cxrf_pal <- list(
  seen      = "#2B5C8A",  # real images / seen (in-distribution)
  unseen    = "#D1495B",  # fake images / unseen (out-of-distribution)
  encoder   = c(dinov3 = "#1B7F79", `rad-dino` = "#4FA39E", biomedclip = "#8CC7C1"),
  pixel     = "#C98B2B",
  llm       = c(claude = "#5B3F8C", gpt = "#8668B5", gemini = "#B6A3D6"),
  ink       = "#222222",
  ink2      = "#6E6E6E",
  ref       = "#BDBDBD",
  grid      = "#EBEBEB",
  highlight = "#E3B23C"
)

theme_cxrf <- function(base_size = 7, grid_x = TRUE, grid_y = FALSE) {
  theme_minimal(base_size = base_size, base_family = cxrf_font) %+replace%
    theme(
      text              = element_text(colour = cxrf_pal$ink),
      axis.text         = element_text(colour = cxrf_pal$ink2, size = base_size),
      axis.title        = element_text(size = base_size + 1),
      axis.title.x      = element_text(margin = margin(t = 4)),
      axis.title.y      = element_text(margin = margin(r = 4), angle = 90),
      axis.ticks        = element_line(colour = cxrf_pal$ref, linewidth = 0.25),
      axis.ticks.length = unit(1.5, "pt"),
      panel.grid.minor  = element_blank(),
      panel.grid.major.x = if (grid_x) element_line(colour = cxrf_pal$grid, linewidth = 0.25) else element_blank(),
      panel.grid.major.y = if (grid_y) element_line(colour = cxrf_pal$grid, linewidth = 0.25) else element_blank(),
      strip.text        = element_text(face = "bold", size = base_size + 0.5, hjust = 0,
                                       margin = margin(b = 3)),
      plot.title        = element_text(face = "bold", size = base_size + 1, hjust = 0,
                                       margin = margin(b = 2, l = 9)),   # l: room for the panel tag
      plot.subtitle     = element_text(colour = cxrf_pal$ink2, size = base_size, hjust = 0,
                                       margin = margin(b = 4, l = 9)),
      plot.caption      = element_text(colour = cxrf_pal$ink2, size = base_size - 1, hjust = 0),
      plot.tag          = element_text(face = "bold", size = 8, family = cxrf_font),
      plot.tag.position = c(0, 1),
      plot.title.position = "plot",
      plot.caption.position = "plot",
      legend.position   = "none",
      legend.title      = element_text(size = base_size, colour = cxrf_pal$ink),
      legend.text       = element_text(size = base_size - 0.5, colour = cxrf_pal$ink2),
      plot.margin       = margin(4, 6, 4, 4)
    )
}

# Interval geometry convention: point = median, thick = 66%, thin = 95% (ggdist).
cxrf_halfeye <- function(..., density = "bounded") {
  # Use with posterior DRAWS (x = draw). For rates near 0 or 1, the bounded KDE
  # avoids the infinite-density spikes an analytic Beta density would draw.
  stat_halfeye(.width = c(0.66, 0.95), point_size = 1.4, interval_size_range = c(0.6, 1.6),
               slab_alpha = 0.55, normalize = "groups", density = density, ...)
}

# Posterior draws for a binomial rate (Jeffreys prior Beta(k + 0.5, n - k + 0.5)).
rate_draws <- function(k, n, ndraws = 4000, seed = 1) {
  set.seed(seed)
  stats::rbeta(ndraws, k + 0.5, n - k + 0.5)
}

# Save vector PDF (submission) + PNG preview (for review), sizes in mm.
save_cxrf <- function(plot, file_stem, width_mm = 183, height_mm = 90) {
  dir.create(dirname(file_stem), recursive = TRUE, showWarnings = FALSE)
  ggsave(paste0(file_stem, ".pdf"), plot, width = width_mm, height = height_mm, units = "mm",
         device = cairo_pdf)
  ggsave(paste0(file_stem, ".png"), plot, width = width_mm, height = height_mm, units = "mm",
         dpi = 400, bg = "white")
  invisible(file_stem)
}
