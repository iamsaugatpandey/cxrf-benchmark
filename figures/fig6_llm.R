# Fig. 6 — How do frontier LLMs compare with trained detectors on identical images? (H5)
# a ROC curves (zero-shot, run 1) for six LLMs, with the trained-detector family medians as reference points
# b reliability diagrams (10 equal-mass bins), c test-retest: probability on run 1 vs runs 2-3,
# d discrimination against cost per 1,000 images.
# Inputs: results/scores/llm/llm_scores.csv, results/tables/{llm_metrics,llm_retest,trained_on_llm_subset}.csv
source("figures/theme_cxrf.R")
S <- read.csv("results/scores/llm/llm_scores.csv") |> filter(status == "ok") |> mutate(y = as.integer(cls == "fake"))
met <- read.csv("results/tables/llm_metrics.csv") |> filter(condition == "zeroshot")
tr <- read.csv("results/tables/trained_on_llm_subset.csv")
mlab <- c(`claude-opus-5-5` = "Claude Opus 5.5", `claude-haiku-5-5` = "Claude Haiku 5.5",
          `gpt-6-astra` = "GPT-6 Astra", `gpt-5.4-mini` = "GPT-5.4 mini",
          `gemini-4-argon` = "Gemini 4 Argon", `gemini-3.1-flash-lite` = "Gemini 3.1 Flash-Lite")
mcol <- c(`Claude Opus 5.5` = "#5B3F8C", `Claude Haiku 5.5` = "#A28CC7", `GPT-6 Astra` = "#2E6E5E",
          `GPT-5.4 mini` = "#8DBFAF", `Gemini 4 Argon` = "#B5562B", `Gemini 3.1 Flash-Lite` = "#E0A27E")
z <- S |> filter(condition == "zeroshot", run == 1) |> mutate(m = factor(mlab[model], levels = mlab))

roc <- z |> group_by(m) |> group_modify(function(g, k) {
  th <- c(Inf, sort(unique(g$probability), decreasing = TRUE), -Inf)
  tibble(fpr = sapply(th, function(t) mean(g$probability[g$y == 0] > t)),
         tpr = sapply(th, function(t) mean(g$probability[g$y == 1] > t)))
}) |> ungroup()
auc_lab <- met |> mutate(m = factor(mlab[model], levels = mlab)) |>
  transmute(m, lab = sprintf("%s  %.2f [%.2f, %.2f]", m, auc, auc_lo95, auc_hi95))
pa <- ggplot(roc, aes(fpr, tpr, colour = m)) +
  geom_abline(colour = cxrf_pal$ref, linewidth = 0.3, linetype = "22") +
  # straight segments between operating points: with tied scores (models reuse round numbers) the ROC between
  # two thresholds is the diagonal chord, so a step plot would overstate discrimination
  geom_path(linewidth = 0.5) +
  scale_colour_manual(NULL, values = mcol, labels = setNames(auc_lab$lab, auc_lab$m)) +
  scale_x_continuous("False-positive rate", labels = scales::percent, expand = c(0, 0)) +
  scale_y_continuous("True-positive rate", labels = scales::percent, expand = c(0, 0)) +
  coord_equal() + labs(tag = "a", title = "Only one model clearly beat chance", subtitle = "Zero-shot, 2,000 identical test images; AUC [95% CI]") +
  theme_cxrf(grid_y = TRUE) + theme(legend.position = "inside", legend.position.inside = c(0.99, 0.02),
                                    legend.justification = c(1, 0), legend.key.height = unit(6, "pt"), legend.key.width = unit(8, "pt"),
                                    legend.text = element_text(size = 5.5), legend.background = element_rect(fill = alpha("white", 0.9), colour = NA))

# equal-mass bins on the stated probability; tied values stay in one bin (models reuse a few round numbers),
# so a model can have fewer than 10 points
rel <- z |> group_by(m) |>
  mutate(bin = as.integer(cut(probability, unique(quantile(probability, seq(0, 1, 0.1))), include.lowest = TRUE))) |>
  group_by(m, bin) |>
  summarise(pred = mean(probability) / 100, obs = mean(y), n = n(), .groups = "drop")
pb <- ggplot(rel, aes(pred, obs, colour = m)) +
  geom_abline(colour = cxrf_pal$ref, linewidth = 0.3, linetype = "22") +
  geom_line(linewidth = 0.45) + geom_point(aes(size = n)) + scale_size_area(max_size = 2.2, guide = "none") +
  scale_colour_manual(NULL, values = mcol) +
  scale_x_continuous("Stated probability of AI origin", labels = scales::percent, limits = c(0, 1)) +
  scale_y_continuous("Observed fraction AI-generated", labels = scales::percent, limits = c(0, 1)) +
  coord_equal() + labs(tag = "b", title = "Stated probabilities were poorly calibrated", subtitle = "Up to 10 equal-mass bins (ties kept together); point size = images") +
  theme_cxrf(grid_y = TRUE) + theme(legend.position = "none")

rt <- S |> filter(condition == "zeroshot") |> group_by(model, image_id) |> filter(n() == 3) |>
  summarise(sd = sd(probability), .groups = "drop") |> mutate(m = factor(mlab[model], levels = rev(mlab)))
pc <- ggplot(rt, aes(x = sd, y = m, fill = m)) +
  stat_dots(quantiles = 50, side = "top", colour = NA, binwidth = 1.2, overflow = "compress", scale = 0.85) +
  stat_pointinterval(.width = c(.66, .95), point_size = 1, colour = cxrf_pal$ink, position = position_nudge(y = -0.1)) +
  scale_fill_manual(values = mcol) +
  scale_x_continuous("SD across 3 runs of the same image (probability points)") + scale_y_discrete(NULL) +
  labs(tag = "c", title = "Same image, different answers", subtitle = "403 images x 3 zero-shot runs; 1 dot = 2% of images") +
  theme_cxrf() + theme(legend.position = "none")

trm <- tr |> mutate(lab = c(fft = "Spectrum", format = "Format", knn = "kNN", mlp = "MLP", mlp_lbr = "MLP+LBR",
                            ocsvm = "OC-SVM", resnet = "ResNet-18")[family])
pd <- ggplot(met |> mutate(m = factor(mlab[model], levels = mlab)), aes(cost_per_1000, auc, colour = m)) +
  geom_hline(yintercept = 0.5, colour = cxrf_pal$ref, linewidth = 0.3, linetype = "22") +
  annotate("rect", xmin = 0.03, xmax = 15, ymin = min(trm$auc_med[trm$family != "ocsvm"]),
           ymax = max(trm$auc_med[trm$family != "ocsvm"]), fill = cxrf_pal$seen, alpha = 0.12) +
  annotate("text", x = 0.035, y = max(trm$auc_med[trm$family != "ocsvm"]) + 0.004, hjust = 0, vjust = 0, size = 1.9,
           colour = cxrf_pal$seen, family = cxrf_font,
           label = sprintf("trained detector families, median AUC %.2f–%.2f (same images)",
                           min(trm$auc_med[trm$family != "ocsvm"]), max(trm$auc_med[trm$family != "ocsvm"]))) +
  geom_linerange(aes(ymin = auc_lo95, ymax = auc_hi95), linewidth = 0.4) + geom_point(size = 1.4) +
  ggrepel::geom_text_repel(aes(label = m), size = 1.9, family = cxrf_font, seed = 1, min.segment.length = 0.3,
                           box.padding = 0.35, segment.size = 0.2, show.legend = FALSE) +
  scale_colour_manual(NULL, values = mcol) +
  scale_x_log10("Cost per 1,000 images (US$, batch, log scale)", labels = scales::label_dollar(accuracy = 0.01)) +
  scale_y_continuous("AUC", limits = c(0.4, 0.9)) +
  labs(tag = "d", title = "Higher price did not buy accuracy", subtitle = "Point = zero-shot AUC, bar = 95% CI; dashed line = chance") +
  theme_cxrf(grid_y = TRUE) + theme(legend.position = "none")

fig <- (pa | pb) / (pc | pd) + plot_layout(heights = c(1.15, 1))
save_cxrf(fig, "figures/out/fig6_llm", width_mm = 183, height_mm = 170)
cat("saved figures/out/fig6_llm.{pdf,png}\n")
