# =============================================================
# Reliability & Statistical Analysis — R
# Machine Health Monitoring & Fault Diagnosis System
# =============================================================
# Covers:
#   1. Load and inspect the scored dataset
#   2. Reliability analysis (Weibull, survival curves)
#   3. Correlation & trend analysis
#   4. Statistical hypothesis testing (healthy vs fault)
#   5. Visualisations (ggplot2)
#   6. Export summary tables to CSV
#
# Requirements:
#   install.packages(c("ggplot2","dplyr","tidyr","survival",
#                      "survminer","corrplot","RSQLite","DBI","lubridate"))
#
# Usage:
#   source("r/reliability_analysis.R")   # from project root
#   OR: Rscript r/reliability_analysis.R
# =============================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(survival)
  library(corrplot)
  library(RSQLite)
  library(DBI)
})

# ── Paths ─────────────────────────────────────────────────────────────────────
root_dir   <- normalizePath(file.path(dirname(sys.frame(1)$ofile), ".."), mustWork = FALSE)
proc_dir   <- file.path(root_dir, "data", "processed")
db_path    <- file.path(proc_dir, "cmdb.sqlite")
output_dir <- file.path(root_dir, "reports", "r_plots")
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

cat("=============================================================\n")
cat("  Reliability & Statistical Analysis\n")
cat("  Machine Health Monitoring & Fault Diagnosis System\n")
cat("=============================================================\n\n")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────────────────────────────────────

load_data <- function() {
  # Prefer the scored CSV; fall back to clean dataset
  scored <- file.path(proc_dir, "ai4i_health_scored.csv")
  clean  <- file.path(proc_dir, "ai4i_clean.csv")

  if (file.exists(scored)) {
    df <- read.csv(scored, stringsAsFactors = FALSE)
    cat(sprintf("[data] Loaded scored dataset: %d rows × %d cols\n", nrow(df), ncol(df)))
  } else if (file.exists(clean)) {
    df <- read.csv(clean, stringsAsFactors = FALSE)
    cat(sprintf("[data] Loaded clean dataset:  %d rows × %d cols\n", nrow(df), ncol(df)))
  } else {
    stop("[ERROR] No dataset found. Run the Python pipeline first (python main.py --steps 1 2 3 4).")
  }

  # Ensure key columns exist
  if (!"machine_failure" %in% names(df)) df$machine_failure <- 0L
  if (!"fault_type"      %in% names(df)) {
    df$fault_type <- ifelse(df$machine_failure == 1, "Fault", "Normal")
  }
  if (!"health_status"   %in% names(df)) df$health_status <- "Unknown"
  if (!"tool_wear_min"   %in% names(df)) df$tool_wear_min <- NA_real_

  return(df)
}

df <- load_data()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Descriptive Statistics
# ─────────────────────────────────────────────────────────────────────────────

cat("\n── Descriptive Statistics ───────────────────────────────────\n")
sensor_cols <- c("rotational_speed_rpm", "torque_Nm", "air_temp_K",
                 "process_temp_K", "tool_wear_min")
sensor_cols <- intersect(sensor_cols, names(df))

stats_table <- df %>%
  select(all_of(sensor_cols)) %>%
  pivot_longer(everything(), names_to = "feature", values_to = "value") %>%
  group_by(feature) %>%
  summarise(
    n       = n(),
    mean    = round(mean(value, na.rm = TRUE), 3),
    sd      = round(sd(value,   na.rm = TRUE), 3),
    min     = round(min(value,  na.rm = TRUE), 3),
    q25     = round(quantile(value, 0.25, na.rm = TRUE), 3),
    median  = round(median(value,   na.rm = TRUE), 3),
    q75     = round(quantile(value, 0.75, na.rm = TRUE), 3),
    max     = round(max(value,  na.rm = TRUE), 3),
    .groups = "drop"
  )

print(as.data.frame(stats_table))
write.csv(stats_table,
          file.path(output_dir, "descriptive_stats.csv"),
          row.names = FALSE)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Reliability Analysis — Weibull / Kaplan-Meier
# ─────────────────────────────────────────────────────────────────────────────

cat("\n── Reliability Analysis ─────────────────────────────────────\n")

# Use tool_wear_min as a proxy for "time to failure"
if ("tool_wear_min" %in% names(df) && any(!is.na(df$tool_wear_min))) {

  rel_df <- df %>%
    select(tool_wear_min, machine_failure) %>%
    filter(!is.na(tool_wear_min)) %>%
    mutate(
      time  = pmax(tool_wear_min, 1),   # avoid 0-time
      event = as.integer(machine_failure)
    )

  # Kaplan-Meier survival estimate
  km_fit <- survfit(Surv(time, event) ~ 1, data = rel_df)

  km_df <- data.frame(
    time     = km_fit$time,
    survival = km_fit$surv,
    lower    = km_fit$lower,
    upper    = km_fit$upper
  )

  cat(sprintf("[KM]  Median survival (tool wear at 50%% failure): %.1f min\n",
              quantile(km_fit, probs = 0.5)$quantile))

  # Plot KM curve
  p_km <- ggplot(km_df, aes(x = time, y = survival)) +
    geom_ribbon(aes(ymin = lower, ymax = upper), fill = "#2980b9", alpha = 0.15) +
    geom_step(color = "#2980b9", linewidth = 1) +
    geom_hline(yintercept = 0.5, linetype = "dashed", color = "#e74c3c", linewidth = 0.7) +
    scale_y_continuous(labels = scales::percent_format(), limits = c(0, 1)) +
    labs(
      title    = "Kaplan-Meier Survival Curve",
      subtitle = "Time to failure proxy: Tool Wear (minutes)",
      x        = "Tool Wear (min)",
      y        = "Reliability R(t)",
      caption  = "Dashed line = 50% reliability threshold"
    ) +
    theme_bw(base_size = 12) +
    theme(plot.title = element_text(face = "bold"))

  ggsave(file.path(output_dir, "km_survival_curve.png"),
         p_km, width = 8, height = 5, dpi = 150)
  cat("[plot] KM survival curve saved\n")

} else {
  cat("[WARN] tool_wear_min not available — skipping survival analysis\n")
}


# ─────────────────────────────────────────────────────────────────────────────
# 4. Fault Rate by Operating Condition
# ─────────────────────────────────────────────────────────────────────────────

cat("\n── Fault Rate Analysis ──────────────────────────────────────\n")

if ("fault_type" %in% names(df)) {
  fault_dist <- df %>%
    count(fault_type, name = "count") %>%
    mutate(
      pct    = round(count / sum(count) * 100, 2),
      is_fault = fault_type != "Normal"
    ) %>%
    arrange(desc(count))

  cat("Fault distribution:\n")
  print(as.data.frame(fault_dist))

  p_fault <- ggplot(fault_dist, aes(x = reorder(fault_type, count), y = count,
                                     fill = is_fault)) +
    geom_col(width = 0.65) +
    geom_text(aes(label = paste0(pct, "%")), hjust = -0.1, size = 3.5) +
    scale_fill_manual(values = c("TRUE" = "#e74c3c", "FALSE" = "#27ae60"),
                      guide = "none") +
    coord_flip() +
    labs(title = "Fault Type Distribution", x = NULL, y = "Count") +
    theme_bw(base_size = 12) +
    theme(plot.title = element_text(face = "bold")) +
    expand_limits(y = max(fault_dist$count) * 1.12)

  ggsave(file.path(output_dir, "fault_distribution.png"),
         p_fault, width = 8, height = 4.5, dpi = 150)
  cat("[plot] Fault distribution saved\n")
}


# ─────────────────────────────────────────────────────────────────────────────
# 5. Statistical Tests — Healthy vs Fault
# ─────────────────────────────────────────────────────────────────────────────

cat("\n── Hypothesis Testing (Normal vs Fault) ─────────────────────\n")

if ("machine_failure" %in% names(df) && length(sensor_cols) > 0) {
  normal <- df %>% filter(machine_failure == 0)
  fault  <- df %>% filter(machine_failure == 1)

  test_results <- lapply(sensor_cols, function(col) {
    x <- normal[[col]]
    y <- fault[[col]]
    x <- x[!is.na(x)]; y <- y[!is.na(y)]
    if (length(x) < 2 || length(y) < 2) return(NULL)
    test <- wilcox.test(x, y, exact = FALSE)
    data.frame(
      feature    = col,
      mean_normal= round(mean(x), 3),
      mean_fault = round(mean(y), 3),
      p_value    = round(test$p.value, 6),
      significant = test$p.value < 0.05
    )
  })

  test_df <- do.call(rbind, Filter(Negate(is.null), test_results))
  cat("Wilcoxon rank-sum test (Normal vs Fault):\n")
  print(as.data.frame(test_df))

  write.csv(test_df,
            file.path(output_dir, "hypothesis_tests.csv"),
            row.names = FALSE)
}


# ─────────────────────────────────────────────────────────────────────────────
# 6. Correlation Analysis
# ─────────────────────────────────────────────────────────────────────────────

cat("\n── Correlation Analysis ─────────────────────────────────────\n")

numeric_cols <- df %>%
  select(where(is.numeric)) %>%
  select(-matches("^(uid|machine_failure|TWF|HDF|PWF|OSF|RNF)")) %>%
  names()
numeric_cols <- head(numeric_cols, 10)   # cap at 10 columns

if (length(numeric_cols) >= 3) {
  cor_mat <- cor(df[, numeric_cols], use = "pairwise.complete.obs")

  png(file.path(output_dir, "correlation_matrix.png"),
      width = 900, height = 800, res = 150)
  corrplot::corrplot(
    cor_mat,
    method     = "color",
    type       = "upper",
    addCoef.col = "black",
    number.cex  = 0.7,
    tl.col      = "#1a2744",
    tl.srt      = 45,
    col         = colorRampPalette(c("#e74c3c", "white", "#2980b9"))(200),
    title       = "Feature Correlation Matrix",
    mar         = c(0, 0, 1.5, 0)
  )
  dev.off()
  cat("[plot] Correlation matrix saved\n")

  # Find top correlated pairs
  cor_pairs <- as.data.frame(as.table(cor_mat)) %>%
    filter(Var1 != Var2, as.character(Var1) < as.character(Var2)) %>%
    mutate(abs_cor = abs(Freq)) %>%
    arrange(desc(abs_cor)) %>%
    head(10)
  names(cor_pairs) <- c("feature_1", "feature_2", "correlation", "abs_correlation")
  cat("\nTop 5 correlated feature pairs:\n")
  print(head(as.data.frame(cor_pairs[, 1:3]), 5))
}


# ─────────────────────────────────────────────────────────────────────────────
# 7. Health Score Trend Over Observations
# ─────────────────────────────────────────────────────────────────────────────

if ("health_score" %in% names(df)) {
  cat("\n── Health Score Trend ───────────────────────────────────────\n")

  trend_df <- df %>%
    mutate(obs = row_number()) %>%
    select(obs, health_score, health_status) %>%
    filter(!is.na(health_score))

  # Smoothed trend
  p_trend <- ggplot(trend_df, aes(x = obs, y = health_score)) +
    geom_point(aes(color = health_status), alpha = 0.3, size = 0.8) +
    geom_smooth(method = "loess", span = 0.1, color = "#2c3e50", linewidth = 1.2, se = FALSE) +
    geom_hline(yintercept = 80, linetype = "dashed", color = "#27ae60",  linewidth = 0.7) +
    geom_hline(yintercept = 60, linetype = "dashed", color = "#f39c12",  linewidth = 0.7) +
    geom_hline(yintercept = 40, linetype = "dashed", color = "#e74c3c",  linewidth = 0.7) +
    scale_color_manual(values = c(
      "Good" = "#27ae60", "Warning" = "#f39c12",
      "Degraded" = "#e67e22", "Critical" = "#e74c3c", "Unknown" = "#95a5a6"
    )) +
    labs(
      title    = "Asset Health Score Trend",
      subtitle = "Loess smoothed — dashed lines: Good / Warning / Degraded thresholds",
      x        = "Observation Index",
      y        = "Health Score (0–100)",
      color    = "Status"
    ) +
    theme_bw(base_size = 12) +
    theme(plot.title = element_text(face = "bold"),
          legend.position = "bottom")

  ggsave(file.path(output_dir, "health_score_trend.png"),
         p_trend, width = 10, height = 5, dpi = 150)
  cat("[plot] Health score trend saved\n")

  # Summary stats by status
  hs_summary <- trend_df %>%
    group_by(health_status) %>%
    summarise(
      count  = n(),
      mean_hs = round(mean(health_score), 1),
      min_hs  = round(min(health_score), 1),
      max_hs  = round(max(health_score), 1),
      .groups = "drop"
    )
  cat("Health score summary by status:\n")
  print(as.data.frame(hs_summary))
  write.csv(hs_summary, file.path(output_dir, "health_summary_by_status.csv"), row.names = FALSE)
}


# ─────────────────────────────────────────────────────────────────────────────
# 8. Box plots — Feature distributions per fault type
# ─────────────────────────────────────────────────────────────────────────────

if ("fault_type" %in% names(df) && length(sensor_cols) > 0) {
  cat("\n── Box Plots by Fault Type ──────────────────────────────────\n")

  box_df <- df %>%
    select(fault_type, all_of(sensor_cols)) %>%
    pivot_longer(-fault_type, names_to = "feature", values_to = "value") %>%
    filter(!is.na(value))

  p_box <- ggplot(box_df, aes(x = fault_type, y = value, fill = fault_type)) +
    geom_boxplot(outlier.size = 0.4, outlier.alpha = 0.4) +
    facet_wrap(~feature, scales = "free_y", ncol = 3) +
    scale_fill_brewer(palette = "Set2", guide = "none") +
    labs(
      title = "Feature Distributions by Fault Type",
      x     = "Fault Type",
      y     = "Value (normalised)"
    ) +
    theme_bw(base_size = 11) +
    theme(
      plot.title  = element_text(face = "bold"),
      axis.text.x = element_text(angle = 30, hjust = 1, size = 8)
    )

  ggsave(file.path(output_dir, "feature_boxplots.png"),
         p_box, width = 12, height = 7, dpi = 150)
  cat("[plot] Feature boxplots saved\n")
}


# ─────────────────────────────────────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────────────────────────────────────
cat(sprintf("\n[OK]  All R outputs saved to: %s\n", output_dir))
cat("       Files: descriptive_stats.csv, hypothesis_tests.csv,\n")
cat("              health_summary_by_status.csv, *.png plots\n\n")
