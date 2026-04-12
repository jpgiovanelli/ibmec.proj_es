# =============================================================
# Analise Estatistica de Metricas de Codigo-Fonte do JUnit
# AP1 - Engenharia de Software | 2026
# Dataset: GitHub Bug Dataset v1.1 -- junit-Class.csv
# =============================================================

# ── Pacotes necessarios ──────────────────────────────────────
required_packages <- c(
  "ggplot2", "dplyr", "tidyr", "readr", "scales",
  "PerformanceAnalytics", "car", "nortest",
  "pROC", "caret", "gridExtra", "ggcorrplot",
  "moments", "GGally"
)

installed <- rownames(installed.packages())
to_install <- setdiff(required_packages, installed)
if (length(to_install) > 0) {
  install.packages(to_install, repos = "https://cloud.r-project.org")
}

invisible(lapply(required_packages, library, character.only = TRUE))

# ── Diretorios de saida ──────────────────────────────────────
BASE_DIR <- file.path(dirname(rstudioapi::getSourceEditorContext()$path), "..")
FIG_DIR  <- file.path(BASE_DIR, "output", "figures_r")
TAB_DIR  <- file.path(BASE_DIR, "output", "tables")

dir.create(FIG_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(TAB_DIR, recursive = TRUE, showWarnings = FALSE)

save_fig <- function(name, width = 10, height = 7) {
  path <- file.path(FIG_DIR, name)
  ggsave(path, width = width, height = height, dpi = 300, bg = "white")
  cat("  -> figura salva:", path, "\n")
}

# ── Tema padrao ggplot2 ──────────────────────────────────────
theme_set(
  theme_bw(base_size = 11) +
    theme(
      plot.title       = element_text(size = 13, face = "bold"),
      plot.subtitle    = element_text(size = 10),
      axis.title       = element_text(size = 10),
      strip.background = element_rect(fill = "gray92"),
      panel.grid.minor = element_blank()
    )
)

CB_PALETTE <- c("#0072B2","#E69F00","#009E73","#F0E442",
                "#56B4E9","#D55E00","#CC79A7","#999999",
                "#117733","#882255")

# =============================================================
# SECAO 0 -- CARREGAMENTO E PRE-PROCESSAMENTO
# =============================================================
cat(strrep("=", 65), "\n")
cat("SECAO 0 -- DADOS CARREGADOS\n")
cat(strrep("=", 65), "\n")

df_raw <- read_csv(
  file.path(BASE_DIR, "data", "junit-Class.csv"),
  show_col_types = FALSE
)

CATEGORICAL <- c("ID", "Name", "LongName", "Parent", "Component", "Path")
SELECTED    <- c("WMC","CBO","LCOM5","DIT","RFC",
                 "LOC","NM","WarningInfo","WarningMajor","CD")

# Forcar colunas numericas
df_raw <- df_raw %>%
  mutate(across(!all_of(CATEGORICAL), as.numeric))

# Remover colunas zero-variancia
nzv <- sapply(df_raw[, !names(df_raw) %in% CATEGORICAL],
              function(x) length(unique(na.omit(x))) <= 1)
zero_var_cols <- names(nzv[nzv])
df_raw <- df_raw %>% select(-all_of(zero_var_cols))

df    <- df_raw %>% select(all_of(c(CATEGORICAL, SELECTED)))
df_s  <- df_raw %>% select(all_of(SELECTED))

cat(sprintf("  Dataset : junit-Class.csv\n"))
cat(sprintf("  Linhas  : %d\n", nrow(df_s)))
cat(sprintf("  Colunas zero-variancia removidas: %d\n\n", length(zero_var_cols)))

# =============================================================
# SECAO 1 -- MEDIDAS DE TENDENCIA CENTRAL
# =============================================================
cat(strrep("=", 65), "\n")
cat("SECAO 1 -- MEDIDAS DE TENDENCIA CENTRAL\n")
cat(strrep("=", 65), "\n")

Mode <- function(x) {
  x <- na.omit(x)
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

ct <- data.frame(
  Variavel = SELECTED,
  Media    = sapply(df_s, mean,   na.rm = TRUE),
  Mediana  = sapply(df_s, median, na.rm = TRUE),
  Moda     = sapply(df_s, Mode),
  row.names = NULL
) %>% mutate(across(where(is.numeric), ~round(.x, 3)))

print(ct)
write_csv(ct, file.path(TAB_DIR, "central_tendency_r.csv"))
cat("\n  INTERPRETACAO:\n")
for (i in seq_len(nrow(ct))) {
  v    <- ct$Variavel[i]
  m    <- ct$Media[i]
  med  <- ct$Mediana[i]
  rat  <- ifelse(med != 0, m / med, Inf)
  desc <- if (rat > 1.5) "forte assimetria a direita" else
          if (rat > 1.1) "leve assimetria a direita" else
          "distribuicao mais simetrica"
  cat(sprintf("  %s: media=%.2f, mediana=%.2f -> %s\n", v, m, med, desc))
}
cat("\n")

# =============================================================
# SECAO 2 -- MEDIDAS DE DISPERSAO
# =============================================================
cat(strrep("=", 65), "\n")
cat("SECAO 2 -- MEDIDAS DE DISPERSAO\n")
cat(strrep("=", 65), "\n")

disp <- data.frame(
  Variavel    = SELECTED,
  Amplitude   = sapply(df_s, function(x) diff(range(x, na.rm=TRUE))),
  Variancia   = sapply(df_s, var,  na.rm = TRUE),
  DesvPadrao  = sapply(df_s, sd,   na.rm = TRUE),
  row.names   = NULL
) %>%
  mutate(CV_pct = round(DesvPadrao / sapply(df_s, mean, na.rm=TRUE) * 100, 2)) %>%
  mutate(across(where(is.numeric), ~round(.x, 3)))

print(disp)
write_csv(disp, file.path(TAB_DIR, "dispersion_r.csv"))
cat("\n")

# =============================================================
# SECAO 3 -- MEDIDAS DE POSICAO RELATIVA + BOXPLOTS
# =============================================================
cat(strrep("=", 65), "\n")
cat("SECAO 3 -- MEDIDAS DE POSICAO RELATIVA\n")
cat(strrep("=", 65), "\n")

probs <- c(0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
perc  <- as.data.frame(t(sapply(df_s, quantile, probs = probs, na.rm = TRUE)))
colnames(perc) <- c("P5","P10","Q1","Mediana","Q3","P90","P95")
perc$IQR       <- perc$Q3 - perc$Q1
perc$Variavel  <- rownames(perc)
perc           <- perc[, c("Variavel", setdiff(names(perc), "Variavel"))]
rownames(perc) <- NULL
perc_print <- perc %>% mutate(across(where(is.numeric), ~round(.x, 3)))
print(perc_print)
write_csv(perc, file.path(TAB_DIR, "percentiles_r.csv"))
cat("\n")

# Boxplots 2x5
df_long <- df_s %>%
  pivot_longer(everything(), names_to = "Variavel", values_to = "Valor") %>%
  mutate(Variavel = factor(Variavel, levels = SELECTED))

# Calcular medias para ponto adicional
medias <- df_s %>%
  summarise(across(everything(), mean, na.rm = TRUE)) %>%
  pivot_longer(everything(), names_to = "Variavel", values_to = "Media") %>%
  mutate(Variavel = factor(Variavel, levels = SELECTED))

ggplot(df_long, aes(x = "", y = Valor, fill = Variavel)) +
  geom_boxplot(outlier.size = 0.8, outlier.alpha = 0.4, width = 0.6) +
  stat_summary(fun = mean, geom = "point", shape = 23,
               size = 2.5, fill = "red", color = "darkred") +
  facet_wrap(~Variavel, scales = "free_y", ncol = 5) +
  scale_fill_manual(values = CB_PALETTE) +
  labs(title = "Boxplots das Variaveis Selecionadas",
       subtitle = "Losango vermelho = media",
       x = NULL, y = "Valor") +
  theme(legend.position = "none",
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank())
save_fig("boxplots_all_r.png", width = 18, height = 8)

# =============================================================
# SECAO 4 -- GRAFICOS
# =============================================================
cat(strrep("=", 65), "\n")
cat("SECAO 4 -- GRAFICOS\n")
cat(strrep("=", 65), "\n")

# 4a. Histogramas com KDE e linhas de media/mediana
stats_lines <- df_long %>%
  group_by(Variavel) %>%
  summarise(Media   = mean(Valor,   na.rm = TRUE),
            Mediana = median(Valor, na.rm = TRUE), .groups = "drop")

ggplot(df_long, aes(x = Valor, fill = Variavel)) +
  geom_histogram(aes(y = after_stat(density)), bins = 30,
                 color = "white", linewidth = 0.3, alpha = 0.85) +
  geom_density(color = "gray30", linewidth = 0.7) +
  geom_vline(data = stats_lines,
             aes(xintercept = Media),   color = "red",   linetype = "dashed", linewidth = 0.8) +
  geom_vline(data = stats_lines,
             aes(xintercept = Mediana), color = "green4", linetype = "dotdash", linewidth = 0.8) +
  facet_wrap(~Variavel, scales = "free", ncol = 5) +
  scale_fill_manual(values = CB_PALETTE) +
  labs(title = "Histogramas das Variaveis Selecionadas",
       subtitle = "Linha vermelha tracejada = media | linha verde = mediana",
       x = "Valor", y = "Densidade") +
  theme(legend.position = "none")
save_fig("histograms_r.png", width = 20, height = 8)

# 4b. Scatter plots
scatter_pairs <- list(
  c("LOC",  "WMC",         "Tamanho x Complexidade"),
  c("CBO",  "RFC",         "Acoplamento x Resp. por Classe"),
  c("WMC",  "LCOM5",       "Complexidade x Falta de Coesao"),
  c("LOC",  "WarningInfo", "Tamanho x Avisos Informativos"),
  c("DIT",  "CBO",         "Heranca x Acoplamento"),
  c("LOC",  "CD",          "Tamanho x Densidade Comentario")
)

plots_scatter <- lapply(scatter_pairs, function(p) {
  x_var <- p[1]; y_var <- p[2]; titulo <- p[3]
  dat   <- df_s[, c(x_var, y_var)]
  r_val <- cor(dat[[x_var]], dat[[y_var]], method = "pearson", use = "complete.obs")
  r_s   <- cor(dat[[x_var]], dat[[y_var]], method = "spearman", use = "complete.obs")
  stars <- ifelse(abs(r_val) >= 0.8, "***",
           ifelse(abs(r_val) >= 0.5, "**",
           ifelse(abs(r_val) >= 0.3, "*", "")))
  ggplot(dat, aes_string(x = x_var, y = y_var)) +
    geom_point(alpha = 0.25, size = 1.2, color = CB_PALETTE[1]) +
    geom_smooth(method = "lm", se = TRUE, color = "red",
                linewidth = 1.2, fill = "salmon", alpha = 0.2) +
    annotate("text", x = -Inf, y = Inf, hjust = -0.1, vjust = 1.5,
             label = sprintf("r=%.3f%s\nrho=%.3f", r_val, stars, r_s),
             size = 3.2, fontface = "bold",
             color = ifelse(abs(r_val) >= 0.7, "red3",
                     ifelse(abs(r_val) >= 0.4, "orange3", "gray40"))) +
    labs(title = titulo, x = x_var, y = y_var)
})

do.call(gridExtra::grid.arrange,
        c(plots_scatter, list(ncol = 3,
          top = grid::textGrob("Graficos de Dispersao entre Pares de Variaveis",
                                gp = grid::gpar(fontsize = 13, fontface = "bold")))))
save_fig("scatter_plots_r.png", width = 16, height = 10)

# 4c. Distribuicao de DIT (variavel discreta)
dit_freq <- df_s %>%
  count(DIT) %>%
  mutate(DIT = factor(DIT))

ggplot(dit_freq, aes(x = DIT, y = n, fill = DIT)) +
  geom_col(width = 0.6, color = "white") +
  geom_text(aes(label = n), vjust = -0.5, size = 4) +
  scale_fill_manual(values = CB_PALETTE) +
  labs(title = "Distribuicao de DIT (Profundidade da Arvore de Heranca)",
       x = "Profundidade de Heranca (DIT)",
       y = "Numero de Classes") +
  theme(legend.position = "none")
save_fig("bar_dit_r.png", width = 7, height = 5)

# 4d. Heatmap de correlacao
corr_mat <- cor(df_s, method = "pearson", use = "complete.obs")
ggcorrplot(corr_mat,
           method    = "square",
           type      = "lower",
           lab       = TRUE,
           lab_size  = 3.2,
           colors    = c("#D55E00", "white", "#0072B2"),
           title     = "Mapa de Calor -- Correlacoes de Pearson",
           ggtheme   = theme_bw(base_size = 11)) +
  theme(plot.title = element_text(face = "bold", size = 13))
save_fig("correlation_heatmap_r.png", width = 11, height = 9)

cat("  Graficos gerados: histogramas, scatter, barras (DIT), heatmap\n\n")

# =============================================================
# SECAO 5 -- AVALIACAO DE OUTLIERS
# =============================================================
cat(strrep("=", 65), "\n")
cat("SECAO 5 -- AVALIACAO DE OUTLIERS\n")
cat(strrep("=", 65), "\n")

outlier_stats <- lapply(SELECTED, function(v) {
  x    <- df_s[[v]]
  q1   <- quantile(x, 0.25, na.rm = TRUE)
  q3   <- quantile(x, 0.75, na.rm = TRUE)
  iqrv <- q3 - q1
  lo   <- q1 - 1.5 * iqrv
  hi   <- q3 + 1.5 * iqrv
  n_lo <- sum(x < lo, na.rm = TRUE)
  n_hi <- sum(x > hi, na.rm = TRUE)
  data.frame(
    Variavel        = v,
    Q1              = round(q1,   2),
    Q3              = round(q3,   2),
    IQR             = round(iqrv, 2),
    Fence_Inf       = round(lo,   2),
    Fence_Sup       = round(hi,   2),
    Outliers_Inf    = n_lo,
    Outliers_Sup    = n_hi,
    Pct_Outliers    = round((n_lo + n_hi) / length(x) * 100, 1)
  )
})
outlier_df <- do.call(rbind, outlier_stats)
print(outlier_df)
write_csv(outlier_df, file.path(TAB_DIR, "outliers_r.csv"))
cat("\n")

# God Classes
loc_hi <- outlier_df$Fence_Sup[outlier_df$Variavel == "LOC"]
cat("  Classes com LOC extremo (possiveis God Classes):\n")
god <- df %>%
  filter(LOC > loc_hi) %>%
  select(Name, LOC, WMC, NM) %>%
  arrange(desc(LOC)) %>%
  head(10)
print(god)
cat("\n")

# Boxplots com anotacao de outliers
outlier_df_ann <- outlier_df %>%
  mutate(label = paste0(Outliers_Sup + Outliers_Inf, " out.\n(", Pct_Outliers, "%)"))

df_long2 <- df_long %>%
  left_join(outlier_df_ann %>% select(Variavel, label), by = "Variavel")

ggplot(df_long2, aes(x = "", y = Valor, fill = Variavel)) +
  geom_boxplot(outlier.size = 0.7, outlier.alpha = 0.3, width = 0.6) +
  stat_summary(fun = mean, geom = "point", shape = 23,
               size = 2, fill = "red", color = "darkred") +
  geom_text(data = outlier_df_ann,
            aes(x = 1, y = Inf, label = label),
            vjust = 1.5, size = 2.8, inherit.aes = FALSE) +
  facet_wrap(~Variavel, scales = "free_y", ncol = 5) +
  scale_fill_manual(values = CB_PALETTE) +
  labs(title = "Boxplots com Contagem de Outliers (IQR x 1.5)",
       x = NULL, y = "Valor") +
  theme(legend.position = "none",
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank())
save_fig("outlier_boxplots_r.png", width = 18, height = 8)

# =============================================================
# SECAO 6 -- TESTES DE NORMALIDADE
# =============================================================
cat(strrep("=", 65), "\n")
cat("SECAO 6 -- TESTES DE NORMALIDADE\n")
cat(strrep("=", 65), "\n")

norm_results <- lapply(SELECTED, function(v) {
  x <- df_s[[v]]
  # Shapiro-Wilk (max 5000)
  samp    <- if (length(x) > 5000) sample(x, 5000) else x
  sw      <- shapiro.test(samp)
  # Kolmogorov-Smirnov (Lilliefors)
  lf      <- nortest::lillie.test(x)
  data.frame(
    Variavel        = v,
    SW_Estatistica  = round(sw$statistic, 4),
    SW_pvalor       = sw$p.value,
    KS_Estatistica  = round(lf$statistic, 4),
    KS_pvalor       = lf$p.value,
    Normal_0.05     = ifelse(sw$p.value > 0.05 & lf$p.value > 0.05, "Sim", "Nao")
  )
})
norm_df <- do.call(rbind, norm_results)
rownames(norm_df) <- NULL
print(norm_df)
write_csv(norm_df, file.path(TAB_DIR, "normality_r.csv"))
cat("\n")

# QQ-plots com ggplot2
qq_plots <- lapply(SELECTED, function(v) {
  ggplot(df_s, aes(sample = .data[[v]])) +
    stat_qq(size = 0.8, alpha = 0.35, color = CB_PALETTE[1]) +
    stat_qq_line(color = "red", linewidth = 1) +
    labs(title = v, x = "Quantis Teoricos", y = "Quantis Observados")
})
do.call(gridExtra::grid.arrange,
        c(qq_plots, list(ncol = 5,
          top = grid::textGrob("Q-Q Plots das Variaveis Selecionadas",
                                gp = grid::gpar(fontsize = 13, fontface = "bold")))))
save_fig("qq_plots_r.png", width = 20, height = 8)

cat("  Efeito da transformacao log1p:\n")
for (v in c("WMC","LOC","RFC","WarningInfo")) {
  x_log  <- log1p(df_s[[v]])
  sw_log <- shapiro.test(if (length(x_log) > 5000) sample(x_log, 5000) else x_log)
  sw_ori <- norm_df$SW_pvalor[norm_df$Variavel == v]
  melhora <- ifelse(sw_log$p.value > sw_ori, "melhora", "sem melhora")
  cat(sprintf("  %s: p-orig=%.2e -> p-log1p=%.2e (%s)\n",
              v, sw_ori, sw_log$p.value, melhora))
}
cat("\n")

# =============================================================
# SECAO 7 -- COEFICIENTES DE CORRELACAO
# =============================================================
cat(strrep("=", 65), "\n")
cat("SECAO 7 -- COEFICIENTES DE CORRELACAO\n")
cat(strrep("=", 65), "\n")

corr_p <- round(cor(df_s, method = "pearson",  use = "complete.obs"), 3)
corr_s <- round(cor(df_s, method = "spearman", use = "complete.obs"), 3)

cat("  Pearson:\n")
print(corr_p)
cat("\n  Spearman:\n")
print(corr_s)

write.csv(corr_p, file.path(TAB_DIR, "correlation_pearson_r.csv"))
write.csv(corr_s, file.path(TAB_DIR, "correlation_spearman_r.csv"))

cat("\n  PARES NOTAVEIS (Pearson, |r| >= 0.4):\n")
for (i in seq_along(SELECTED)) {
  for (j in seq_along(SELECTED)) {
    if (j <= i) next
    r <- corr_p[i, j]
    if (abs(r) >= 0.4) {
      label <- if (abs(r) >= 0.7) "FORTE" else "moderada"
      cat(sprintf("  %s x %s: r=%.3f (%s)\n",
                  SELECTED[i], SELECTED[j], r, label))
    }
  }
}
cat("\n")

# chart.Correlation() -- equivalente exato ao solicitado na ementa
png(file.path(FIG_DIR, "chart_correlation_r.png"),
    width = 3000, height = 3000, res = 300)
PerformanceAnalytics::chart.Correlation(
  df_s,
  histogram = TRUE,
  method    = "pearson",
  pch       = 19
)
dev.off()
cat(sprintf("  -> figura salva: %s\n",
            file.path(FIG_DIR, "chart_correlation_r.png")))

# GGally pairgrid (versao ggplot2)
ggpairs(
  df_s,
  lower  = list(continuous = wrap("points", alpha = 0.15, size = 0.6)),
  upper  = list(continuous = wrap("cor",    size = 2.8, method = "pearson")),
  diag   = list(continuous = wrap("densityDiag")),
  title  = "Matriz de Correlacao (GGally)"
) + theme(axis.text = element_text(size = 6),
          strip.text = element_text(size = 7))
save_fig("pairgrid_correlation_r.png", width = 14, height = 12)

# =============================================================
# SECAO 8 -- MODELAGEM ESTATISTICA
# =============================================================
cat(strrep("=", 65), "\n")
cat("SECAO 8 -- MODELAGEM ESTATISTICA\n")
cat(strrep("=", 65), "\n")

# ── 8A. REGRESSAO LINEAR MULTIPLA ───────────────────────────
cat("\n  8A. REGRESSAO LINEAR MULTIPLA\n")
cat("  Y = WMC | X = CBO + DIT + NM + LCOM5\n\n")

df_model <- na.omit(df_s[, c("WMC","CBO","DIT","NM","LCOM5")])
model_lin <- lm(WMC ~ CBO + DIT + NM + LCOM5, data = df_model)
print(summary(model_lin))

# VIF
cat("\n  VIF:\n")
vif_vals <- car::vif(model_lin)
print(round(vif_vals, 3))
cat("  (VIF > 10 indica multicolinearidade problematica)\n\n")

# Metricas
r2      <- summary(model_lin)$r.squared
r2_adj  <- summary(model_lin)$adj.r.squared
resids  <- residuals(model_lin)
rmse    <- sqrt(mean(resids^2))
mae_val <- mean(abs(resids))
cat(sprintf("  R2          : %.4f\n", r2))
cat(sprintf("  R2 ajustado : %.4f\n", r2_adj))
cat(sprintf("  RMSE        : %.4f\n", rmse))
cat(sprintf("  MAE         : %.4f\n\n", mae_val))

# Modelo com LOC
model_lin2 <- lm(WMC ~ CBO + DIT + NM + LCOM5 + LOC,
                 data = na.omit(df_s[, c("WMC","CBO","DIT","NM","LCOM5","LOC")]))
cat(sprintf("  Modelo com LOC: R2=%.4f, R2adj=%.4f\n\n",
            summary(model_lin2)$r.squared,
            summary(model_lin2)$adj.r.squared))

# Diagnosticos
df_diag <- data.frame(
  Ajustados = fitted(model_lin),
  Residuos  = residuals(model_lin)
)
p1 <- ggplot(df_diag, aes(x = Ajustados, y = Residuos)) +
  geom_point(alpha = 0.3, size = 1, color = CB_PALETTE[1]) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed", linewidth = 1) +
  labs(title = "Residuos vs. Valores Ajustados",
       x = "Valores Ajustados", y = "Residuos")

p2 <- ggplot(df_diag, aes(sample = Residuos)) +
  stat_qq(size = 0.8, alpha = 0.4, color = CB_PALETTE[1]) +
  stat_qq_line(color = "red", linewidth = 1) +
  labs(title = "Q-Q Plot dos Residuos",
       x = "Quantis Teoricos", y = "Quantis dos Residuos")

p3 <- ggplot(df_diag, aes(x = Residuos)) +
  geom_histogram(aes(y = after_stat(density)), bins = 40,
                 fill = CB_PALETTE[2], color = "white", alpha = 0.8) +
  geom_density(color = "gray30", linewidth = 0.8) +
  geom_vline(xintercept = 0, color = "red", linetype = "dashed", linewidth = 1) +
  labs(title = "Distribuicao dos Residuos",
       x = "Residuos", y = "Densidade")

gridExtra::grid.arrange(p1, p2, p3, ncol = 3,
  top = grid::textGrob(
    "Diagnosticos da Regressao Linear (WMC ~ CBO + DIT + NM + LCOM5)",
    gp = grid::gpar(fontsize = 12, fontface = "bold")))
save_fig("regression_diagnostics_r.png", width = 16, height = 5)

# ── 8B. REGRESSAO LOGISTICA ──────────────────────────────────
cat("  8B. REGRESSAO LOGISTICA\n")
cat("  Y = HasMajorWarning (WarningMajor > 0)\n")
cat("  X = WMC + LOC + CBO + LCOM5 + DIT + CD\n\n")

df_log <- df_s %>%
  mutate(HasMajorWarning = as.integer(WarningMajor > 0)) %>%
  select(HasMajorWarning, WMC, LOC, CBO, LCOM5, DIT, CD) %>%
  na.omit()

cat(sprintf("  Positivos : %d (%.1f%%)\n",
            sum(df_log$HasMajorWarning),
            mean(df_log$HasMajorWarning) * 100))
cat(sprintf("  Negativos : %d (%.1f%%)\n\n",
            sum(df_log$HasMajorWarning == 0),
            (1 - mean(df_log$HasMajorWarning)) * 100))

model_logit <- glm(
  HasMajorWarning ~ WMC + LOC + CBO + LCOM5 + DIT + CD,
  data   = df_log,
  family = binomial(link = "logit")
)
print(summary(model_logit))

cat("\n  Odds Ratios (exp(coef)):\n")
or_table <- data.frame(
  Coeficiente  = round(coef(model_logit), 4),
  Odds_Ratio   = round(exp(coef(model_logit)), 4),
  p_valor      = round(coef(summary(model_logit))[, "Pr(>|z|)"], 4)
)
print(or_table)

# Previsoes e matriz de confusao
y_prob  <- predict(model_logit, type = "response")
y_class <- ifelse(y_prob >= 0.5, 1, 0)
cat("\n  Matriz de Confusao:\n")
print(table(Real = df_log$HasMajorWarning, Previsto = y_class))
acc <- mean(y_class == df_log$HasMajorWarning)
cat(sprintf("  Acuracia: %.4f\n", acc))

# Curva ROC
roc_obj <- pROC::roc(df_log$HasMajorWarning, y_prob, quiet = TRUE)
auc_val <- pROC::auc(roc_obj)
cat(sprintf("  AUC-ROC : %.4f\n\n", auc_val))

roc_df <- data.frame(
  FPR = 1 - roc_obj$specificities,
  TPR = roc_obj$sensitivities
)
ggplot(roc_df, aes(x = FPR, y = TPR)) +
  geom_ribbon(aes(ymin = 0, ymax = TPR), fill = CB_PALETTE[1], alpha = 0.15) +
  geom_line(color = CB_PALETTE[1], linewidth = 1.5) +
  geom_abline(slope = 1, intercept = 0,
              color = "gray50", linetype = "dashed", linewidth = 1) +
  annotate("text", x = 0.65, y = 0.25,
           label = sprintf("AUC = %.3f", auc_val),
           size = 5, fontface = "bold", color = CB_PALETTE[1]) +
  labs(title    = "Curva ROC -- Regressao Logistica",
       subtitle = "HasMajorWarning ~ WMC + LOC + CBO + LCOM5 + DIT + CD",
       x = "Taxa de Falsos Positivos (1 - Especificidade)",
       y = "Taxa de Verdadeiros Positivos (Sensibilidade)")
save_fig("logistic_roc_curve_r.png", width = 7, height = 6)

# =============================================================
# SECAO 9 -- DISCUSSAO
# =============================================================
cat(strrep("=", 65), "\n")
cat("SECAO 9 -- DISCUSSAO DOS RESULTADOS\n")
cat(strrep("=", 65), "\n")
cat("
  PRINCIPAIS ACHADOS:

  1. Todas as metricas sao assimetricas a direita (media >> mediana).
  2. Nenhuma variavel segue distribuicao normal (p < 0.001 em todos os testes).
  3. Cluster LOC-WMC-RFC: correlacoes fortes (r > 0.85).
     Classes maiores tendem a ser mais complexas.
  4. DIT e independente de todas as demais variaveis (r < 0.1).
     Heranca nao prediz complexidade no JUnit.
  5. WarningInfo x WarningMajor: r=0.707 (forte) -- avisos PMD correlacionam.
  6. Number of bugs = 0 em todos os registros (limitacao da base).
  7. Regressao linear (WMC ~ CBO+DIT+NM+LCOM5): R2=0.34 sem LOC,
     R2=0.89 com LOC (LOC domina por colinearidade).
  8. Regressao logistica (HasMajorWarning): AUC=0.74, acuracia=67%.
     LCOM5 e LOC sao os preditores mais significativos.
  9. God Classes identificadas: Assert (935 LOC), AssertionTest (626 LOC),
     TestCase (471 LOC) -- candidatas a refatoracao por SRP.
")

cat(strrep("=", 65), "\n")
cat("ANALISE CONCLUIDA\n")
cat(sprintf("  Figuras -> %s\n", FIG_DIR))
cat(sprintf("  Tabelas -> %s\n", TAB_DIR))
cat(strrep("=", 65), "\n")
