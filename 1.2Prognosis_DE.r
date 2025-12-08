rm(list = ls())

setwd('/home/zhangwenxiang/workspace/python/UKB/OS/v13_prognosis')

df_balanced <- read.csv('./data/df_balanced.csv')

protein_exp <- df_balanced[,10:dim(df_balanced)[2]]
sample_label <- df_balanced$label_0_1

################################################################################
# DE
################################################################################
## ===== packages =====
# pkgs <- c("limma","ggplot2","ggrepel")
# for (p in pkgs) if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
library(limma)
library(ggplot2)
library(ggrepel)

stopifnot(nrow(protein_exp) == length(sample_label))

expr <- t(as.matrix(protein_exp))

if (is.null(rownames(expr))) rownames(expr) <- colnames(protein_exp)

group <- factor(sample_label, levels = c(0,1), labels = c("C0","C1"))
design <- model.matrix(~ 0 + group)
colnames(design) <- levels(group)

contr_pairwise <- limma::makeContrasts(
  C0_vs_C1 = C0 - C1,
  levels = design
)

base_dir <- "limma_DE"
dir.create(base_dir, showWarnings = FALSE)
# dir.create(file.path(base_dir, "one_vs_rest"), showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(base_dir, "pairwise"), showWarnings = FALSE, recursive = TRUE)

volcano_plot <- function(tt, title_str, out_png, out_pdf = NULL,
                         padj_cut = 0.05, lfc_cut = 1,
                         col_up = "#D62728",    
                         col_down = "#1F77B4", 
                         col_ns = "grey70") {   
  df <- tt
  df$gene <- rownames(df)

  pplot <- df$adj.P.Val
  pplot[is.na(pplot)] <- 1
  pplot <- pmax(pplot, .Machine$double.xmin)
  df$negLog10FDR <- -log10(pplot)

  # Up / Down / NS
  df$sig <- "NS"
  df$sig[df$adj.P.Val <= padj_cut & df$logFC >=  lfc_cut] <- "Up"
  df$sig[df$adj.P.Val <= padj_cut & df$logFC <= -lfc_cut] <- "Down"

  lab_df <- subset(df, sig %in% c("Up","Down"))

  g <- ggplot(df, aes(x = logFC, y = negLog10FDR)) +
    geom_point(aes(color = sig), alpha = 0.8, size = 1.4) +
    scale_color_manual(values = c(Up = col_up, Down = col_down, NS = col_ns)) +
    geom_vline(xintercept = c(-lfc_cut, lfc_cut), linetype = "dashed") +
    geom_hline(yintercept = -log10(padj_cut), linetype = "dashed") +
    ggrepel::geom_text_repel(
      data = lab_df,
      aes(label = gene, color = sig),
      size = 2.6,
      max.overlaps = Inf,         
      box.padding = 0.25,
      point.padding = 0.1,
      segment.alpha = 0.4,
      min.segment.length = 0,
      seed = 123                   
    ) +
    labs(title = title_str, x = "log2 Fold Change", y = "-log10(FDR)") +
    theme_classic(base_size = 12) +
    theme(legend.position = "top", plot.title = element_text(hjust = 0.5))

  w <- ifelse(nrow(lab_df) > 300, 10, 7)
  h <- ifelse(nrow(lab_df) > 300, 8, 5)

  ggsave(out_png, g, width = w, height = h, dpi = 300)
  if (!is.null(out_pdf)) ggsave(out_pdf, g, width = w, height = h, device = grDevices::pdf)
}

run_contrasts <- function(expr, design, contrasts, out_dir,
                          padj_cut = 0.05, lfc_cut = 1, top_n_label = 15) {
  fit0 <- limma::lmFit(expr, design)
  fit2 <- limma::contrasts.fit(fit0, contrasts)
  fit2 <- limma::eBayes(fit2, trend = TRUE, robust = TRUE)
  
  for (i in seq_len(ncol(contrasts))) {
    cat(i)
    cname <- colnames(contrasts)[i]
    tt <- limma::topTable(fit2, coef = i, number = Inf, sort.by = "P")
    write.csv(tt, file = file.path(out_dir, paste0("full_", cname, ".csv")), row.names = TRUE)
    
    sig_idx <- which(tt$adj.P.Val <= padj_cut & abs(tt$logFC) >= lfc_cut)
    if (length(sig_idx) > 0) {
      sig_df <- tt[sig_idx, , drop = FALSE]
      write.csv(sig_df, file = file.path(out_dir, paste0("significant_", cname, ".csv")),
                row.names = TRUE)
      up_genes   <- rownames(sig_df)[sig_df$logFC >=  lfc_cut]
      down_genes <- rownames(sig_df)[sig_df$logFC <= -lfc_cut]
      write.table(up_genes,   file = file.path(out_dir, paste0("up_", cname, ".txt")),
                  quote = FALSE, row.names = FALSE, col.names = FALSE)
      write.table(down_genes, file = file.path(out_dir, paste0("down_", cname, ".txt")),
                  quote = FALSE, row.names = FALSE, col.names = FALSE)
    } else {
      message(sprintf("[WARN] %s 无显著差异（按 FDR<=%.3f 且 |logFC|>=%.2f）", 
                      cname, padj_cut, lfc_cut))
    }
    
    volcano_plot(tt,
                 title_str = cname,
                 out_png   = file.path(out_dir, paste0("volcano_", cname, ".pdf")),
                 padj_cut = padj_cut, lfc_cut = lfc_cut)
  }
}

out2 <- file.path(base_dir, "pairwise")
run_contrasts(expr, design, contr_pairwise, out2,
              padj_cut = 0.05, lfc_cut = 0.5, top_n_label = 15)

message("done and dir：", normalizePath(base_dir))
