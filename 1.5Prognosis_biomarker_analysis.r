# %%
rm(list = ls())
path_dir <- '/home/zhangwenxiang/workspace/python/UKB/OS/v13_prognosis'
setwd(path_dir)

# %% [markdown]
# # import R packages

# %%
library("clusterProfiler")
library("org.Hs.eg.db")
library(biomaRt)
library(ReactomePA)
library(msigdbr)

# %% [markdown]
# # read biomarker information

# %%
importance_overall <- read.csv(paste0(path_dir, '/xgb_binary_nestedcv_results_AP/shap/overall/oof_mean_abs_shap.csv'))

# %%
Top_k <- 50

importance_overall <- importance_overall$feature[1:Top_k]

write.csv(importance_overall, file = paste0(path_dir , '/biomarker_analysis/importance_overall.csv'))

# %%
biomarker_proteins <- toupper(importance_overall)

# %% [markdown]
# # gene name --> ensemble id

# %%
conv <- bitr(biomarker_proteins,
             fromType = "SYMBOL",
             toType   = c("ENTREZID","ENSEMBL"),
             OrgDb    = org.Hs.eg.db)

if (nrow(conv) == 0) stop("没有基因被成功映射，请检查符号是否为 HGNC。")
entrez <- unique(conv$ENTREZID)
cat(sprintf("成功映射到 ENTREZ ID 数：%d\n", length(entrez)))

# %% [markdown]
# # ###############################################################################
# ## BP
BP_result <- enrichGO(
  gene = entrez,                
  OrgDb = org.Hs.eg.db,             
  keyType = "ENTREZID",             
  ont = "BP",                       
  pAdjustMethod = "BH",             
  readable = TRUE,                   
  pvalueCutoff = 0.05
)

BP_result@result$ID[BP_result@result$p.adjust < 0.05]
barplot(BP_result, showCategory = 10, title = "GO Enrichment Analysis")
dotplot(BP_result, showCategory = 10, title = "GO Enrichment Analysis")

write.csv(BP_result@result, paste0(getwd(), '/biomarker_analysis/BP_results_importance_overall.csv'))

# %% [markdown]
# ## MF
MF_result <- enrichGO(
  gene = entrez,                
  OrgDb = org.Hs.eg.db,             
  keyType = "ENTREZID",             
  ont = "MF",                       
  pAdjustMethod = "BH",             
  readable = TRUE,                   
  pvalueCutoff = 0.05
)
MF_result@result$ID[MF_result@result$p.adjust < 0.05]
barplot(MF_result, showCategory = 10, title = "GO Enrichment Analysis")
dotplot(MF_result, showCategory = 10, title = "GO Enrichment Analysis")

write.csv(MF_result@result, paste0(getwd(), '/biomarker_analysis/MF_result_importance_overall.csv'))
