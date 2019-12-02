library(lmQCM)

setwd(dirname(rstudioapi::getSourceEditorContext()$path))
options(stringsAsFactors = F)
dataset = {} # LUAD


########################################################################
#                      Clinical
########################################################################
dataset[['clinical']] = read.table('LUAD/clinical/nationwidechildrens.org_clinical_patient_luad.txt', header = T, sep = '\t')
dataset[['clinical']] = dataset[['clinical']][3:dim(dataset[['clinical']])[1],]
colnames(dataset[['clinical']])
head(dataset[['clinical']])

dataset[['clinical']]$age_at_initial_pathologic_diagnosis = strtoi(dataset[['clinical']]$age_at_initial_pathologic_diagnosis)
dataset[['clinical']]$last_contact_days_to = strtoi(dataset[['clinical']]$last_contact_days_to)
dataset[['clinical']]$death_days_to = strtoi(dataset[['clinical']]$death_days_to)
dataset[['clinical']]$tobacco_smoking_history_indicator = strtoi(dataset[['clinical']]$tobacco_smoking_history_indicator)

print('use valid \'death_days_to\' to replace \'last_contact_days_to\'')
# days_to_last_followup and days_to_death
dataset[['clinical']]$survival_days = dataset[['clinical']]$last_contact_days_to
dataset[['clinical']]$survival_days[!is.na(dataset[['clinical']]$death_days_to)] = dataset[['clinical']]$death_days_to[!is.na(dataset[['clinical']]$death_days_to)]
# extract useful columns
dataset[['clinical']] = dataset[['clinical']][, c('bcr_patient_barcode', 'gender','age_at_initial_pathologic_diagnosis', 'vital_status','survival_days')] 
dataset[['clinical']] = dataset[['clinical']][complete.cases(dataset[['clinical']]),]
print(paste0('[clinical] ', dim(dataset[['clinical']])[1], ' complete rows found in clinical data.'))

########################################################################
#                      TMB
########################################################################

dataset[['TMB']] = read.table("LUAD/TMB/LUAD_TMB.csv", header = T, row.names = 1, sep = ",")

print(paste0('[TMB] ', dim(dataset[['TMB']])[1], ' complete rows found in clinical data.'))

########################################################################
#                      CNB
########################################################################

# Xiaohui Zhan:
# https://www.nature.com/articles/ng.3725
# a) For high quality CNVs ,the length of segmental region >20kb
#
# b) The number of probes spanning a CNV (a segmental region) to be
#    at least 10 to decrease false positives in calling CNVs.
#
# c) For a segmental region ,if the segment mean < |0.2|,this segmental
#    region should be discard.(Generally ,we using +/-0.2 as threshold
#    for a duplication/deletion. Because lots of noise will be introduced
#    from the microarray. The thresholds(+/- 0.2) were derived by examining
#    the distribution of segment mean values from tumor and normal samples)

SNP = read.table("LUAD/CNB/broad.mit.edu_PANCAN_Genome_Wide_SNP_6_whitelisted.seg", sep = "\t", header = T)
SNP$LENGTH = SNP$End - SNP$Start
SNP.filtered = SNP[(SNP$LENGTH >= 20000) &
                     (SNP$Num_Probes >= 10) &
                     (abs(SNP$Segment_Mean) >= 0.2),]



# SNP = read.table("data/UCSC Xena/broad.mit.edu_PANCAN_Genome_Wide_SNP_6_whitelisted.xena", sep = "\t", header = T)
SNP.filtered.sum = aggregate(SNP.filtered$LENGTH, by=list(Category=SNP.filtered$Sample), FUN=sum)

# 1Mb = 1,000 kb = 1,000,000 pb
SNP.filtered.sum$LENGTH_KB = SNP.filtered.sum$x/1000

# remove normal group 10A, 11A, ...
SNP.filtered.sum = SNP.filtered.sum[as.numeric(substr(SNP.filtered.sum$Category, 14, 15)) == 1, ] # only based on primary cancer (01A)
barcode = substr(SNP.filtered.sum$Category, 1, 12)
length(unique(barcode)) == length(barcode)
# samebarcode = names(table(barcode)[table(barcode)>=2])
# same = unlist(lapply(samebarcode, function(x) which(grepl(x, SNP.filtered.sum$Category))))
# SNP.filtered.sum$Category[same]

#### Get patients information
pinfo = read.table("LUAD/CNB/data/UCSC Xena/TCGA_phenotype_denseDataOnlyDownload.tsv", sep = "\t", header = T)


pinfo$barcode = substr(pinfo$sample, 1, 12)

SNP.filtered.sum$CANCER = pinfo$X_primary_disease[match(barcode, pinfo$barcode)]


study.abbr = read.table("LUAD/CNB/data/TCGA study abbreviations.tsv", sep = "\t", header = T)
study.abbr$Study.Name = tolower(study.abbr$Study.Name)

SNP.f2 = SNP.filtered.sum[SNP.filtered.sum$CANCER %in% study.abbr$Study.Name,]
SNP.f2$CANCER_ABBR = study.abbr$Study.Abbreviation[match(SNP.f2$CANCER, study.abbr$Study.Name)]
SNP.f3 = SNP.f2[SNP.f2$CANCER_ABBR %in% c("BLCA", "BRCA","CESC","HNSC","KIRC", "KIRP", "LIHC", "LUAD",
                                          "LUSC", "OV", "PAAD","STAD"),]
table(SNP.f3$CANCER_ABBR)
dataset[['CNB']] = SNP.f3[SNP.f3$CANCER_ABBR == "LUAD",]
dataset[['CNB']] = data.frame(cbind(dataset[['CNB']]$Category, dataset[['CNB']]$LENGTH_KB))
colnames(dataset[['CNB']]) = c("barcode", "LENGTH_KB")
dataset[['CNB']]$barcode = unlist(lapply(dataset[['CNB']]$barcode, function(x) substr(x, 1, 12) ))

print(paste0('[CNB] ', dim(dataset[['CNB']])[1], ' complete rows found in clinical data.'))

########################################################################
#                      mRNA-seq
########################################################################
RNAseq.origin = read.table("LUAD/mRNAseq/gdac.broadinstitute.org_LUAD.Merge_rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__RSEM_genes_normalized__data.Level_3.2016012800.0.0/LUAD.rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__RSEM_genes_normalized__data.data.txt",
                           sep = "\t", header = T, row.names = 1)
RNAseq = RNAseq.origin[2:dim(RNAseq.origin)[1],]
# only keep primary cancer patients
RNAseq = RNAseq[, as.numeric(substr(colnames(RNAseq), 14, 15)) < 10]
coln = gsub(".", '-', substr(colnames(RNAseq), 1, 12), fixed = T)
rown = rownames(RNAseq)

RNAseq = as.data.frame(matrix(as.numeric(as.matrix(RNAseq)), nrow = dim(RNAseq)[1], byrow = F))
colnames(RNAseq) = coln
rownames(RNAseq) = rown

# convert na to 0
RNAseq[is.na(RNAseq)] <- 0
RNAseq.filtered = fastFilter(RNAseq, lowest_percentile_mean = 0.2,
                             lowest_percentile_variance = 0.2, var.func = "var")
## Remove duplicated gene symbols (only keep with max expression)
uniGene = gsub("\\|.*$","", rownames(RNAseq.filtered))
row2remove <- numeric()
finalSymCharTable <- table(uniGene)
for (i in 1:length(finalSymCharTable)){
  if (as.numeric(finalSymCharTable[i]) > 1){ # if exist duplicated Gene
    genename <- names(finalSymCharTable[i])
    idx_with_max_mean <- which.max(rowMeans(RNAseq.filtered[which(uniGene == genename),]))
    # print(idx_with_max_mean)
    row2remove <- c( row2remove, (which(uniGene == genename)[-idx_with_max_mean]) )
  }
}
if (length(row2remove) > 0){ # Otherwise numerical(0) will remove all data in tmpExp
  RNAseq.filtered <- RNAseq.filtered[-row2remove,]
  uniGene <- uniGene[-row2remove]
}
#Remove gene symbol after vertical line: from ABC|123 to ABC:
rownames(RNAseq.filtered) <- gsub("\\|.*$","", rownames(RNAseq.filtered))

print(paste('perform log2 transform on RNAseq with shape', dim(RNAseq.filtered)[1], 'x', dim(RNAseq.filtered)[2]))
RNAseq.filtered = log2(RNAseq.filtered+1)

dataset[['mRNAseq']] = RNAseq.filtered
########################################################################
#                      miRNA-seq
########################################################################
miRNAseq = read.table("LUAD/miRNAseq/gdac.broadinstitute.org_LUAD.Merge_mirnaseq__illuminahiseq_mirnaseq__bcgsc_ca__Level_3__miR_gene_expression__data.Level_3.2016012800.0.0/LUAD.mirnaseq__illuminahiseq_mirnaseq__bcgsc_ca__Level_3__miR_gene_expression__data.data.txt",
                      sep = "\t", header = T, row.names = 1)
miRNAseq = miRNAseq[,miRNAseq[1,] == "reads_per_million_miRNA_mapped"]
miRNAseq = miRNAseq[2:dim(miRNAseq)[1],]
# only keep primary cancer patients
miRNAseq = miRNAseq[, as.numeric(substr(colnames(miRNAseq), 14, 15)) < 10]
coln = gsub(".", '-', substr(colnames(miRNAseq), 1, 12), fixed = T)
rown = rownames(miRNAseq)
miRNAseq = as.data.frame(matrix(as.numeric(as.matrix(miRNAseq)), nrow = dim(miRNAseq)[1], byrow = F))
colnames(miRNAseq) = coln
rownames(miRNAseq) = rown

# convert na to 0
miRNAseq[is.na(miRNAseq)] <- 0
miRNAseq.filtered = fastFilter(miRNAseq, lowest_percentile_mean = 0.2,
                             lowest_percentile_variance = 0.2, var.func = "var")
## Remove duplicated gene symbols (only keep with max expression)
uniGene = gsub("\\|.*$","", rownames(miRNAseq.filtered))
row2remove <- numeric()
finalSymCharTable <- table(uniGene)
for (i in 1:length(finalSymCharTable)){
  if (as.numeric(finalSymCharTable[i]) > 1){ # if exist duplicated Gene
    genename <- names(finalSymCharTable[i])
    idx_with_max_mean <- which.max(rowMeans(miRNAseq.filtered[which(uniGene == genename),]))
    # print(idx_with_max_mean)
    row2remove <- c( row2remove, (which(uniGene == genename)[-idx_with_max_mean]) )
  }
}
if (length(row2remove) > 0){ # Otherwise numerical(0) will remove all data in tmpExp
  miRNAseq.filtered <- miRNAseq.filtered[-row2remove,]
  uniGene <- uniGene[-row2remove]
}
#Remove gene symbol after vertical line: from ABC|123 to ABC:
rownames(miRNAseq.filtered) <- gsub("\\|.*$","", rownames(miRNAseq.filtered))

print(paste('perform log2 transform on miRNAseq with shape', dim(miRNAseq.filtered)[1], 'x', dim(miRNAseq.filtered)[2]))
miRNAseq.filtered = log2(miRNAseq.filtered+1)

dataset[['miRNAseq']] = miRNAseq.filtered

########################################################################
#                      mRNA-seq / miRNA-seq co-expression
########################################################################
lmQCM_object <- lmQCM(dataset[['mRNAseq']], gamma = 0.7, t = 1, lambda = 1, beta = 0.4,
                       minClusterSize = 10, CCmethod = "spearman", normalization = F)
dataset[['mRNAseq_eigengene_matrix']] = t(lmQCM_object@eigengene.matrix)

clusters.names = lmQCM_object@clusters.names
text.output = data.frame(row.names = 1:length(clusters.names))
for (i in 1:length(clusters.names)){
  text.output[i,1] = paste(unlist(clusters.names[i]), collapse = ', ')
}
write.csv(text.output, "LUAD/multiomics_preprocessing_results/mRNAseq_modules.csv")


lmQCM_object <- lmQCM(dataset[['miRNAseq']], gamma = 0.4, t = 1, lambda = 1, beta = 0.6,
                      minClusterSize = 4, CCmethod = "spearman", normalization = F)
dataset[['miRNAseq_eigengene_matrix']] = t(lmQCM_object@eigengene.matrix)


########################################################################
#                      get mutual patients
########################################################################

names(dataset)

bcd1 = dataset[['clinical']]$bcr_patient_barcode
bcd2 = rownames(dataset[['TMB']])
bcd3 = dataset[['CNB']]$barcode
bcd4 = rownames(dataset[['mRNAseq_eigengene_matrix']])
bcd5 = rownames(dataset[['miRNAseq_eigengene_matrix']])


bcd = Reduce(intersect, list(bcd1,bcd2,bcd3,bcd4,bcd5)) #use Reduce to get the intersection

dataset.intersection = {}
dataset.intersection[['clinical']] = dataset[['clinical']][bcd1 %in% bcd, ]
dataset.intersection[['TMB']] = dataset[['TMB']][bcd2 %in% bcd, ]
dataset.intersection[['CNB']] = dataset[['CNB']][bcd3 %in% bcd, ]
dataset.intersection[['mRNAseq_eigengene_matrix']] = dataset[['mRNAseq_eigengene_matrix']][bcd4 %in% bcd, ]
dataset.intersection[['miRNAseq_eigengene_matrix']] = dataset[['miRNAseq_eigengene_matrix']][bcd5 %in% bcd, ]
dataset.intersection[['mRNAseq']] = t(dataset[['mRNAseq']][colnames(dataset[['mRNAseq']]) %in% bcd, ])
dataset.intersection[['miRNAseq']] = t(dataset[['miRNAseq']][colnames(dataset[['miRNAseq']]) %in% bcd, ])

write.csv(dataset.intersection[['clinical']], file = "LUAD/multiomics_preprocessing_results/clinical.csv")
write.csv(dataset.intersection[['TMB']], file = "LUAD/multiomics_preprocessing_results/TMB.csv")
write.csv(dataset.intersection[['CNB']], file = "LUAD/multiomics_preprocessing_results/CNB.csv")
write.csv(dataset.intersection[['mRNAseq_eigengene_matrix']], file = "LUAD/multiomics_preprocessing_results/mRNAseq_eigengene_matrix.csv")
write.csv(dataset.intersection[['miRNAseq_eigengene_matrix']], file = "LUAD/multiomics_preprocessing_results/miRNAseq_eigengene_matrix.csv")
write.csv(dataset.intersection[['mRNAseq']], file = "LUAD/multiomics_preprocessing_results/mRNAseq.csv")
write.csv(dataset.intersection[['miRNAseq']], file = "LUAD/multiomics_preprocessing_results/miRNAseq.csv")
