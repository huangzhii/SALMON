# SALMON 🐟
SALMON: Survival Analysis Learning with Multi-Omics Neural Networks

### SALMON architecture with the implementation of Cox proportional hazards regression networks:
![SALMON Architecture](figures/Figure1.png)

### Publication:
Huang, Zhi, et al. "SALMON: Survival Analysis Learning with Multi-Omics Neural Networks on Breast Cancer." Frontiers in Genetics 10 (2019): 166. [[Link]](https://www.frontiersin.org/articles/10.3389/fgene.2019.00166/abstract)


## 1. Installation
* Download Anaconda from https://www.anaconda.com/download/#linux/ with python 3 version.
* Install Pytorch, tqdm, imblearn, and lifelines.

```bash
conda install pytorch torchvision -c pytorch
conda install -c conda-forge tqdm
conda install -c conda-forge imbalanced-learn
conda install -c conda-forge lifelines
```

### Alternative installation
To install the libraries with the specified version, `requirements.txt` can be used as a configuration file with the following command:

```bash
pip install -r requirements.txt
```

## 2. Run the main routine


```bash
cd experiments
python main.py
```
