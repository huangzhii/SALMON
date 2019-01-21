# SALMON
SALMON: Survival Analysis Learning with Multi-Omics Neural Networks

### SALMON architecture with the implementation of Cox proportional hazards regression networks:
![SALMON Architecture](figures/Figure1.png)


# 1. Installation
* Download Anaconda from https://www.anaconda.com/download/#linux/ with python 3 version.
* Install Pytorch, tqdm, imblearn, and lifelines.

```bash
conda install pytorch torchvision -c pytorch
conda install -c conda-forge tqdm
conda install -c conda-forge imbalanced-learn
conda install -c conda-forge lifelines
```

# 2. Run the main routine


```bash
cd experiments
python main.py
```