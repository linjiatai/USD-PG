# Unsupervised Single-Domain Generalization for Tissue Classification via Progressive Domain Transformation

<img width="1404" height="933" alt="87fe9731-9903-40e2-ba58-950df7ec7964" src="https://github.com/user-attachments/assets/5a1d00fb-d843-4838-9951-aa021040457c" />

## Introduction
The iplementation of:
**[Unsupervised Single-Domain Generalization for Tissue Classification via Progressive Domain Transformation](https://)**

You can also download the repository from https://github.com/linjiatai/USD-PG.git.

## Abtract
Tissue classification is one of the fundamental tasks in computational pathology, but domain shifts in digital pathology images limit the generalization of classification
models. Domain generalization has emerged as a leading solution to address this gap, with related research often using multiple public datasets to demonstrate model
generalization ability across different sources. To further explore this, we introduce the GDPH-CRC-HE-MS dataset, consisting of 101 H&E-stained colorectal cancer slides
from Guangdong Provincial People’s Hospital, scanned by 1 to 6 different scanners. In this study, we propose an unsupervised single-domain progressive generalization
(USD-PG) framework, which incorporates two progressive data transformations: style progressive data transformation (Style-PDT) and spatial progressive data transformation
(Spatial-PDT). This approach prevents unreasonable texture and color changes caused by completely random transformations during the early training stages. We evaluate the
generalization ability of the USD-PG framework on the new GDPH-CRC-HE-MS dataset as well as the publicly available NCT-CRC-HE-100K dataset. Our results demonstrate
that USD-PG achieves superior performance in single-source domain generalization for tissue classification, effectively handling both scanner-based and data-source domain
shifts. It highlights the potential of USD-PG for enhancing domain generalization in tissue classification and its applicability in clinical settings.

## Requirements
- CUDA
- 1×GPU
- Python 3.7
- numpy==1.21.5
- pytorch==1.11.0
- torchvision==0.12.0
- scikit-learn==0.24.2
## Usage

### GDPH-CRC-HE-MS Dataset
You should download the GDPH-CRC-HE-MS dataset at [OneDrive](https://1drv.ms/u/c/a5c29d99ada8ad03/EYawHkl-3kJClFUywW1rxAEBMJREziay6_CrIVc7wBDlfQ?e=Sir6wg) or [Baidu Netdisk](https://pan.baidu.com/s/1k_ScOvAERWrjhJWfAA6RyQ?pwd=t3c2) (with pass code **t3c2**), and the NCT-CRC-HE-100K dataset at [Official Website](https://zenodo.org/records/1214456). Then, you can put them into ```DATA/``` fold with the following data structure:
```
DATA/

    |_ GDPH-CRC-HE-MS/
    |     |_ training/
    |         |_ S01/
    |            |_ BACK/
    |            |_ NORM/
    |            |_ MUC/
    |            |_ DEB/
    |            |_ LYM/
    |            |_ ADI/
    |            |_ STR/
    |            |_ TUM/
    |            |_ MUS/
    |         |_ S02/
    |         |_ S03/
    |         |_ S04/
    |         |_ S05/
    |         |_ S06/
    |     |_ test/
    |         |_ S01/
    |            |_ .../
    |_ NCT-CRC-HE-100K/
    |      |_ ...
      
```

### Pretraining
```
python main.py --...
```

### Fine-turning
```
python main.py --...
```
### Testing
```
python main.py --...
```
## Citation
If you find the code useful, please consider citing our paper using the following BibTeX entry.
```
@ARTICLE{lin2026usd-pg,
  author={Lin, Jiatai and Han, Guoqiang and Pan, Xipeng and Liu, Zaiyi and Chen, Hao and Li, Danyi and Jia, Xiping and Shi, Zhenwei and Wang, Zhizhen and Cui, Yanfen and Li, Haiming and Liang, Changhong and Liang, Li and Wang, Ying and Han, Chu},
  journal={IEEE Transactions on Medical Imaging}, 
  title={PDBL: Improving Histopathological Tissue Classification with Plug-and-Play Pyramidal Deep-Broad Learning}, 
  year={2022},
  pages={1-1},
  doi={10.1109/TMI.2022.3161787}}
```
