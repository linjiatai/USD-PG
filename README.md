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

## Usage

### Dataset Preparation
You should download the GDPH-CRC-HE-MS dataset at [OneDrive](https://1drv.ms/u/c/a5c29d99ada8ad03/EYawHkl-3kJClFUywW1rxAEBMJREziay6_CrIVc7wBDlfQ?e=Sir6wg) or [Baidu Netdisk](https://pan.baidu.com/s/1k_ScOvAERWrjhJWfAA6RyQ?pwd=t3c2) (with pass code **t3c2**) and you can put them into ```DATA/``` fold with the following data structure. Then we randomly selected 1% of the samples for the second-stage model training and placed them in the ```DATA/one_percent_labeled``` folder, with the structure as shown below::
```
DATA/

    |_one_percent_labeled/
    |     |_ S01
    |         |_ train
    |            |_ BACK/
    |            |_ NORM/
    |            |_ MUC/
    |            |_ DEB/
    |            |_ LYM/
    |            |_ ADI/
    |            |_ STR/
    |            |_ TUM/
    |            |_ MUS/
    |         |_ val
    |            |_ ...
    |     |_ S02
    |     |_ S03
    |     |_ ...
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
      
```

### Main training
```
python 1-main_train.py --source_domain S03 --root ./DATA/GDPH-CRC-HE-MS/training/S01 --files ['preprocess/S03_LAB.yaml', 'preprocess/S03_HSV.yaml', 'preprocess/S03_HED.yaml']
```

### Fine-turning
```
python 2-main_lincls.py --source_domain S03
```
### Testing
```
python 3-patch_cls_score.py --source_domain S03 --target_domain S01
```
## Citation
If you find the code useful, please consider citing our paper using the following BibTeX entry.
```
@article{lin2026unsupervised,
  title={Unsupervised single-domain generalization for tissue classification via progressive domain transformation},
  author={Lin, Jiatai and Li, Qian and Cui, Yanfen and Zhao, Bingchao and Deng, Tianpeng and Huang, Jingqi and Shi, Zhenwei and Cui, Enming and Liu, Zaiyi and Zhao, Ke and others},
  journal={Medical Image Analysis},
  pages={104118},
  year={2026},
  publisher={Elsevier}
}
```
