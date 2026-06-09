# Attack-Resilient Differential Privacy-Enhanced Federated Learning for Multi-Centre Polyp Segmentation

## Overview

Medical image segmentation models often require data from multiple healthcare institutions to achieve strong performance. However, sharing patient data across centres raises significant privacy concerns.

This project develops a privacy-preserving federated learning framework for polyp segmentation using the PolypGen dataset. The framework combines Federated Learning (FL) and Differential Privacy (DP) to enable collaborative training while protecting sensitive patient information.

The work evaluates both model performance and privacy robustness through Membership Inference Attacks (MIA) and Gradient Leakage Attacks.

**Research Paper:** Accepted for presentation at ICT4SD 2026.


## Objectives

* Train a centralized segmentation model as a baseline.
* Implement Federated Learning using FedAvg.
* Integrate Differential Privacy using Opacus DP-SGD.
* Evaluate segmentation performance under privacy constraints.
* Assess privacy protection against inference and reconstruction attacks.


## Dataset

**PolypGen Dataset**

* 2225 colonoscopy images
* Multi-centre medical imaging dataset
* Partitioned into 5 non-IID clients to simulate distributed hospitals


## Methodology

### Centralized Learning (CL)

All training data is combined and used to train a single U-Net model.

### Federated Learning (FL)

Each client trains locally on its own dataset. Model updates are aggregated using Federated Averaging (FedAvg) without sharing raw patient data.

### Federated Learning with Differential Privacy (FL+DP)

Differential Privacy is applied using Opacus DP-SGD by:

* Gradient clipping
* Gaussian noise addition
* Privacy accounting

This reduces the risk of sensitive information leakage from model updates.


## Technology Stack

* Python
* PyTorch
* Flower
* Opacus
* NumPy
* Matplotlib
* U-Net Architecture


## Evaluation Metrics

* Dice Coefficient
* Intersection over Union (IoU)
* Precision
* Recall
* F1 Score


## Experimental Results

| Method                          | Dice Score |
| ------------------------------- | ---------- |
| Centralized Learning (CL)       | 0.8936     |
| Federated Learning (FL)         | 0.8949     |
| Federated Learning + DP (FL+DP) | 0.8593     |

### Key Observation

Federated Learning achieved performance comparable to centralized training while eliminating the need to share raw patient data. Differential Privacy significantly reduced privacy leakage with only a modest reduction in segmentation performance.


## Privacy Attack Evaluation

### Membership Inference Attack (MIA)

Evaluated whether an attacker could determine if a sample was part of the training dataset.

Result:

* FL ROC-AUC ≈ 0.55
* FL+DP ROC-AUC ≈ 0.54

### Gradient Leakage Attack

Attempted reconstruction of original images from shared gradients.

Result:

* CL and FL: Recoverable image information
* FL+DP: Significant degradation in reconstruction quality


## Future Work

* Secure Aggregation
* Adaptive Privacy Budget Scheduling
* Larger Multi-Centre Deployments
* Transformer-based Medical Segmentation Models

## Citation

If you use this work, please cite the corresponding ICT4SD 2026 conference paper after publication.
