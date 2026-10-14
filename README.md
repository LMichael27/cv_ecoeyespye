# EcoEyeSpy

Detecting and segmenting illegal waste sites in aerial and satellite imagery using the
[AerialWaste](http://aerialwaste.org) dataset.

Course project for *Deep Learning for Computer Vision*, Bocconi University.
**Authors:** Sofia Villa, Michael Ladaa, Omar Mukhtar

📄 **Paper:** [paper/EcoEyeSpy_report.pdf](paper/EcoEyeSpy_report.pdf)

---

## Overview

Waste deposits in overhead imagery are small, rare and easily confused with construction
sites or industrial structures. In AerialWaste only ~1.4% of the 11,703 images carry object
annotations, so we work in a sparse-supervision regime. EcoEyeSpy combines three
complementary stages:

| Stage | Model | Purpose | Main result |
|---|---|---|---|
| 1. Scene screening | ResNet-50 + FPN binary classifier | Pre-filter images that likely contain waste | ROC–AUC ≈ 0.88, positive recall 0.98 |
| 2. Detection | YOLOv8m (896 px) | Localise individual waste instances | mAP50 ≈ 0.35, mAP50–95 ≈ 0.18 |
| 3. Segmentation | DeepLabV3 (ResNet-50) and few-shot SAM | Pixel-level delineation of waste regions | DeepLabV3 test mIoU 0.72, SAM (100-shot) mean IoU 0.74 |

## Repository structure

```
binary_classification/   Image-level waste vs. background classifier (ResNet-50 + FPN)
deeplab_training/        DeepLabV3 semantic segmentation: training, validation, test, results
fewshot_training/        Few-shot fine-tuning of the Segment Anything Model (SAM)
shared/                  Dataset, metrics, IoU, logging and plotting helpers for segmentation
utility/                 Data preparation: patch split, rasterisation, label map, bbox extraction
paper/                   Final project report
```

The YOLOv8 detection notebooks live on the `sofia-villa` branch of the
[original repository](https://github.com/OmarN-Mukhtar/cv_ecoeyespye).

## Binary classification (stage 1)

| File | Description |
|---|---|
| `binary.py` | Training script for the ResNet-50 + FPN classifier, writes test-set predictions |
| `Binary_test.ipynb` | Notebook version of the training pipeline with validation and threshold tuning |
| `Plots.ipynb` | Evaluation plots: confusion matrix, ROC and PR curves, probability distributions, metrics vs. threshold |
| `binary_for_patches.ipynb` | Patch/box-level variant: ResNet-50 classifier on annotated box crops, with qualitative panels |
| `bbox_fasterrcnn.py` | Experimental Faster R-CNN (ResNet-50 FPN) box detector on the patch dataset |
| `results/` | Test predictions and validation metrics |

**Method.**
- Labels come from the image-level `severity` attribute (`severity ≥ 1` → waste), so the whole
  dataset can be used rather than only the 166 images with box annotations.
- Stratified train/validation split (85/15).
- Class imbalance is handled in two ways: a positive-class weight (`neg/pos`) in the BCE loss
  and a `WeightedRandomSampler` that balances the batches.
- Images are resized to 256×256 and normalised with ImageNet statistics. Albumentations adds
  flips, small shift/scale/rotate transforms and brightness/contrast jitter.
- Architecture: ResNet-50 features c2–c5 → FPN (256 channels) → global average pooling per
  level → concatenated 1024-d vector → dropout → linear logit.
- Training uses AdamW with mixed precision. The backbone is frozen for the first epochs and
  then fine-tuned. The checkpoint with the best validation ROC–AUC is kept.

**Results (validation, threshold 0.5).**

| | Precision | Recall | F1 |
|---|---|---|---|
| Background | 0.98 | 0.64 | 0.77 |
| Waste | 0.57 | 0.98 | 0.72 |

ROC–AUC ≈ 0.88, accuracy 0.75, weighted F1 0.76. The operating point deliberately favours recall
so that few waste sites are missed at the screening stage. Ablations show that removing either
the class weight or the weighted sampler sharply reduces positive recall. The freeze–unfreeze
schedule gives more stable convergence than fine-tuning the backbone from the start.

The patch-level box classifier reaches ROC–AUC 0.94 and F1 0.88 on its validation split
(`results/val_metrics_boxclf.csv`).

## Data

Download AerialWaste from <http://aerialwaste.org>. The scripts expect, relative to the working
directory:

```
training.json, testing.json        COCO-style annotation files
images0/ … images6/  (or images/)  image folders
patches_bbox.json, patches/        patch dataset (generated with utility/)
```

Datasets and model weights (`*.pth`) are not tracked in git.

## Requirements

Python 3.10+, `torch`, `torchvision`, `albumentations`, `scikit-learn`, `pandas`, `numpy`,
`Pillow`, `matplotlib`, `seaborn`. The detection stage additionally needs `ultralytics`, and the
few-shot stage needs `segment-anything`.

```bash
cd binary_classification
python binary.py
```

## References

- R. N. Torres and P. Fraternali. *AerialWaste dataset for landfill discovery in aerial and satellite images.* Scientific Data, 10:63, 2023.
- L.-C. Chen, G. Papandreou, F. Schroff, H. Adam. *Rethinking atrous convolution for semantic image segmentation.* 2017.
- A. Kirillov et al. *Segment Anything.* arXiv:2304.02643, 2023.
