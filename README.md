# Breaking Camouflage: classifying camouflaged subjects in COD10K

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danishali-codes/breaking-camouflage-cod10k/blob/main/Week6_AppliedViT_5584978.ipynb)

Danish Ali, School of Computer Science, University of Wollongong in Dubai

This repository holds the code and results for the paper "Breaking Camouflage through Pre-training, Resolution and Scale in Fine-grained Image Classification". The work began as coursework for MAAI901 Deep Learning for Visual Data. It compares four pre-trained model families on 52 sub-classes of camouflaged subjects from COD10K-v3 (Fan et al., 2020). One of the 52 classes is human, so the task is not only animal recognition.

| Family | Pre-training |
|---|---|
| ViT | supervised, ImageNet-21k then ImageNet-1k labels |
| ConvNeXt | supervised CNN, ImageNet-22k then ImageNet-1k labels |
| CLIP | 400M image and text pairs |
| DINOv2 | self-supervised on LVD-142M, no labels |

Each family was fine-tuned in three configurations: Base at 224 px, Base at 448 px and Large at 224 px. Large at 448 px was not run. Every configuration used three seeds, and backbone learning rates were chosen on a validation split. The dataset's masks were used to test how much the models rely on the subject and how much on the rest of the image.

## Results

Top-1 accuracy on the 1,949-image test set, mean ± standard deviation over three seeds. Uniform random guessing gives 1.9%, and always predicting the most frequent class (pipefish) gives 7.2%.

| Model | Base, 224 px | Base, 448 px | Large, 224 px |
|---|---|---|---|
| DINOv2 | 78.1 ± 0.1 | 83.6 ± 0.7 | **84.3 ± 1.2** |
| ConvNeXt | 77.0 ± 0.6 | 82.9 ± 0.8 | 79.5 ± 1.0 |
| ViT | 70.8 ± 0.6 | 78.6 ± 1.0 | 75.1 ± 0.5 |
| CLIP | 70.0 ± 0.6 | 72.3 ± 2.2 | 76.0 ± 1.4 |

Main findings:

- The family name alone does not predict accuracy. At Base 224 px, DINOv2 and ConvNeXt are not statistically distinguishable under a paired bootstrap, and both beat the supervised ViT by 6 to 7 points. The four configurations differ in training data, objective, patch size and architecture at once, so these results do not separate the effect of pre-training from the effect of architecture.
- Resolution helps small subjects most. Training at 448 px adds 5 to 8 points for three of the four families. For subjects covering under 2.1% of the image, the gain is 10.8 to 12.3 points.
- Much of the predictive information lies outside the subject's appearance. With the subject's pixels filled with flat grey, the models still score 38 to 59%. The grey region keeps the subject's outline, position and size, so this figure is an upper bound on what the habitat alone provides.

## Running the notebook

1. Open `Week6_AppliedViT_5584978.ipynb` in Colab (use the button above).
2. Runtime → Change runtime type → GPU. An A100 is fastest; a T4 also works.
3. Runtime → Run all.

No setup is needed. The notebook installs its packages and downloads COD10K-v3 automatically from a public Kaggle copy of the official release (about 2.5 GB). To use your own copy instead, drop `COD10K-v3.zip` into Colab's Files panel before running, or set `DATA_SOURCE = "upload"` in the config cell.

Two run modes, set with `RUN_MODE` in the config cell:
- `"full"` reproduces every number in the paper. It takes about 3 GPU-hours on an A100.
- `"quick"` runs the whole pipeline with 1 seed and 3 epochs, in roughly an hour on a T4. Its numbers are lower than the paper's; use it to check that everything runs.

The results in `results/` are from the full run.

## Reproducibility details

Pre-trained weights:

| Model | Library | Identifier |
|---|---|---|
| ViT-B/16 | timm | `vit_base_patch16_224.augreg2_in21k_ft_in1k` |
| ViT-L/16 | timm | `vit_large_patch16_224.augreg_in21k_ft_in1k` |
| ConvNeXt-B | timm | `convnext_base.fb_in22k_ft_in1k` |
| ConvNeXt-L | timm | `convnext_large.fb_in22k_ft_in1k` |
| DINOv2 ViT-B/14 | timm | `vit_base_patch14_dinov2.lvd142m` |
| DINOv2 ViT-L/14 | timm | `vit_large_patch14_dinov2.lvd142m` |
| CLIP ViT-B/16 | OpenCLIP | `ViT-B-16`, pretrained tag `openai` |
| CLIP ViT-L/14 | OpenCLIP | `ViT-L-14`, pretrained tag `openai` |

Classes and split:

- The 52 retained class names and the 17 dropped ones are in `results/dataset_info.json`. Per-class train and test counts are in the `classes` sheet of `results/ALL_RESULTS.xlsx`.
- The official COD10K train and test partitions are kept. A sub-class is retained if it has at least 10 training and 5 test images; the catch-all `Other` class is excluded.
- Validation is a stratified 15% of the retained training partition, made with scikit-learn's `train_test_split` and `random_state=0`. The split is the same for every seed. No list of image IDs is stored here; the notebook rebuilds the split from the dataset.
- Training seeds are 0, 1 and 2.

Image handling:

- Images are cached with the longest side reduced to at most 640 px.
- Training uses random resized crops with scale 0.7 to 1.0 and horizontal flips, with no colour jitter.
- Evaluation resizes the whole image to a square of the input size with bicubic interpolation. It does not centre-crop and does not preserve aspect ratio.

Mask tests, applied to the cached image before resizing:

- Subject only: every pixel outside the ground-truth mask is set to grey, RGB (124, 116, 104).
- Background only: every pixel inside the mask is set to the same grey.
- Crop: a square with side 1.3 times the longer side of the mask's bounding box plus 16 px, clipped to the image.

Statistics:

- The paired bootstrap averages per-image correctness over the three seeds of each configuration, then resamples the 1,949 test images 5,000 times and takes percentile intervals.
- Grad-CAM is computed for the true class on a fixed random sample of 500 test images. The target is the final stage for ConvNeXt and the first normalisation layer of the last transformer block for ViT, DINOv2 and CLIP.

Library versions were not pinned or recorded. The notebook installs the current `timm` and `open_clip_torch` when it runs.

## Repository contents

```
Week6_AppliedViT_5584978.ipynb   all experiments, with saved outputs
make_figure.py                   rebuilds the paper's Figure 1 from ALL_RESULTS.xlsx
results/
  ALL_RESULTS.xlsx               every result, one sheet per analysis
  all_runs.csv                   one row per run (zero-shot, probes, sweep, main runs)
  best_lr.json                   backbone learning rate chosen for each family
  dataset_info.json              the 52 classes kept, classes dropped, split sizes
  figures/                       charts produced by the notebook
  runs/                          per-run metrics (.json) and test-set predictions (.npz)
```

Each `.npz` in `runs/` holds the test logits and labels for one run, in the order of the sorted test-image file names. They are enough to recompute the accuracy, bootstrap and per-class results without retraining.

To rebuild Figure 1 with its error bars:

```
python make_figure.py results/ALL_RESULTS.xlsx subject_size.pdf
```

## Data and references

COD10K-v3 is not included here. It is released by the Media Computing Lab, Nankai University, for non-commercial research use.

- Fan, D.-P., Ji, G.-P., Sun, G., Cheng, M.-M., Shen, J. and Shao, L. (2020) 'Camouflaged object detection', *CVPR*.
- Pre-trained weights come from [timm](https://github.com/huggingface/pytorch-image-models) and [OpenCLIP](https://github.com/mlfoundations/open_clip).
