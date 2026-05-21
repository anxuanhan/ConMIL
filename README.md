# ConMIL: interactive and contrastive text-guided multiple instance learning for whole slide image classification
<img width="1565" height="784" alt="image" src="https://github.com/user-attachments/assets/101bef46-c243-4336-af91-1eabe7586d18" />




## ✅ Requirements
- Python 3.8+
- PyTorch + torchvision
- CONCH codebase (conch.open_clip_custom)
- openslide-python + OpenSlide system libs
- h5py, numpy, pandas, scikit-learn
- opencv-python, pillow, matplotlib, tifffile
- Optional normalization: wsi-normalizer and/or torchstain

## 📂 Data Format

## 📁 Slides Folder
The slides folder should contain WSI files `.svs` `.tif` `.ndpi`

### CAMELYON16
For CAMELYON16, the official dataset already provides a reference file, `reference.csv`, which can be directly used as the label CSV.

The CSV file should contain:

```csv
image,type
tumor_001.tif,tumor
tumor_002.tif,tumor
normal_001.tif,normal
normal_002.tif,normal
```

Here, `image` refers to the slide filename, and `type` refers to the slide-level class label.

### TCGA-BRCA
For TCGA-BRCA, users need to manually organize the labels into a CSV file.

The label CSV should contain:
- `patient`: patient or slide identifier
- `type`: subtype label (e.g., `IDC` or `ILC`)

Example `labels.csv`:

```csv
patient,type
TCGA-3C-AALI,IDC
TCGA-3C-AALJ,IDC
TCGA-3C-AALK,IDC
TCGA-3C-AALM,ILC
```

The identifiers in the CSV file should match the slide filenames or patient IDs used in the dataset.



## 🚀 How to run
Set your paths and run step by step.

### 1) Tissue mask generation or mask preparation

For CAMELYON16, tissue masks are already available from the dataset, so this preprocessing step can be skipped. You can directly use the provided masks for feature extraction.

For TCGA-BRCA, tissue masks need to be generated before feature extraction. Run:

```bash
python Preprocessing/process_mark.py \
  -pic_path "HE_slides_raw" \
  -file_type svs \
  -min_area 1000000 \
  -min_hole 100000 \
  -out_dir "Process_pics" \
  -red_dilate_iterations 1 \
  -blue_dilate_iterations 3 \
  -green_dilate_iterations 8 \
  -black_kernel_size 8 \
  -black_close_iterations 12 \
  -black_dilate_iterations 10 \
  -overwrite
```

### 2) Feature extraction with CONCH

Download the CONCH checkpoint from the official [CONCH GitHub repository](https://github.com/mahmoodlab/CONCH) and specify the checkpoint path in the command below.

This step performs:
- tissue patch extraction from WSIs
- optional Macenko stain normalization
- feature extraction using the CONCH vision encoder

Key arguments:
- `-patch_size`: patch size used for feature extraction
- `-read_size`: image region size read from the slide before resizing
- `-level`: WSI pyramid level used for patch extraction
- `-m macenko`: apply Macenko stain normalization
- `--batch_size`: inference batch size
- `-blank_threshold`: threshold for filtering background patches

Example:

```bash
python cut_norm_feature_copy.py \
 -input_folder /path/to/slides \
 -mask_folder /path/to/masks \
 -output_folder /path/to/features_conch \
 -patch_size 224 \
 -read_size 256 \
 -level 1 \
 -file_type svs \
 -m macenko \
 --batch_size 128 \
 --checkpoint_path /path/to/CONCH/pytorch_model.bin \
 --num_workers 8 \
 -blank_threshold 0.6
```

### 3) Encode text prompts

Before running this step, edit `POS_QUERIES` and `NEG_QUERIES` in `encode_text_queries.py` to match the pathological characteristics of your dataset and classification task.

This step encodes the pathological text prompts into text embeddings using the CONCH text encoder. These text features are later used for contrastive semantic similarity computation during text-guided MIL training and patch retrieval.

Example:

```bash
python encode_text_queries.py \
  --checkpoint_path /path/to/CONCH/pytorch_model.bin \
  --output_dir /path/to/text_features
```

### 4) Patch retrieval for prompt checking

After encoding the text prompts, you can retrieve the top-ranked patches based on their semantic similarity to the text prompts. This step is useful for checking whether the prompts are retrieving pathologically relevant regions.

If the retrieved patches do not match the expected pathological patterns, you can revise `POS_QUERIES` and `NEG_QUERIES` in `encode_text_queries.py`, then re-run the text encoding and patch retrieval steps.

Key arguments:
- `--text_feature_dir`: directory containing encoded text features
- `--h5_dir`: directory containing extracted WSI features
- `--topk`: number of top-ranked patches to retrieve for each slide
- `--neg_weight`: weighting factor for negative prompt similarity
- `--mode`: retrieval mode, such as positive-only or combined positive-negative scoring
- `--sharpness_thresh`: threshold for filtering blurry patches

Example:

```bash
python retrieve_patches.py \
  --text_feature_dir /path/to/text_features \
  --h5_dir /path/to/features_conch \
  --output_dir /path/to/retrieval_results \
  --slide_folder /path/to/slides
```


### 5) Train Text-Guided MIL (KAN head)

This step trains the ConMIL framework using extracted WSI features and encoded pathological text embeddings.

Key arguments:
- `--use_neg`: whether to use negative prompts for contrastive semantic scoring
- `--neg_weight`: weighting factor for negative prompt similarity
- `--kan_grid`: grid size used in the KAN classifier

Pretrained ConMIL checkpoints are also available on [Hugging Face](https://huggingface.co/ananananxuan/ConMIL/tree/main).

Example:

```bash
python train_text_guided_mil_kan.py \
    --feat_dir "$FEAT_DIR" \
    --label_csv "$LABEL_CSV" \
    --text_feat_dir "$TEXT_FEAT_DIR" \
    --use_neg true \
    --neg_weight 0.5 \
    --kan_grid 6 \
    --epochs 20 \
    --output "$OUTPUT_DIR"
```


### 6) Attention heatmap visualization (optional)

This step visualizes the patch-level attention weights learned by ConMIL on the whole-slide image. The generated heatmaps highlight regions that contribute most strongly to the final slide-level prediction.

These visualizations can be used to:
- inspect whether the model focuses on pathology-relevant regions
- compare attention patterns across different prompts or models
- support qualitative interpretation of weakly supervised learning behavior

Key arguments:

- `--h5`: extracted WSI feature file
- `--ckpt`: trained ConMIL checkpoint
- `--text_feat_dir`: directory containing encoded text embeddings

Example:

```bash
python visualize_attention_heatmap_v2.py \
  --slide /path/to/slides/example.svs \
  --mask /path/to/masks/example_mask.tif \
  --h5 /path/to/features_conch/example.h5 \
  --ckpt /path/to/results/textguided_kan6_posneg_best.pt \
  --text_feat_dir /path/to/text_features \
  --output /path/to/attention_map/example_heatmap_v2.png
```

### 7) KAN interpretability analysis (optional)

This step analyzes the learnable nonlinear activation functions in the KAN classifier. Unlike conventional MLP classifiers that use fixed activation functions, KAN learns edge-level nonlinear mappings, enabling more interpretable feature-response relationships.

The generated analysis can help:
- visualize how semantic features influence classification decisions
- inspect nonlinear activation patterns learned by the model
- improve interpretability beyond spatial attention heatmaps


Example:

```bash
python kan_interpretability.py \
  --ckpt /path/to/results/textguided_kan6_posneg_best.pt \
  --feat_dir /path/to/features_conch \
  --label_csv /path/to/labels.csv \
  --text_feat_dir /path/to/text_features \
  --output /path/to/kan_analysis
```
## 📝 Text prompts
Edit POS_QUERIES and NEG_QUERIES in encode_text_queries.py to match your dataset domain.

