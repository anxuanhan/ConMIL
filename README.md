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

1) (Optional) tissue mask/thumbnail preview
```bash
python process.py \
  -pic_path /path/to/slides \
  -out_dir /path/to/process_pics \
  -file_type svs \
  -min_area 1000000 \
  -min_hole 100000
```

2) Feature extraction with CONCH
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

3) Encode text prompts
```bash
python encode_text_queries.py \
  --checkpoint_path /path/to/CONCH/pytorch_model.bin \
  --output_dir /path/to/text_features
```

4) Train Text-Guided MIL (KAN head)
```bash
python train_text_guided_mil_kan.py \
  --feat_dir /path/to/features_conch \
  --label_csv /path/to/labels.csv \
  --text_feat_dir /path/to/text_features \
  --use_neg true \
  --neg_weight 0.5 \
  --kan_grid 6 \
  --epochs 20 \
  --output /path/to/results
```

5) Patch retrieval (optional)
```bash
python retrieve_patches.py \
  --text_feature_dir /path/to/text_features \
  --h5_dir /path/to/features_conch \
  --output_dir /path/to/retrieval_results \
  --slide_folder /path/to/slides \
  --file_type svs \
  --level 1 \
  --read_size 256 \
  --topk 20 \
  --neg_weight 0.5 \
  --mode combined \
  --sharpness_thresh 100
```

6) Attention heatmap (optional)
```bash
python visualize_attention_heatmap_v2.py \
  --slide /path/to/slides/example.svs \
  --mask /path/to/masks/example_mask.tif \
  --h5 /path/to/features_conch/example.h5 \
  --ckpt /path/to/results/textguided_kan6_posneg_best.pt \
  --text_feat_dir /path/to/text_features \
  --output /path/to/attention_map/example_heatmap_v2.png \
  --low_pct 1 --high_pct 99 --gamma 0.5
```

7) KAN interpretability (optional)
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

