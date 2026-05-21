# ConMIL (Text-Guided MIL)

This repo contains a dataset-agnostic pipeline for text-guided multiple instance learning (MIL) on WSI data with a KAN classifier head. It uses CONCH ViT-B-16 for patch feature extraction and prompt encoding.

## ✅ Requirements
- Python 3.8+
- PyTorch + torchvision
- CONCH codebase (conch.open_clip_custom)
- openslide-python + OpenSlide system libs
- h5py, numpy, pandas, scikit-learn
- opencv-python, pillow, matplotlib, tifffile
- Optional normalization: wsi-normalizer and/or torchstain

## 📂 Data format
- Slides folder contains WSI files (.svs, .tif, .ndpi).
- Label CSV must contain:
  - image: filename of the slide (e.g., tumor_005.tif)
  - type: class label string, expected values: tumor or normal

Example labels.csv:
```
image,type
tumor_001.tif,tumor
normal_003.tif,normal
```

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

## ⚠️ Notes / known gaps
The following files are referenced but not present in this folder and must be added if needed:
- text_guided_mil_kan.py (model builder imported by train_text_guided_mil_kan.py)
- text_guided_mil.py and train_text_guided_mil.py (baseline non-KAN training)
- visualize_attention_heatmap.py (older heatmap script, optional)
- Preprocessing/process_mark.py (if you want advanced tissue mask processing)

If these files already exist in other folders, copy them into this repo to make it self-contained.
