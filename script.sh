#!/usr/bin/env bash
set -euo pipefail

# Dataset-agnostic template. Fill in paths for your dataset.
DATA_ROOT="/path/to/dataset"
SLIDES_DIR="$DATA_ROOT/slides"              # WSI files (.svs/.tif/.ndpi)
MASKS_DIR="$DATA_ROOT/masks"               # Optional tissue masks
FEAT_DIR="$DATA_ROOT/features_conch"       # Output H5 features
TEXT_FEAT_DIR="$DATA_ROOT/text_features"   # Output text features
LABEL_CSV="$DATA_ROOT/labels.csv"          # CSV with columns: image,type
OUTPUT_DIR="$DATA_ROOT/results"            # Training outputs
CONCH_CKPT="/path/to/CONCH/pytorch_model.bin"

# 1) (Optional) quick tissue mask/thumbnail visualization
python process.py \
    -pic_path "$SLIDES_DIR" \
    -out_dir "$DATA_ROOT/process_pics" \
    -file_type svs \
    -min_area 1000000 \
    -min_hole 100000

# 2) Feature extraction with CONCH (produces .h5 with features/coordinates)
python cut_norm_feature_copy.py \
    -input_folder "$SLIDES_DIR" \
    -mask_folder "$MASKS_DIR" \
    -output_folder "$FEAT_DIR" \
    -patch_size 224 \
    -read_size 256 \
    -level 1 \
    -file_type svs \
    -m macenko \
    --batch_size 128 \
    --checkpoint_path "$CONCH_CKPT" \
    --num_workers 8 \
    -blank_threshold 0.6

# 3) Encode positive/negative text prompts
python encode_text_queries.py \
    --checkpoint_path "$CONCH_CKPT" \
    --output_dir "$TEXT_FEAT_DIR"

# 4) Train Text-Guided MIL (KAN head)
python train_text_guided_mil_kan.py \
    --feat_dir "$FEAT_DIR" \
    --label_csv "$LABEL_CSV" \
    --text_feat_dir "$TEXT_FEAT_DIR" \
    --use_neg true \
    --neg_weight 0.5 \
    --kan_grid 6 \
    --epochs 20 \
    --output "$OUTPUT_DIR"

# 5) Patch retrieval (optional)
python retrieve_patches.py \
    --text_feature_dir "$TEXT_FEAT_DIR" \
    --h5_dir "$FEAT_DIR" \
    --output_dir "$DATA_ROOT/retrieval_results" \
    --slide_folder "$SLIDES_DIR" \
    --file_type svs \
    --level 1 \
    --read_size 256 \
    --topk 20 \
    --neg_weight 0.5 \
    --mode combined \
    --sharpness_thresh 100

# 6) Attention heatmap (optional)
python visualize_attention_heatmap_v2.py \
    --slide "$SLIDES_DIR/example.svs" \
    --mask "$MASKS_DIR/example_mask.tif" \
    --h5 "$FEAT_DIR/example.h5" \
    --ckpt "$OUTPUT_DIR/textguided_kan6_posneg_best.pt" \
    --text_feat_dir "$TEXT_FEAT_DIR" \
    --output "$DATA_ROOT/attention_map/example_heatmap_v2.png" \
    --low_pct 1 --high_pct 99 --gamma 0.5

# 7) KAN interpretability (optional)
python kan_interpretability.py \
    --ckpt "$OUTPUT_DIR/textguided_kan6_posneg_best.pt" \
    --feat_dir "$FEAT_DIR" \
    --label_csv "$LABEL_CSV" \
    --text_feat_dir "$TEXT_FEAT_DIR" \
    --output "$DATA_ROOT/kan_analysis"







