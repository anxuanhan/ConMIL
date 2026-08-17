# ConMIL: interactive and contrastive text-guided multiple instance learning for whole slide image classification
<img width="1565" height="784" alt="image" src="https://github.com/user-attachments/assets/101bef46-c243-4336-af91-1eabe7586d18" />




## ✅ Requirements
- Python == 3.10.16
- PyTorch == 1.13.1+cu116
- torchvision == 0.14.1+cu116
- NumPy == 1.26.4
- pandas == 2.2.2
- scikit-learn == 1.4.2
- h5py == 3.11.0

For the complete list of dependencies and package versions, please refer to `requirements.txt`.

## 📂 Data Format

The slides folder should contain WSI files (e.g., `.svs`, `.tif`, `.ndpi`).

The label CSV should contain the slide/patient identifier and the corresponding class label.

For CAMELYON16, the official `reference.csv` can be directly used.




## 🚀 How to run


### 1) Tissue mask generation or mask preparation

Generate tissue masks before feature extraction:

```bash
python process_mark.py \
  -pic_path "data/slides" \
  -out_dir "data/background_masks" \
  -file_type svs \
  -min_area 1000000 \
  -min_hole 100000
```

### 2) Feature extraction with CONCH

Download the CONCH checkpoint from the official [CONCH GitHub repository](https://github.com/mahmoodlab/CONCH) and specify the checkpoint path in the command below.

This step performs:
- tissue patch extraction from WSIs
- optional Macenko stain normalization
- feature extraction using the CONCH vision encoder


Example:

```bash
python cut_norm_feature.py \
  -input_folder "data/slides" \
  -mask_folder "data/background_masks" \
  -output_folder "features/features_conch" \
  -patch_size 224 \
  -read_size 256 \
  -level 1 \
  -file_type svs \
  -r "examples/reference_patch.png" \
  -m macenko \
  --batch_size 128 \
  --checkpoint_path "checkpoints/conch/pytorch_model.bin" \
  --num_workers 8 \
  -blank_threshold 0.6
```

### 3) Encode text prompts

Before running this step, edit `POS_QUERIES` and `NEG_QUERIES` in `encode_text_queries.py` to match the pathological characteristics of your dataset and classification task.

The prompts can be provided by pathology experts or generated with the assistance of a large language model (LLM), such as ChatGPT. When using an LLM, we recommend specifying the classification task and requesting discriminative patch-level pathological features for both positive and negative classes.

For example, the figure below shows an example interaction with ChatGPT for generating positive and negative pathological prompts for CAMELYON16:

<img width="819" height="494" alt="image" src="https://github.com/user-attachments/assets/43f18c8e-7f2a-4a88-8b15-7ebb4d35c037" />


Then run: 
```bash
python encode_text_queries.py \
  --checkpoint_path "checkpoints/conch/pytorch_model.bin" \
  --output_dir "text_features"
```

### 4) Patch retrieval for prompt checking

After encoding the text prompts, you can retrieve the top-ranked patches based on their semantic similarity to the text prompts. This step is useful for checking whether the prompts are retrieving pathologically relevant regions.

If the retrieved patches do not match the expected pathological patterns, you can revise `POS_QUERIES` and `NEG_QUERIES` in `encode_text_queries.py`, then re-run the text encoding and patch retrieval steps.



Example:

```bash
python retrieve_patches.py \
  --text_feature_dir "text_features" \
  --h5_dir "features/features_conch" \
  --output_dir "retrieval_results" \
  --slide_folder "data/slides" \
  --file_type svs \
  --level 1 \
  --read_size 256 \
  --topk 20 \
  --neg_weight 0.5 \
  --mode pos \   
  --sharpness_thresh 100
```
Available retrieval modes: pos / neg / combined

### 5) Train Text-Guided MIL (KAN head)

This step trains the ConMIL framework using extracted WSI features and encoded pathological text embeddings.


Pretrained ConMIL checkpoints are also available on [Hugging Face](https://huggingface.co/ananananxuan/ConMIL/tree/main).

Example:

```bash
python train_text_guided_mil_kan.py \
  --feat_dir "features/features_conch" \
  --label_csv "evaluation/reference.csv" \
  --text_feat_dir "text_features" \
  --use_neg true \
  --neg_weight 0.5 \
  --kan_grid 6 \
  --epochs 20 \
  --output "results"

```


### 6) Attention heatmap visualization (optional)

This step visualizes the patch-level attention weights learned by ConMIL on the whole-slide image. The generated heatmaps highlight regions that contribute most strongly to the final slide-level prediction.

These visualizations can be used to:
- inspect whether the model focuses on pathology-relevant regions
- compare attention patterns across different prompts or models
- support qualitative interpretation of weakly supervised learning behavior


Example:

```bash
python visualize_attention_heatmap.py \
  --slide "data/slides/example.svs" \
  --mask "data/masks/example_mask.tif" \
  --h5 "features/features_conch/example.h5" \
  --ckpt "results/textguided_posneg_best.pt" \
  --text_feat_dir "text_features" \
  --output "attention_maps/example_heatmap.png" \
  --low_pct 1 \
  --high_pct 99 \
  --gamma 0.5
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
  --ckpt "results/textguided_posneg_best.pt" \
  --feat_dir "features/features_conch" \
  --label_csv "evaluation/reference.csv" \
  --text_feat_dir "text_features" \
  --output "kan_analysis"
```

