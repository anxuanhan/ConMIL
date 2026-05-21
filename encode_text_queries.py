"""
encode_text_queries.py
----------------------
将pos/neg文字queries编码为特征向量，保存为.npy文件

用法:
    python encode_text_queries.py \
        --checkpoint_path /path/to/CONCH/checkpoints/conch/pytorch_model.bin \
        --output_dir ./text_features

输出:
    text_features/pos_features.npy   shape: (P, 512)
    text_features/neg_features.npy   shape: (Q, 512)
    text_features/queries.txt        记录每行对应哪条query
"""

import os
import argparse
import numpy as np
import torch
from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer

# ================================================================
# 在这里修改你的queries
# ================================================================
POS_QUERIES = [
    "pleomorphic nuclei with prominent nucleoli and irregular nuclear membranes",
    "tightly packed malignant epithelial cell clusters",
    "metastatic adenocarcinoma in lymph node sinuses",
    "high nuclear to cytoplasmic ratio in malignant cells",
]

NEG_QUERIES = [
    "dense populations of small round mature lymphocytes",
    "normal lymphoid stroma and germinal centers",
]
# ================================================================


def load_conch_model(checkpoint_path, device):
    print(f"📦 Loading CONCH model from: {checkpoint_path}")
    model, _ = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=checkpoint_path)
    model.eval()
    model = model.to(device)
    print("✅ Model loaded")
    return model


def encode_texts(model, queries, device):
    """
    Returns: np.ndarray (Q, 512), L2-normalized
    """
    tokenizer = get_tokenizer()
    tokenized = tokenize(tokenizer, queries).to(device)
    with torch.inference_mode():
        feats = model.encode_text(tokenized, normalize=True)
    return feats.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Encode text queries with CONCH")
    parser.add_argument('--checkpoint_path', required=True,
                        help='Path to CONCH pytorch_model.bin')
    parser.add_argument('--output_dir', default='./text_features',
                        help='Directory to save .npy files (default: ./text_features)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")

    model = load_conch_model(args.checkpoint_path, device)
    os.makedirs(args.output_dir, exist_ok=True)

    # 编码pos queries
    print(f"\n📝 Encoding {len(POS_QUERIES)} positive queries...")
    for i, q in enumerate(POS_QUERIES):
        print(f"  [{i}] {q}")
    pos_feats = encode_texts(model, POS_QUERIES, device)  # (P, 512)
    print(f"   → shape: {pos_feats.shape}, norm[0]: {np.linalg.norm(pos_feats[0]):.4f}")

    # 编码neg queries
    print(f"\n📝 Encoding {len(NEG_QUERIES)} negative queries...")
    for i, q in enumerate(NEG_QUERIES):
        print(f"  [{i}] {q}")
    neg_feats = encode_texts(model, NEG_QUERIES, device)  # (Q, 512)
    print(f"   → shape: {neg_feats.shape}, norm[0]: {np.linalg.norm(neg_feats[0]):.4f}")

    # 保存
    pos_path = os.path.join(args.output_dir, 'pos_features.npy')
    neg_path = os.path.join(args.output_dir, 'neg_features.npy')
    np.save(pos_path, pos_feats)
    np.save(neg_path, neg_feats)
    print(f"\n💾 Saved:")
    print(f"   {pos_path}  {pos_feats.shape}")
    print(f"   {neg_path}  {neg_feats.shape}")

    # 保存query文本记录
    txt_path = os.path.join(args.output_dir, 'queries.txt')
    with open(txt_path, 'w') as f:
        f.write("=== Positive Queries ===\n")
        for i, q in enumerate(POS_QUERIES):
            f.write(f"[{i}] {q}\n")
        f.write("\n=== Negative Queries ===\n")
        for i, q in enumerate(NEG_QUERIES):
            f.write(f"[{i}] {q}\n")
    print(f"   {txt_path}  (query记录)")
    print(f"\n✅ Done!")


if __name__ == '__main__':
    main()