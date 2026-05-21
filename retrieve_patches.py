"""
retrieve_patches.py
-------------------
加载预先计算好的文字特征(.npy) 和 patch特征(.h5)，
计算相似度，检索最相似的patches并保存结果图像。

用法:
    python retrieve_patches.py \
        --text_feature_dir ./text_features \
        --h5_dir /path/to/features_conch \
        --output_dir ./retrieval_results \
        --slide_folder /path/to/images \
        --file_type tif \
        --level 1 \
        --read_size 224 \
        --topk 30 \
        --neg_weight 0.5 \
        --mode combined

--mode 参数说明:
    pos      — 只按 positive 相似度排序
    neg      — 只按 negative 相似度排序
    combined — pos - neg_weight * neg (默认)
"""

import os
import argparse
import numpy as np
import cv2
import h5py
from PIL import Image, ImageDraw
import openslide


# ================================================================
# 加载特征
# ================================================================

def load_text_features(text_feature_dir):
    pos_path = os.path.join(text_feature_dir, 'pos_features.npy')
    neg_path = os.path.join(text_feature_dir, 'neg_features.npy')
    txt_path = os.path.join(text_feature_dir, 'queries.txt')

    if not os.path.exists(pos_path):
        raise FileNotFoundError(f"pos_features.npy not found in {text_feature_dir}\n"
                                f"Please run encode_text_queries.py first.")

    pos_feats = np.load(pos_path).astype(np.float32)  # (P, 512)
    neg_feats = np.load(neg_path).astype(np.float32) if os.path.exists(neg_path) else None

    print(f"📂 Loaded text features from: {text_feature_dir}")
    print(f"   pos_features: {pos_feats.shape}")
    if neg_feats is not None:
        print(f"   neg_features: {neg_feats.shape}")

    # 打印query内容
    if os.path.exists(txt_path):
        print(f"\n   Query记录:")
        with open(txt_path) as f:
            for line in f:
                print(f"   {line}", end='')
        print()

    return pos_feats, neg_feats


def load_patch_features(h5_dir):
    all_features      = []
    all_coords        = []
    all_slide_names   = []
    all_patch_indices = []

    h5_files = sorted([f for f in os.listdir(h5_dir) if f.endswith('.h5')])
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in {h5_dir}")

    print(f"\n📂 Loading patch features from {len(h5_files)} h5 files...")
    for h5_file in h5_files:
        slide_name = os.path.splitext(h5_file)[0]
        with h5py.File(os.path.join(h5_dir, h5_file), 'r') as f:
            feats   = f['features'][:]
            coords  = f['coordinates'][:]
            indices = f['patch_indices'][:]
        all_features.append(feats)
        all_coords.append(coords)
        all_patch_indices.append(indices)
        all_slide_names.extend([slide_name] * len(feats))
        print(f"   {slide_name}: {len(feats)} patches")

    features      = np.vstack(all_features).astype(np.float32)   # (N, 512)
    coords        = np.vstack(all_coords).astype(np.int64)        # (N, 2)
    patch_indices = np.concatenate(all_patch_indices).astype(np.int64)

    print(f"\n   Total patches: {features.shape[0]}")
    print(f"   Feature dim:   {features.shape[1]}")
    return features, coords, all_slide_names, patch_indices


# ================================================================
# 相似度计算
# ================================================================

def compute_scores(patch_features, pos_feats, neg_feats, neg_weight, mode):
    """
    mode:
      'pos'      → mean(pos_similarities)
      'neg'      → mean(neg_similarities)
      'combined' → mean(pos) - neg_weight * mean(neg)

    patch_features: (N, 512)
    pos_feats:      (P, 512) 或 None
    neg_feats:      (Q, 512) 或 None
    """
    pos_scores = (patch_features @ pos_feats.T).mean(axis=1) if pos_feats is not None else None
    neg_scores = (patch_features @ neg_feats.T).mean(axis=1) if neg_feats is not None else None

    print(f"\n📊 Score stats (mode={mode}):")

    if mode == 'pos':
        if pos_scores is None:
            raise ValueError("pos_features.npy is required for mode='pos'")
        print(f"   pos_score — mean: {pos_scores.mean():.4f}, max: {pos_scores.max():.4f}")
        return pos_scores

    elif mode == 'neg':
        if neg_scores is None:
            raise ValueError("neg_features.npy is required for mode='neg'")
        print(f"   neg_score — mean: {neg_scores.mean():.4f}, max: {neg_scores.max():.4f}")
        return neg_scores

    elif mode == 'combined':
        if pos_scores is None:
            raise ValueError("pos_features.npy is required for mode='combined'")
        if neg_scores is None:
            raise ValueError("neg_features.npy is required for mode='combined'")
        final = pos_scores - neg_weight * neg_scores
        print(f"   pos_score  — mean: {pos_scores.mean():.4f}, max: {pos_scores.max():.4f}")
        print(f"   neg_score  — mean: {neg_scores.mean():.4f}, max: {neg_scores.max():.4f}")
        print(f"   final      — mean: {final.mean():.4f},  max: {final.max():.4f}")
        return final

    else:
        raise ValueError(f"Unknown mode: {mode!r}. Choose from 'pos', 'neg', 'combined'.")


# ================================================================
# 读取patch图像
# ================================================================

def laplacian_sharpness(img_pil):
    """
    用Laplacian方差衡量图像清晰度
    越高越清晰，模糊图像通常 < 50
    """
    gray = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def extract_patch(slide_path, x, y, level, read_size, patch_size):
    """x, y 是level-0坐标（h5中存储的坐标）"""
    try:
        slide  = openslide.OpenSlide(slide_path)
        region = slide.read_region((x, y), level, (read_size, read_size)).convert('RGB')
        slide.close()
        if read_size != patch_size:
            region = region.resize((patch_size, patch_size), Image.BILINEAR)
        return region
    except Exception as e:
        print(f"  Warning: Could not read patch ({x},{y}): {e}")
        return None


# ================================================================
# 保存结果
# ================================================================

def save_results(results, output_dir, slide_folder, file_type,
                 level, read_size, patch_size, neg_weight, mode):
    os.makedirs(output_dir, exist_ok=True)

    # 文本结果
    txt_path = os.path.join(output_dir, 'retrieval_results.txt')
    with open(txt_path, 'w') as f:
        f.write(f"mode: {mode}\n")
        f.write(f"neg_weight: {neg_weight}\n\n")
        f.write(f"{'Rank':<6} {'Score':<10} {'Slide':<40} {'X':<12} {'Y':<12} {'PatchIdx'}\n")
        f.write("-" * 95 + "\n")
        for r in results:
            f.write(f"{r['rank']:<6} {r['score']:<10.4f} {r['slide_name']:<40} "
                    f"{r['x']:<12} {r['y']:<12} {r['patch_idx']}\n")
    print(f"\n📄 Text results: {txt_path}")

    if slide_folder is None:
        print("   (No slide_folder provided, skipping patch image saving)")
        return

    # Patch图像
    patch_dir = os.path.join(output_dir, 'patches')
    os.makedirs(patch_dir, exist_ok=True)
    print(f"🖼️  Saving {len(results)} patch images...")

    saved = []
    for r in results:
        slide_path = os.path.join(slide_folder, f"{r['slide_name']}.{file_type}")
        if not os.path.exists(slide_path):
            print(f"  Slide not found: {slide_path}")
            continue
        patch = extract_patch(slide_path, r['x'], r['y'], level, read_size, patch_size)
        if patch is None:
            continue
        sharpness = r.get('sharpness', -1)
        fname = (f"rank{r['rank']:03d}_score{r['score']:.3f}_sharp{sharpness:.0f}_"
                 f"{r['slide_name']}_x{r['x']}_y{r['y']}.png")
        patch.save(os.path.join(patch_dir, fname))
        saved.append((patch, r))

    print(f"  Saved: {len(saved)} patches")

    # Grid图
    if saved:
        _save_grid(saved, output_dir, patch_size, ncols=5)

    print(f"✅ Saved {len(saved)} patches.")


def _save_grid(patches_and_results, output_dir, patch_size, ncols=5):
    cell    = patch_size
    margin  = 4
    label_h = 26
    n       = len(patches_and_results)
    nrows   = (n + ncols - 1) // ncols
    grid    = Image.new('RGB',
                        (ncols*(cell+margin)+margin, nrows*(cell+label_h+margin)+margin),
                        color=(30, 30, 30))
    draw = ImageDraw.Draw(grid)

    for i, (img, r) in enumerate(patches_and_results):
        col = i % ncols
        row = i // ncols
        px  = margin + col * (cell + margin)
        py  = margin + row * (cell + label_h + margin)
        grid.paste(img.resize((cell, cell)), (px, py))
        label = f"#{r['rank']}  {r['score']:.3f}  {r['slide_name'][:12]}"
        draw.text((px+2, py+cell+3), label, fill=(210, 210, 80))

    grid_path = os.path.join(output_dir, 'retrieval_grid.png')
    grid.save(grid_path)
    print(f"🗂️  Grid image: {grid_path}")


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Retrieve patches using pre-computed text features",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--text_feature_dir', required=True,
                        help='Directory with pos_features.npy / neg_features.npy\n'
                             '(output of encode_text_queries.py)')
    parser.add_argument('--h5_dir',           required=True,
                        help='Directory containing patch .h5 feature files')
    parser.add_argument('--output_dir',       required=True,
                        help='Directory to save results\n'
                             '(results will be saved to <output_dir>/<mode>/)')
    parser.add_argument('--topk',             type=int,   default=30)
    parser.add_argument('--neg_weight',       type=float, default=0.5,
                        help='Weight for negative queries (default: 0.5)\n'
                             'Only used when --mode combined')
    parser.add_argument('--mode',             choices=['pos', 'neg', 'combined'],
                        default='combined',
                        help='Retrieval scoring mode (default: combined):\n'
                             '  pos      — rank by positive similarity only\n'
                             '  neg      — rank by negative similarity only\n'
                             '  combined — pos - neg_weight * neg')
    parser.add_argument('--sharpness_thresh', type=float, default=50,
                        help='Laplacian variance threshold for blur filtering.\n'
                             'Patches below this value are skipped (default: 50).\n'
                             'Set to 0 to disable blur filtering.')
    parser.add_argument('--slide_folder',     default=None,
                        help='(Optional) WSI folder — needed to save patch images')
    parser.add_argument('--file_type',        choices=['ndpi', 'svs', 'tif'], default='tif')
    parser.add_argument('--level',            type=int,   default=1,
                        help='WSI level used during feature extraction (default: 1)')
    parser.add_argument('--read_size',        type=int,   default=224)
    parser.add_argument('--patch_size',       type=int,   default=224)
    args = parser.parse_args()

    # 输出目录按mode自动分子目录，避免不同模式互相覆盖
    args.output_dir = os.path.join(args.output_dir, args.mode)
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🔧 Mode: {args.mode}")

    # 1. 加载文字特征
    pos_feats, neg_feats = load_text_features(args.text_feature_dir)

    # 校验mode与可用特征是否匹配
    if args.mode == 'neg' and neg_feats is None:
        raise FileNotFoundError("mode='neg' requires neg_features.npy, but it was not found.")
    if args.mode == 'combined' and neg_feats is None:
        raise FileNotFoundError("mode='combined' requires neg_features.npy, but it was not found.")

    # 2. 加载patch特征
    patch_features, patch_coords, slide_names, patch_indices = \
        load_patch_features(args.h5_dir)

    # 3. 计算所有patch的分数
    scores         = compute_scores(patch_features, pos_feats, neg_feats,
                                    args.neg_weight, args.mode)
    sorted_indices = np.argsort(scores)[::-1]  # 全部排好序

    # 4. 动态扩展：逐个取patch，清晰度过滤后凑够topk个
    target_k = args.topk
    max_scan  = min(target_k * 10, len(sorted_indices))
    results   = []
    rank      = 1

    print(f"\n🔍 Collecting top-{target_k} sharp patches "
          f"(sharpness_thresh={args.sharpness_thresh}, max_scan={max_scan})...")

    for idx in sorted_indices[:max_scan]:
        if len(results) >= target_k:
            break

        score      = float(scores[idx])
        slide_name = slide_names[idx]
        x          = int(patch_coords[idx][0])
        y          = int(patch_coords[idx][1])

        # 如果提供了slide_folder，做清晰度预检
        if args.slide_folder is not None and args.sharpness_thresh > 0:
            slide_path = os.path.join(args.slide_folder, f"{slide_name}.{args.file_type}")
            if os.path.exists(slide_path):
                patch = extract_patch(slide_path, x, y, args.level,
                                      args.read_size, args.patch_size)
                if patch is None:
                    continue
                sharpness = laplacian_sharpness(patch)
                if sharpness < args.sharpness_thresh:
                    print(f"  Skip blur: rank_candidate sharp={sharpness:.1f} "
                          f"{slide_name} ({x},{y})")
                    continue
            else:
                sharpness = -1  # slide找不到，不过滤
        else:
            sharpness = -1  # 不过滤

        results.append({
            'rank':       rank,
            'score':      score,
            'slide_name': slide_name,
            'x':          x,
            'y':          y,
            'patch_idx':  int(patch_indices[idx]),
            'sharpness':  sharpness,
        })
        rank += 1

    print(f"  ✅ Collected {len(results)} sharp patches "
          f"(scanned top-{max_scan} candidates)")

    # 5. 打印top-10
    print(f"\n{'='*75}")
    print(f"Top-{min(10, args.topk)} results  (mode={args.mode}, neg_weight={args.neg_weight})")
    print(f"{'='*75}")
    print(f"{'Rank':<6} {'Score':<10} {'Slide':<35} {'(x, y)'}")
    print("-" * 75)
    for r in results[:10]:
        print(f"{r['rank']:<6} {r['score']:<10.4f} {r['slide_name']:<35} ({r['x']}, {r['y']})")

    # 6. 保存
    save_results(
        results,
        output_dir=args.output_dir,
        slide_folder=args.slide_folder,
        file_type=args.file_type,
        level=args.level,
        read_size=args.read_size,
        patch_size=args.patch_size,
        neg_weight=args.neg_weight,
        mode=args.mode,
    )

    print(f"\n✅ All results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()