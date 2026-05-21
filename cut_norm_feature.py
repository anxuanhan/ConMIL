import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
import openslide
import argparse
import torch
import h5py
from tqdm import tqdm
from multiprocessing import Pool


from conch.open_clip_custom import create_model_from_pretrained


try:
    from wsi_normalizer import imread, MacenkoNormalizer
    HAS_MACENKO = True
except ImportError:
    HAS_MACENKO = False
    print("Warning: wsi-normalizer not installed. Macenko method will not be available.")

try:
    import torchstain
    HAS_TORCHSTAIN = True
except ImportError:
    HAS_TORCHSTAIN = False
    print("Warning: torchstain not installed. GPU Macenko will not be available.")


CONCH_MEAN = (0.48145466, 0.4578275,  0.40821073)
CONCH_STD  = (0.26862954, 0.26130258, 0.27577711)

class GPUMacenkoBatchNormalizer:
    def __init__(self, target_img_path, device, target_size=224):
        self.device = device
        self.target_size = int(target_size)

        target = cv2.imread(target_img_path)
        if target is None:
            raise FileNotFoundError(f"Cannot read reference image: {target_img_path}")
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        if target.shape[0] != self.target_size or target.shape[1] != self.target_size:
            target = cv2.resize(target, (self.target_size, self.target_size),
                                interpolation=cv2.INTER_AREA)

        target_tensor = torch.from_numpy(target).permute(2, 0, 1).float().to(device)
        self.normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')

        try:
            self.normalizer.fit(target_tensor)
            print("✅ GPU Macenko normalizer fitted successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to fit Macenko normalizer: {e}")

    def normalize_batch(self, batch_images):

        try:
            batch_tensor = torch.from_numpy(batch_images).permute(0, 3, 1, 2).float().to(self.device)
            normalized_list = []
            for i in range(batch_tensor.shape[0]):
                single_img = batch_tensor[i]
                result = self.normalizer.normalize(I=single_img, stains=False)
                norm_img = result[0] if isinstance(result, tuple) else result
                if norm_img.shape[0] != 3:
                    norm_img = norm_img.permute(2, 0, 1)
                normalized_list.append(norm_img)
            normalized_tensor = torch.stack(normalized_list, dim=0)
            return normalized_tensor.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        except Exception as e:
            print(f"Warning: GPU batch normalization failed: {e}, returning originals")
            return batch_images



def get_conch_model(checkpoint_path):

    print(f"📦 Loading CONCH model from: {checkpoint_path}")
    model, preprocess = create_model_from_pretrained(
        "conch_ViT-B-16",
        checkpoint_path=checkpoint_path
    )
    model.eval()
    print("✅ CONCH model loaded successfully")
    print("   Feature dimension = 512 (ViT-B-16 + projection head)")
    for t in preprocess.transforms:
        if hasattr(t, 'mean'):
            print(f"   preprocess Normalize → mean={t.mean}, std={t.std}")
    return model, preprocess



def batch_preprocess_numpy(normalized_batch: np.ndarray,
                            device: torch.device) -> torch.Tensor:

    # (B, H, W, 3) → (B, 3, H, W), float32, 0-1
    t = (torch.from_numpy(normalized_batch)
         .permute(0, 3, 1, 2)
         .float()
         .div_(255.0))

    mean = torch.tensor(CONCH_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std  = torch.tensor(CONCH_STD,  dtype=torch.float32).view(1, 3, 1, 1)
    t = (t - mean) / std

    return t.to(device, non_blocking=True)



def transform_to_lab(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    return lab.astype(np.float64)


def reinhard_normalize(source_img, target_stats):
    target_mean, target_std = target_stats
    source_lab = transform_to_lab(source_img)
    L, A, B = cv2.split(source_lab)
    src_mean = np.array([L.mean(), A.mean(), B.mean()])
    src_std  = np.array([L.std(),  A.std(),  B.std()])
    L = ((L - src_mean[0]) * (target_std[0] / src_std[0])) + target_mean[0]
    A = ((A - src_mean[1]) * (target_std[1] / src_std[1])) + target_mean[1]
    B = ((B - src_mean[2]) * (target_std[2] / src_std[2])) + target_mean[2]
    L = np.clip(L, 0, 255)
    A = np.clip(A, 0, 255)
    B = np.clip(B, 0, 255)
    return cv2.cvtColor(cv2.merge([L, A, B]).astype(np.uint8), cv2.COLOR_LAB2RGB)


def calculate_target_stats(target_img_path):
    target_img = cv2.imread(target_img_path)
    if target_img is None:
        raise FileNotFoundError(f"Cannot read reference image: {target_img_path}")
    target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    target_lab = transform_to_lab(target_img)
    L, A, B = cv2.split(target_lab)
    return (np.array([L.mean(), A.mean(), B.mean()]),
            np.array([L.std(),  A.std(),  B.std()]))


def prepare_normalizer(method, ref_img_path, device, patch_size):
    if method == 'macenko':
        if HAS_TORCHSTAIN and torch.cuda.is_available():
            print("🚀 Using GPU batch Macenko normalizer (FAST!)")
            return GPUMacenkoBatchNormalizer(ref_img_path, device, target_size=patch_size), None
        elif HAS_MACENKO:
            print("⚠️  Falling back to CPU Macenko (SLOW)")
            normalizer = MacenkoNormalizer()
            normalizer.fit(imread(ref_img_path))
            return normalizer, None
        else:
            raise ImportError(
                "Macenko requires torchstain (GPU) or wsi-normalizer (CPU)")
    else:  # reinhard
        print(f"Calculating Reinhard target stats from: {os.path.basename(ref_img_path)}")
        stats = calculate_target_stats(ref_img_path)
        print(f"Target stats — Mean: {stats[0]}, Std: {stats[1]}")
        return None, stats


def normalize_patch(patch_rgb, method, macenko_normalizer, target_stats):
    try:
        if method == 'macenko':
            return macenko_normalizer.transform(patch_rgb)
        else:
            return reinhard_normalize(patch_rgb, target_stats)
    except Exception as e:
        print(f"Warning: Normalization failed: {e}")
        return patch_rgb


def is_blank_patch(patch, threshold=0.8):
    gray = cv2.cvtColor(np.array(patch), cv2.COLOR_RGB2GRAY)
    return np.sum(gray > 230) / gray.size > threshold



global_slide      = None
global_slide_path = None


def process_batch_worker(batch_data):

    global global_slide, global_slide_path

    (batch_idx, batch_positions, slide_path_local,
     read_size_val_local, patch_size_local, blank_threshold_local, level_local) = batch_data

    try:
        if global_slide is None or global_slide_path != slide_path_local:
            if global_slide is not None:
                global_slide.close()
            global_slide      = openslide.OpenSlide(slide_path_local)
            global_slide_path = slide_path_local

        slide_local = global_slide
        batch_images    = []
        batch_coords    = []
        batch_indices   = []
        batch_kept_pos  = []
        batch_kept_mask = []
        batch_filt_pos  = []

        for pos in batch_positions:
            try:
                region = slide_local.read_region(
                    (pos['x'], pos['y']), level_local,
                    (read_size_val_local, read_size_val_local)
                ).convert("RGB").resize(
                    (patch_size_local, patch_size_local), Image.BILINEAR
                )

                if is_blank_patch(region, blank_threshold_local):
                    batch_filt_pos.append((pos['x'], pos['y']))
                    continue

                batch_images.append(np.array(region))
                batch_coords.append([pos['x'], pos['y']])
                batch_indices.append(pos['index'])
                batch_kept_pos.append((pos['x'], pos['y']))
                batch_kept_mask.append((pos['x_mask'], pos['y_mask']))

            except Exception:
                batch_filt_pos.append((pos['x'], pos['y']))

        return {
            'batch_idx':      batch_idx,
            'images':         batch_images,
            'coords':         batch_coords,
            'indices':        batch_indices,
            'kept_pos':       batch_kept_pos,
            'kept_pos_mask':  batch_kept_mask,
            'filtered_pos':   batch_filt_pos,
            'total_in_batch': len(batch_positions),
        }
    except Exception as e:
        print(f"[Worker] CRITICAL ERROR in Batch {batch_idx}: {e}", flush=True)
        return None



def process_single_slide(slide_path, mask_path, model, device,
                         method, macenko_normalizer, target_stats,
                         patch_size, read_size, blank_threshold, batch_size,
                         output_folder, checkpoint_interval=20, save_filter_log=True,
                         grid_output_dir=None, num_workers=6, ref_img_path=None,
                         save_raw_patches=None, save_normalized_patches=None,
                         level=0):

    slide_name      = os.path.splitext(os.path.basename(slide_path))[0]
    h5_path         = os.path.join(output_folder, f"{slide_name}.h5")
    filter_log_path = os.path.join(output_folder, f"{slide_name}_filter_log.txt")

    raw_patch_dir        = None
    normalized_patch_dir = None
    if save_raw_patches:
        raw_patch_dir = os.path.join(save_raw_patches, slide_name)
        os.makedirs(raw_patch_dir, exist_ok=True)
        print(f"📁 Raw patches → {raw_patch_dir}")
    if save_normalized_patches:
        normalized_patch_dir = os.path.join(save_normalized_patches, slide_name)
        os.makedirs(normalized_patch_dir, exist_ok=True)
        print(f"📁 Normalized patches → {normalized_patch_dir}")

    print(f"\n{'='*60}")
    print(f"Processing slide: {slide_name}")
    print(f"{'='*60}")

    existing_data, processed_indices = load_existing_h5(h5_path)

    filtered_patches         = []
    kept_patch_positions     = []
    filtered_patch_positions = []

    try:
        slide = openslide.OpenSlide(slide_path)
    except openslide.OpenSlideError as e:
        print(f"Error opening slide: {e}")
        return None


    num_levels = slide.level_count
    if level >= num_levels:
        print(f"⚠️  Requested level={level} but slide only has {num_levels} levels. "
              f"Falling back to level {num_levels - 1}.")
        level = num_levels - 1

    level_dimensions = slide.level_dimensions[level]
    level_downsample = slide.level_downsamples[level]
    print(f"WSI level: {level}, dimensions: {level_dimensions}, "
          f"downsample: {level_downsample:.1f}x")


    try:
        import tifffile
        with tifffile.TiffFile(mask_path) as tif:
            series = tif.series[0]
            levels = getattr(series, 'levels', [series])
            arr = levels[len(levels) - 1].asarray()
        if arr.ndim == 3:
            arr = arr[..., 0]
        mask = (arr > 0).astype(np.uint8)
        mask = cv2.resize(mask, (level_dimensions[0], level_dimensions[1]),
                          interpolation=cv2.INTER_NEAREST)
        print(f"   Mask loaded via tifffile: {mask.shape[1]}x{mask.shape[0]}, "
              f"tissue pixels: {np.sum(mask > 0)}")
    except Exception as e:
        print(f"Error: Cannot read mask at {mask_path}: {e}")
        slide.close()
        return None


    mpp_x = slide.properties.get('openslide.mpp-x')
    mpp_y = slide.properties.get('openslide.mpp-y')
    if mpp_x and mpp_y:
        avg_mpp       = (float(mpp_x) + float(mpp_y)) / 2
        magnification = "40x" if avg_mpp < 0.3 else ("20x" if avg_mpp < 0.6 else "Unknown")
    else:
        avg_mpp       = None
        magnification = "Unknown"

    print(f"Magnification: {magnification} (avg_mpp: {avg_mpp if avg_mpp else 'N/A'})")

    read_size_val = (512 if magnification == "40x" else patch_size) if read_size is None else read_size
    step          = read_size_val
    mask_shape    = mask.shape
    print(f"Parameters: level={level}, read_size={read_size_val}, "
          f"patch_size={patch_size}, step={step}")

 
    print("\nStep 1/2: Scanning for valid patch positions...")
    valid_positions = []
    patch_counter   = 0
    total_positions = (mask.shape[0] // step) * (mask.shape[1] // step)

    with tqdm(total=total_positions, desc="Scanning positions", unit="pos") as pbar:
        for y in range(0, mask.shape[0], step):
            for x in range(0, mask.shape[1], step):
                pbar.update(1)
                if np.sum(mask[y:y + step, x:x + step]) == 0:
                    continue   
                if patch_counter not in processed_indices:
                    valid_positions.append({
                        'x':      int(round(x * level_downsample)),
                        'y':      int(round(y * level_downsample)),
                        'x_mask': x,
                        'y_mask': y,
                        'index':  patch_counter,
                        'level':  level,
                    })
                patch_counter += 1

    total_candidates = patch_counter
    already_processed = len(processed_indices)
    to_process        = len(valid_positions)

    print(f"Total candidate positions : {total_candidates}")
    print(f"Already processed         : {already_processed}")
    print(f"To process                : {to_process}")

    if to_process == 0:
        print("✅ All patches already processed!")
        slide.close()
        if grid_output_dir is not None and existing_data is not None:
            os.makedirs(grid_output_dir, exist_ok=True)
            save_grid_visualization(
                slide_path, mask_shape, step,
                [(int(c[0]), int(c[1])) for c in existing_data['coordinates']],
                [], os.path.join(grid_output_dir, f"{slide_name}_grid.png")
            )
        return {'slide_name': slide_name, 'skipped': True}


    print(f"\nStep 2/2: Processing remaining patches...")
    print(f"Batch size: {batch_size}, Checkpoint every {checkpoint_interval} batches")
    print(f"Workers: {num_workers}")

    all_features     = []
    all_coords       = []
    all_indices      = []
    processed_count  = 0
    num_batches      = (to_process + batch_size - 1) // batch_size
    coord_to_index   = {(vp['x'], vp['y']): vp['index'] for vp in valid_positions}

    conch_mean_t = torch.tensor(CONCH_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    conch_std_t  = torch.tensor(CONCH_STD,  dtype=torch.float32).view(1, 3, 1, 1)

    batch_tasks = [
        (batch_idx,
         valid_positions[batch_idx * batch_size: min((batch_idx + 1) * batch_size, to_process)],
         slide_path, read_size_val, patch_size, blank_threshold, level)
        for batch_idx in range(num_batches)
    ]

    with Pool(processes=num_workers) as pool:
        pending_results       = {}
        next_batch_to_process = 0

        with tqdm(total=to_process, desc="Processing patches", unit="patch") as pbar:
            for batch_result in pool.imap_unordered(process_batch_worker, batch_tasks, chunksize=1):
                if batch_result is None:
                    continue

                pending_results[batch_result['batch_idx']] = batch_result

                while next_batch_to_process in pending_results:
                    result = pending_results.pop(next_batch_to_process)
                    pbar.update(result['total_in_batch'])

                    kept_patch_positions.extend(result['kept_pos_mask'])
                    filtered_patch_positions.extend(result['filtered_pos'])

                    if save_filter_log:
                        for fpos in result['filtered_pos']:
                            idx = coord_to_index.get(fpos)
                            if idx is not None:
                                filtered_patches.append(
                                    {'index': idx, 'x': fpos[0], 'y': fpos[1],
                                     'reason': 'blank or error'})

                    if len(result['images']) > 0:
                        batch_images_np = np.stack(result['images'], axis=0)  # (B,H,W,3) uint8

                
                        if raw_patch_dir is not None:
                            for img, coord, pidx in zip(batch_images_np,
                                                        result['coords'],
                                                        result['indices']):
                                Image.fromarray(img).save(
                                    os.path.join(raw_patch_dir,
                                                 f"patch_{pidx:05d}_x{coord[0]}_y{coord[1]}.png"))

                     
                        if method != 'none':
                            if isinstance(macenko_normalizer, GPUMacenkoBatchNormalizer):
                                normalized_batch = macenko_normalizer.normalize_batch(batch_images_np)
                            elif macenko_normalizer is not None:
                                normalized_batch = np.stack([
                                    normalize_patch(img, method, macenko_normalizer, target_stats)
                                    for img in batch_images_np], axis=0)
                            elif target_stats is not None:
                                normalized_batch = np.stack([
                                    reinhard_normalize(img, target_stats)
                                    for img in batch_images_np], axis=0)
                            else:
                                normalized_batch = batch_images_np
                        else:
                            normalized_batch = batch_images_np

                
                        if normalized_patch_dir is not None:
                            for img, coord, pidx in zip(normalized_batch,
                                                        result['coords'],
                                                        result['indices']):
                                Image.fromarray(img).save(
                                    os.path.join(normalized_patch_dir,
                                                 f"patch_{pidx:05d}_x{coord[0]}_y{coord[1]}.png"))

                       
                        batch_tensor = (
                            torch.from_numpy(normalized_batch)   # (B,H,W,3) uint8
                            .permute(0, 3, 1, 2)                 # (B,3,H,W)
                            .float()
                            .div_(255.0)
                        )
                        batch_tensor = (batch_tensor - conch_mean_t) / conch_std_t
                        batch_tensor = batch_tensor.to(device, non_blocking=True)

                        with torch.inference_mode():
                            batch_features = model.encode_image(
                                batch_tensor,
                                proj_contrast=True,  
                                normalize=True       
                            )

                        all_features.append(batch_features.cpu().numpy())  # (B, 512)
                        all_coords.extend(result['coords'])
                        all_indices.extend(result['indices'])
                        processed_count += len(result['images'])

 
                    if ((next_batch_to_process + 1) % checkpoint_interval == 0
                            and len(all_features) > 0):
                        new_data = {
                            'features':      np.vstack(all_features),
                            'coordinates':   np.array(all_coords,   dtype=np.int64),
                            'patch_indices': np.array(all_indices,  dtype=np.int64),
                        }
                        append_to_h5(h5_path, new_data, existing_data)
                        print(f"\n💾 Checkpoint: {len(new_data['features'])} new patches "
                              f"(total so far: {already_processed + processed_count})")
                        all_features = []
                        all_coords   = []
                        all_indices  = []

                    next_batch_to_process += 1


    if save_filter_log and filtered_patches:
        with open(filter_log_path, 'w') as f:
            f.write(f"Slide: {slide_name}\n")
            f.write(f"Total candidates: {total_candidates}\n")
            f.write(f"Filtered out: {len(filtered_patches)}\n")
            f.write(f"Valid patches: {processed_count}\n")
            f.write(f"Blank threshold: {blank_threshold}\n\n")
            f.write("Index, X, Y, Reason\n")
            for fp in filtered_patches:
                f.write(f"{fp['index']}, {fp['x']}, {fp['y']}, {fp['reason']}\n")
        print(f"📋 Filter log → {filter_log_path}")


    if all_features:
        new_data = {
            'features':      np.vstack(all_features),
            'coordinates':   np.array(all_coords,  dtype=np.int64),
            'patch_indices': np.array(all_indices, dtype=np.int64),
        }
        append_to_h5(h5_path, new_data, existing_data)
        print(f"\n💾 Final save: {len(new_data['features'])} patches")

    if not os.path.exists(h5_path):
        print(f"\n⚠️  No patches saved for {slide_name} "
              f"(all {total_candidates} candidates filtered as blank).")
        print(f"   Try lowering --blank_threshold (current: {blank_threshold})")
        slide.close()
        return None

    with h5py.File(h5_path, 'r') as f:
        total_patches = len(f['patch_indices'])

    print(f"\n✅ Slide completed: {total_patches} total patches "
          f"({already_processed} previous + {processed_count} new)")

    if grid_output_dir is not None:
        os.makedirs(grid_output_dir, exist_ok=True)
        if existing_data is not None:
            for coord in existing_data['coordinates']:
                kept_patch_positions.append((int(coord[0]), int(coord[1])))
        print("\n📊 Generating grid visualization...")
        save_grid_visualization(
            slide_path, mask_shape, step,
            kept_patch_positions, filtered_patch_positions,
            os.path.join(grid_output_dir, f"{slide_name}_grid.png")
        )

    slide.close()
    return {'slide_name': slide_name, 'total_patches': total_patches}



def save_grid_visualization(slide_path, mask_shape, step,
                             kept_positions, filtered_positions, output_path):
    print("   Creating grid visualization...")
    try:
        max_thumb = 1000
        padding   = 50
        scale     = min(max_thumb / mask_shape[1], max_thumb / mask_shape[0], 1.0)
        thumb_w   = int(mask_shape[1] * scale)
        thumb_h   = int(mask_shape[0] * scale)

        slide     = openslide.OpenSlide(slide_path)
        thumbnail = slide.get_thumbnail((thumb_w, thumb_h)).convert('RGB')
        slide.close()

        canvas = Image.new('RGB', (thumb_w + 2 * padding, thumb_h + 2 * padding),
                           color=(255, 255, 255))
        canvas.paste(thumbnail, (padding, padding))
        draw = ImageDraw.Draw(canvas, 'RGBA')

        for x, y in kept_positions:
            x1 = int(x * scale) + padding
            y1 = int(y * scale) + padding
            x2 = int((x + step) * scale) + padding
            y2 = int((y + step) * scale) + padding
            for line in [((x1, y1), (x2, y1)), ((x1, y2), (x2, y2)),
                         ((x1, y1), (x1, y2)), ((x2, y1), (x2, y2))]:
                draw.line(line, fill=(0, 0, 0, 255), width=2)

        canvas.save(output_path)
        print("   ✅ Grid visualization saved!")
    except Exception as e:
        print(f"   ❌ Error creating visualization: {e}")
        raise


def load_existing_h5(h5_path):
    if not os.path.exists(h5_path):
        return None, set()
    try:
        with h5py.File(h5_path, 'r') as f:
            existing_indices = set(f['patch_indices'][:].tolist())
            existing_data = {k: f[k][:] for k in ('features', 'coordinates', 'patch_indices')}
        print(f"   📂 Found existing h5 with {len(existing_indices)} patches")
        return existing_data, existing_indices
    except Exception as e:
        print(f"   ⚠️  Error reading existing h5: {e}, starting fresh")
        return None, set()


def append_to_h5(h5_path, new_data, existing_data=None):
    """追加或创建 h5，特征维度 512（CONCH ViT-B-16）"""
    feat_dim = new_data['features'].shape[1]  # 512

    if not os.path.exists(h5_path):
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('features',      data=new_data['features'],
                             compression='gzip', maxshape=(None, feat_dim), chunks=True)
            f.create_dataset('coordinates',   data=new_data['coordinates'],
                             compression='gzip', maxshape=(None, 2), chunks=True)
            f.create_dataset('patch_indices', data=new_data['patch_indices'],
                             compression='gzip', maxshape=(None,), chunks=True)
        return

    try:
        with h5py.File(h5_path, 'a') as f:
            cur = f['features'].shape[0]
            new = cur + new_data['features'].shape[0]
            f['features'].resize((new, feat_dim))
            f['coordinates'].resize((new, 2))
            f['patch_indices'].resize((new,))
            f['features'][cur:]      = new_data['features']
            f['coordinates'][cur:]   = new_data['coordinates']
            f['patch_indices'][cur:] = new_data['patch_indices']
    except (RuntimeError, OSError, ValueError) as e:
        print(f"\n⚠️  Append failed ({e}). Rebuilding h5...")
        try:
            with h5py.File(h5_path, 'r') as f:
                cur_feat    = f['features'][:]
                cur_coords  = f['coordinates'][:]
                cur_indices = f['patch_indices'][:]
            combined_feat    = np.vstack([cur_feat,    new_data['features']])
            combined_coords  = np.vstack([cur_coords,  new_data['coordinates']])
            combined_indices = np.concatenate([cur_indices, new_data['patch_indices']])
        except Exception as re:
            print(f"❌ Read failed ({re}). Overwriting with new data only.")
            combined_feat    = new_data['features']
            combined_coords  = new_data['coordinates']
            combined_indices = new_data['patch_indices']

        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('features',      data=combined_feat,
                             compression='gzip', maxshape=(None, feat_dim), chunks=True)
            f.create_dataset('coordinates',   data=combined_coords,
                             compression='gzip', maxshape=(None, 2), chunks=True)
            f.create_dataset('patch_indices', data=combined_indices,
                             compression='gzip', maxshape=(None,), chunks=True)
        print("✅ Rebuild completed.")


def main():
    parser = argparse.ArgumentParser(
        description="WSI feature extraction with CONCH (optimized)",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('-input_folder',  required=True, help='Folder containing WSI files')
    parser.add_argument('-mask_folder',   required=True, help='Folder containing binary masks (.tif)')
    parser.add_argument('-output_folder', required=True, help='Output folder for h5 feature files')

    parser.add_argument('-patch_size',       type=int,   default=224,
                        help='Patch size fed into CONCH (default: 224)')
    parser.add_argument('-read_size',        type=int,   default=None,
                        help='Read size from WSI before resizing (default: auto)')
    parser.add_argument('-level',            type=int,   default=0,
                        help='WSI pyramid level (default: 0 = highest resolution)')
    parser.add_argument('-blank_threshold',  type=float, default=0.9,
                        help='Blank patch filtering threshold (default: 0.9)')
    parser.add_argument('-file_type',        choices=['ndpi', 'svs', 'tif'], default='svs')

    parser.add_argument('--checkpoint_path', required=True,
                        help='Path to CONCH pytorch_model.bin')

    parser.add_argument('-r', '--ref_img_path', required=True,
                        help='Reference image for color normalization')
    parser.add_argument('-m', '--method',
                        choices=['reinhard', 'macenko', 'none'], default='macenko')

    parser.add_argument('--batch_size',          type=int, default=64)
    parser.add_argument('--checkpoint_interval', type=int, default=20,
                        help='Save checkpoint every N batches (default: 20)')
    parser.add_argument('--num_workers',         type=int, default=6)
    parser.add_argument('--save_filter_log',     action='store_true')
    parser.add_argument('--grid_output_dir',     type=str, default=None)
    parser.add_argument('--save_raw_patches',        type=str, default=None)
    parser.add_argument('--save_normalized_patches', type=str, default=None)

    args = parser.parse_args()


    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("🖥️  CPU mode")


    model, _ = get_conch_model(args.checkpoint_path)
    model = model.to(device)

  
    print(f"\n🎨 Preparing {args.method} normalizer...")
    macenko_normalizer, target_stats = prepare_normalizer(
        args.method, args.ref_img_path, device, args.patch_size)
    print("✅ Normalizer ready")

    ext       = f".{args.file_type}"
    wsi_files = sorted(f for f in os.listdir(args.input_folder) if f.endswith(ext))

    if not wsi_files:
        print(f"❌ No {ext} files found in {args.input_folder}")
        return

    print(f"\n📂 Found {len(wsi_files)} {args.file_type.upper()} files")
    os.makedirs(args.output_folder, exist_ok=True)

    print("\n🚀 Starting processing...")
    success_count = 0
    failed_slides = []

    for idx, wsi_file in enumerate(wsi_files, 1):
        base      = os.path.splitext(wsi_file)[0]
        mask_name = f"{base}_tissue.tif"
        slide_path = os.path.join(args.input_folder,  wsi_file)
        mask_path  = os.path.join(args.mask_folder,   mask_name)

        if not os.path.exists(mask_path):
            print(f"\n[{idx}/{len(wsi_files)}] ⚠️  Mask not found: {mask_name}, skipping.")
            failed_slides.append(base)
            continue

        print(f"\n[{idx}/{len(wsi_files)}] Processing: {wsi_file}")

        result = process_single_slide(
            slide_path=slide_path,
            mask_path=mask_path,
            model=model,
            device=device,
            method=args.method,
            macenko_normalizer=macenko_normalizer,
            target_stats=target_stats,
            patch_size=args.patch_size,
            read_size=args.read_size,
            blank_threshold=args.blank_threshold,
            batch_size=args.batch_size,
            output_folder=args.output_folder,
            checkpoint_interval=args.checkpoint_interval,
            save_filter_log=args.save_filter_log,
            grid_output_dir=args.grid_output_dir,
            num_workers=args.num_workers,
            ref_img_path=args.ref_img_path,
            save_raw_patches=args.save_raw_patches,
            save_normalized_patches=args.save_normalized_patches,
            level=args.level,
        )

        if result is not None:
            success_count += 1
        else:
            print(f"❌ Failed: {wsi_file}")
            failed_slides.append(base)

    print("\n" + "=" * 60)
    print("🎉 PROCESSING COMPLETED!")
    print("=" * 60)
    print(f"✅ Success: {success_count}/{len(wsi_files)}")
    if failed_slides:
        print(f"❌ Failed : {', '.join(failed_slides)}")
    print(f"📁 Output : {args.output_folder}")


if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()