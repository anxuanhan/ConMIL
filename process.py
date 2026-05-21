import argparse
import openslide
import cv2
from PIL import Image
import numpy as np
import os

def process_ndpi_files(input_folder, min_area, min_hole_area, out_dir="."):
    ndpi_files = [f for f in os.listdir(input_folder) if f.endswith(".ndpi")]

    for idx, ndpi_file in enumerate(ndpi_files, start=1):
        # 打开 ndpi 文件
        slide_path = os.path.join(input_folder, ndpi_file)
        slide = openslide.OpenSlide(slide_path)

        # 获取文件前缀名
        file_prefix = os.path.splitext(ndpi_file)[0]

        # 获取 Level 0 的尺寸
        level_0_dimensions = slide.level_dimensions[0]

        # 获取物理分辨率（每像素微米）
        mpp_x = slide.properties.get('openslide.mpp-x')
        mpp_y = slide.properties.get('openslide.mpp-y')
        if mpp_x and mpp_y:
            avg_mpp = (float(mpp_x) + float(mpp_y)) / 2
            magnification = "40x" if avg_mpp < 0.3 else "20x" if avg_mpp < 0.6 else "Unknown"
        else:
            magnification = "Unknown"

        # 创建保存图片的文件夹（由 out_dir 控制）
        origin_dir = os.path.join(out_dir, "origin_pic")
        circle_dir = os.path.join(out_dir, "circle_pic")
        mask_dir = os.path.join(out_dir, "mask_pic")
        os.makedirs(origin_dir, exist_ok=True)
        os.makedirs(circle_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        # 选择检测 level
        detection_level = min(3, slide.level_count - 1)
        level_dimensions = slide.level_dimensions[detection_level]

        # 从选定的 level 读取整个图像用于检测
        detection_img = slide.read_region((0, 0), detection_level, level_dimensions)
        detection_img = detection_img.convert('RGB')
        detection_array = np.array(detection_img)
        
        # 保存原始图像用于origin_pic
        detection_array_original = detection_array.copy()

        # 转换为灰度图
        gray = cv2.cvtColor(detection_array, cv2.COLOR_RGB2GRAY)

        # 使用 Otsu's 二值化方法
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # ===== 恢复原始参数：形态学操作 =====
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
        binary_opened = cv2.morphologyEx(binary_closed, cv2.MORPH_OPEN, kernel, iterations=2)

        # 查找层级轮廓
        contours, hierarchy = cv2.findContours(binary_opened, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        # 根据检测 level 的分辨率调整面积阈值
        downsample_factor = slide.level_downsamples[detection_level]
        adjusted_min_area = min_area / (downsample_factor ** 2)
        adjusted_min_hole_area = min_hole_area / (downsample_factor ** 2)

        outer_contours = []
        hole_contours = []

        if hierarchy is not None:
            hierarchy = hierarchy[0]
            for i, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                if hierarchy[i][3] == -1:  # 外轮廓
                    if area > adjusted_min_area:
                        outer_contours.append(cnt)
                else:  # 内部孔洞
                    if area > adjusted_min_hole_area:
                        hole_contours.append(cnt)

        # 在检测分辨率的图像上绘制轮廓
        result_detection = detection_array.copy()
        result_detection_bgr = cv2.cvtColor(result_detection, cv2.COLOR_RGB2BGR)

        # 绘制轮廓
        cv2.drawContours(result_detection_bgr, outer_contours, -1, (0, 255, 0), 3)
        cv2.drawContours(result_detection_bgr, hole_contours, -1, (255, 0, 0), 3)

        # 转换回 RGB
        result_detection_rgb = cv2.cvtColor(result_detection_bgr, cv2.COLOR_BGR2RGB)

        # 缩放到 800x800（保持长宽比）
        h, w = result_detection_rgb.shape[:2]
        scale = min(800 / w, 800 / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        result_resized = cv2.resize(result_detection_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 创建 800x800 的白色背景
        result_final = np.ones((800, 800, 3), dtype=np.uint8) * 240

        # 计算居中位置
        y_offset = (800 - new_h) // 2
        x_offset = (800 - new_w) // 2

        # 将缩放后的图像放到中心
        result_final[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = result_resized

        # 保存带轮廓的图像
        result_pil = Image.fromarray(result_final)
        result_pil.save(os.path.join(circle_dir, f"{file_prefix}_tissue_contour_800x800.png"))

        # 同样的方法处理原始缩略图（使用未处理的原始图像）
        thumbnail_resized = cv2.resize(detection_array_original, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        thumbnail_final = np.ones((800, 800, 3), dtype=np.uint8) * 240
        thumbnail_final[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = thumbnail_resized
        thumbnail_pil = Image.fromarray(thumbnail_final)
        thumbnail_pil.save(os.path.join(origin_dir, f"{file_prefix}_thumbnail_800x800.png"))

        # 保存 mask
        # 创建更新的mask：组织区域为白色，孔洞为黑色
        mask_with_holes = np.zeros_like(binary_opened)  # 初始化一个全黑的掩膜

        # 将外轮廓填充为白色（255）
        cv2.drawContours(mask_with_holes, outer_contours, -1, 255, -1)

        # 将孔洞区域填充为黑色（0）
        cv2.drawContours(mask_with_holes, hole_contours, -1, 0, -1)

        # 保存二值化图像到 mask_pic 文件夹
        binary_pil = Image.fromarray(mask_with_holes)
        binary_pil.save(os.path.join(mask_dir, f"{file_prefix}_binary_mask_800x800.png"))

        # 打印信息
        print(f"Processing file {idx}/{len(ndpi_files)}: {ndpi_file}\n")
        print(f"Level 0 dimensions: {level_0_dimensions}\n")
        print(f"Detection level {detection_level} dimensions: {level_dimensions}\n")
        print(f"Downsample factor: {downsample_factor:.2f}\n")
        print(f"Estimated magnification: {magnification}\n")
        print(f"Found {len(outer_contours)} tissue regions (green)\n")
        print(f"Found {len(hole_contours)} holes (blue)\n")
        print("===========================================")

        slide.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ndpi/svs files and generate masks and contours.")
    parser.add_argument("-pic_path", required=True, help="Path to the folder containing ndpi/svs files.")
    parser.add_argument("-min_area", type=float, default=1000, help="Minimum area for tissue regions.")
    parser.add_argument("-min_hole", type=float, default=60000, help="Minimum area for holes.")
    parser.add_argument("-file_type", choices=["ndpi", "svs"], default="ndpi", help="Type of files to process: ndpi or svs.")
    parser.add_argument(
        "-out_dir",
        type=str,
        default=".",
        help="Output root directory (will create origin_pic/circle_pic/mask_pic under it; default: current dir).",
    )

    args = parser.parse_args()

    # 支持 ndpi 和 svs 文件
    def process_files(input_folder, min_area, min_hole_area, file_type, out_dir):
        if file_type == "ndpi":
            file_ext = ".ndpi"
        else:
            file_ext = ".svs"
        files = [f for f in os.listdir(input_folder) if f.endswith(file_ext)]
        for idx, file in enumerate(files, start=1):
            slide_path = os.path.join(input_folder, file)
            slide = openslide.OpenSlide(slide_path)
            file_prefix = os.path.splitext(file)[0]
            level_0_dimensions = slide.level_dimensions[0]
            mpp_x = slide.properties.get('openslide.mpp-x')
            mpp_y = slide.properties.get('openslide.mpp-y')
            if mpp_x and mpp_y:
                avg_mpp = (float(mpp_x) + float(mpp_y)) / 2
                magnification = "40x" if avg_mpp < 0.3 else "20x" if avg_mpp < 0.6 else "Unknown"
            else:
                magnification = "Unknown"
            origin_dir = os.path.join(out_dir, "origin_pic")
            circle_dir = os.path.join(out_dir, "circle_pic")
            mask_dir = os.path.join(out_dir, "mask_pic")
            os.makedirs(origin_dir, exist_ok=True)
            os.makedirs(circle_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)
            detection_level = min(3, slide.level_count - 1)
            level_dimensions = slide.level_dimensions[detection_level]
            detection_img = slide.read_region((0, 0), detection_level, level_dimensions)
            detection_img = detection_img.convert('RGB')
            detection_array = np.array(detection_img)
            
            # 保存原始图像用于origin_pic
            detection_array_original = detection_array.copy()
            
            gray = cv2.cvtColor(detection_array, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
            binary_opened = cv2.morphologyEx(binary_closed, cv2.MORPH_OPEN, kernel, iterations=2)
            contours, hierarchy = cv2.findContours(binary_opened, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            downsample_factor = slide.level_downsamples[detection_level]
            adjusted_min_area = min_area / (downsample_factor ** 2)
            adjusted_min_hole_area = min_hole_area / (downsample_factor ** 2)
            outer_contours = []
            hole_contours = []
            if hierarchy is not None:
                hierarchy = hierarchy[0]
                for i, cnt in enumerate(contours):
                    area = cv2.contourArea(cnt)
                    if hierarchy[i][3] == -1:
                        if area > adjusted_min_area:
                            outer_contours.append(cnt)
                    else:
                        if area > adjusted_min_hole_area:
                            hole_contours.append(cnt)
                # 限制最多只保留10个面积最大的孔洞
                if len(hole_contours) > 10:
                    hole_contours = sorted(hole_contours, key=cv2.contourArea, reverse=True)[:10]
                    print(f"   Limited to top 10 largest holes")
                # 过滤组织区域，只保留面积大于最大组织区域1/5的部分
                if len(outer_contours) > 0:
                    areas = [cv2.contourArea(cnt) for cnt in outer_contours]
                    max_area = max(areas)
                    min_fragment_area = max_area / 5
                    filtered_contours = [cnt for cnt in outer_contours if cv2.contourArea(cnt) >= min_fragment_area]
                    if len(filtered_contours) < len(outer_contours):
                        print(f"   Filtered out {len(outer_contours) - len(filtered_contours)} tissue fragments (< 1/5 of max tissue)")
                    outer_contours = filtered_contours
            result_detection = detection_array.copy()
            result_detection_bgr = cv2.cvtColor(result_detection, cv2.COLOR_RGB2BGR)
            cv2.drawContours(result_detection_bgr, outer_contours, -1, (0, 255, 0), 3)
            cv2.drawContours(result_detection_bgr, hole_contours, -1, (255, 0, 0), 3)
            result_detection_rgb = cv2.cvtColor(result_detection_bgr, cv2.COLOR_BGR2RGB)
            h, w = result_detection_rgb.shape[:2]
            scale = min(800 / w, 800 / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            result_resized = cv2.resize(result_detection_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            result_final = np.ones((800, 800, 3), dtype=np.uint8) * 240
            y_offset = (800 - new_h) // 2
            x_offset = (800 - new_w) // 2
            result_final[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = result_resized
            result_pil = Image.fromarray(result_final)
            result_pil.save(os.path.join(circle_dir, f"{file_prefix}_tissue_contour_800x800.png"))
            thumbnail_resized = cv2.resize(detection_array_original, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            thumbnail_final = np.ones((800, 800, 3), dtype=np.uint8) * 240
            thumbnail_final[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = thumbnail_resized
            thumbnail_pil = Image.fromarray(thumbnail_final)
            thumbnail_pil.save(os.path.join(origin_dir, f"{file_prefix}_thumbnail_800x800.png"))
            mask_with_holes = np.zeros_like(binary_opened)
            cv2.drawContours(mask_with_holes, outer_contours, -1, 255, -1)
            cv2.drawContours(mask_with_holes, hole_contours, -1, 0, -1)
            binary_pil = Image.fromarray(mask_with_holes)
            binary_pil.save(os.path.join(mask_dir, f"{file_prefix}_binary_mask_800x800.png"))
            print(f"Processing file {idx}/{len(files)}: {file}\n")
            print(f"Level 0 dimensions: {level_0_dimensions}\n")
            print(f"Detection level {detection_level} dimensions: {level_dimensions}\n")
            print(f"Downsample factor: {downsample_factor:.2f}\n")
            print(f"Estimated magnification: {magnification}\n")
            print(f"Found {len(outer_contours)} tissue regions (green)\n")
            print(f"Found {len(hole_contours)} holes (blue)\n")
            print("===========================================")
            slide.close()

    process_files(args.pic_path, args.min_area, args.min_hole, args.file_type, args.out_dir)