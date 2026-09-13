#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将洪水识别结果（0-1连续值）与mask相乘，生成最终结果
"""

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


def resample_mask_to_flood(mask_path, flood_profile):
    """将mask重采样到与洪水栅格相同的范围和分辨率"""
    with rasterio.open(mask_path) as src:
        mask_data = src.read(1)
        mask_transform = src.transform
        mask_crs = src.crs

        if (mask_data.shape == (flood_profile['height'], flood_profile['width']) and
                mask_transform == flood_profile['transform'] and
                mask_crs == flood_profile['crs']):
            return mask_data

        dst_mask = np.zeros((flood_profile['height'], flood_profile['width']),
                            dtype=mask_data.dtype)

        reproject(
            source=mask_data,
            destination=dst_mask,
            src_transform=mask_transform,
            src_crs=mask_crs,
            dst_transform=flood_profile['transform'],
            dst_crs=flood_profile['crs'],
            resampling=Resampling.nearest
        )

        return dst_mask


def apply_mask_to_floods(flood_dir, mask_path, output_dir):
    """
    将单个mask应用到洪水识别结果

    参数:
        flood_dir:  洪水镶嵌结果目录
        mask_path:  mask文件路径
        output_dir: 输出目录
    """
    flood_dir  = Path(flood_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*70}")
    print(f"应用Mask到洪水识别结果")
    print(f"{'='*70}")
    print(f"洪水结果目录 : {flood_dir}")
    print(f"Mask文件     : {mask_path}")
    print(f"输出目录     : {output_dir}")
    print(f"{'='*70}\n")

    flood_files = sorted(flood_dir.glob("event_Fx_*.tif"))
    print(f"发现 {len(flood_files)} 个洪水栅格文件\n")

    if not flood_files:
        print("警告: 未找到洪水栅格文件")
        return

    processed_count = 0
    error_count     = 0

    for flood_file in tqdm(flood_files, desc="应用mask", ncols=100):
        try:
            with rasterio.open(flood_file) as src:
                flood_data    = src.read(1).astype(np.float32)
                flood_profile = src.profile.copy()

            mask = resample_mask_to_flood(mask_path, flood_profile).astype(np.float32)
            result = flood_data * mask

            output_profile = flood_profile.copy()
            output_profile.update(dtype=rasterio.float32)

            output_filename = flood_file.name.replace("event_Fx_", "event_Fx_Masked_")
            output_file     = output_dir / output_filename

            with rasterio.open(output_file, 'w', **output_profile) as dst:
                dst.write(result, 1)

            processed_count += 1

        except Exception as e:
            print(f"错误: 处理 {flood_file.name} 时出错: {e}")
            error_count += 1
            continue

    print(f"\n{'='*70}")
    print(f"处理完成!")
    print(f"  成功处理: {processed_count} 个文件")
    print(f"  错误:     {error_count} 个文件")
    print(f"{'='*70}\n")


def main():

    # ==========================================================================
    # 配置区域
    # ==========================================================================

    # 洪水镶嵌结果目录（包含 event_Fx_YYYYMMDD_mosaic.tif 文件）
    flood_mosaic_dir = "/home/data/ql2024/flood-tly/greek/data/event_Fx/mosaicked_by_date"

    # Mask文件路径
    mask_path        = "/home/data/ql2024/flood-tly/greek/data/flood_result_30m_filtered/flood_prediction_30m.tif"

    # 输出目录
    output_dir       = "/home/data/ql2024/flood-tly/greek/data/event_Fx/masked"

    # ==========================================================================

    apply_mask_to_floods(
        flood_dir  = flood_mosaic_dir,
        mask_path  = mask_path,
        output_dir = output_dir,
    )

    print("全部完成!")


if __name__ == "__main__":
    main()