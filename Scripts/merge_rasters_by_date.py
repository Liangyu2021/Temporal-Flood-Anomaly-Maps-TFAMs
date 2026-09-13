#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按日期镶嵌（Mosaic）栅格文件 - 支持不同空间范围的拼接
"""

import os
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
from osgeo import gdal, gdalconst, osr

# 启用异常处理
gdal.UseExceptions()


def extract_date_from_filename(filename):
    """从文件名中提取日期标识"""
    patterns = [
        r'Fx[_-]?(\d{5})',  # Fx_08000 或 Fx08000
        r'event[_-]?(\d+)',  # event_12345
        r'(\d{8})',  # 20230101
        r'(\d{6})',  # 202301
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return None


def find_numbered_folders(base_dir):
    """查找所有编号文件夹"""
    numbered_folders = []
    
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            # 修改正则表达式，匹配 _01, _02 等结尾的文件夹
            if re.search(r'_(\d{2})(?:_|$)', item):
                numbered_folders.append(item_path)
    
    return sorted(numbered_folders)


def collect_rasters_by_date(folders, extensions=('.tif', '.tiff', '.img')):
    """收集所有栅格文件并按日期分组"""
    date_groups = defaultdict(list)
    
    for folder in folders:
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(extensions):
                    date_id = extract_date_from_filename(file)
                    if date_id:
                        file_path = os.path.join(root, file)
                        date_groups[date_id].append(file_path)
    
    return date_groups


def get_raster_extent(file_path):
    """
    获取栅格的空间范围
    
    Returns:
        (min_x, max_x, min_y, max_y, pixel_width, pixel_height, projection)
    """
    ds = gdal.Open(file_path)
    if ds is None:
        return None
    
    gt = ds.GetGeoTransform()
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    projection = ds.GetProjection()
    
    min_x = gt[0]
    max_x = gt[0] + cols * gt[1]
    max_y = gt[3]
    min_y = gt[3] + rows * gt[5]
    
    pixel_width = gt[1]
    pixel_height = gt[5]
    
    ds = None
    
    return (min_x, max_x, min_y, max_y, pixel_width, pixel_height, projection)


def calculate_mosaic_extent(input_files):
    """
    计算所有栅格的最大范围和统一分辨率
    
    Returns:
        (min_x, max_x, min_y, max_y, pixel_width, pixel_height, projection)
    """
    extents = []
    pixel_sizes = []
    projections = []
    
    print(f"  分析 {len(input_files)} 个文件的空间范围...")
    
    for file_path in input_files:
        extent = get_raster_extent(file_path)
        if extent:
            extents.append(extent[:4])
            pixel_sizes.append((extent[4], extent[5]))
            projections.append(extent[6])
            print(f"    {os.path.basename(file_path)}: "
                  f"X=[{extent[0]:.2f}, {extent[1]:.2f}], "
                  f"Y=[{extent[2]:.2f}, {extent[3]:.2f}]")
    
    if len(extents) == 0:
        raise ValueError("没有有效的栅格文件")
    
    # 检查投影是否一致
    if len(set(projections)) > 1:
        print("  警告: 检测到不同的投影系统，使用第一个文件的投影")
    
    # 计算最大范围
    min_x = min(e[0] for e in extents)
    max_x = max(e[1] for e in extents)
    min_y = min(e[2] for e in extents)
    max_y = max(e[3] for e in extents)
    
    # 使用最常见的像素大小（或第一个）
    pixel_width = pixel_sizes[0][0]
    pixel_height = pixel_sizes[0][1]
    
    print(f"\n  镶嵌范围: X=[{min_x:.2f}, {max_x:.2f}], Y=[{min_y:.2f}, {max_y:.2f}]")
    print(f"  像素大小: {pixel_width:.6f} x {pixel_height:.6f}")
    
    return (min_x, max_x, min_y, max_y, pixel_width, pixel_height, projections[0])


def mosaic_rasters(input_files, output_file, method='mean'):
    """
    镶嵌多个栅格文件到统一范围
    
    Args:
        input_files: 输入文件列表
        output_file: 输出文件路径
        method: 重叠区域处理方法 ('mean', 'max', 'min', 'first', 'last')
    """
    if len(input_files) == 0:
        print(f"警告: 没有输入文件")
        return
    
    if len(input_files) == 1:
        print(f"  只有1个文件，直接复制")
        src_ds = gdal.Open(input_files[0])
        driver = gdal.GetDriverByName('GTiff')
        driver.CreateCopy(output_file, src_ds, options=['COMPRESS=LZW'])
        src_ds = None
        return
    
    print(f"  开始镶嵌 {len(input_files)} 个文件...")
    
    # 计算镶嵌范围
    min_x, max_x, min_y, max_y, pixel_width, pixel_height, projection = \
        calculate_mosaic_extent(input_files)
    
    # 计算输出栅格尺寸
    cols = int(np.ceil((max_x - min_x) / pixel_width))
    rows = int(np.ceil((max_y - min_y) / abs(pixel_height)))
    
    print(f"  输出尺寸: {cols} x {rows} 像素")
    
    # 创建输出数组
    if method == 'mean':
        mosaic_sum = np.zeros((rows, cols), dtype=np.float64)
        mosaic_count = np.zeros((rows, cols), dtype=np.int32)
    else:
        mosaic_data = np.full((rows, cols), np.nan, dtype=np.float32)
    
    # 读取第一个文件获取 NoData 值
    first_ds = gdal.Open(input_files[0])
    nodata = first_ds.GetRasterBand(1).GetNoDataValue()
    first_ds = None
    
    # 逐个文件镶嵌
    for i, file_path in enumerate(input_files, 1):
        print(f"  处理 [{i}/{len(input_files)}]: {os.path.basename(file_path)}")
        
        ds = gdal.Open(file_path)
        if ds is None:
            print(f"    警告: 无法打开文件")
            continue
        
        # 读取数据
        band = ds.GetRasterBand(1)
        data = band.ReadAsArray()
        
        # 获取该文件的地理变换
        gt = ds.GetGeoTransform()
        file_min_x = gt[0]
        file_max_y = gt[3]
        
        # 计算在输出栅格中的位置
        start_col = int((file_min_x - min_x) / pixel_width)
        start_row = int((max_y - file_max_y) / abs(pixel_height))
        
        end_col = start_col + data.shape[1]
        end_row = start_row + data.shape[0]
        
        # 确保不超出边界
        start_col = max(0, start_col)
        start_row = max(0, start_row)
        end_col = min(cols, end_col)
        end_row = min(rows, end_row)
        
        # 裁剪数据以匹配边界
        data_start_col = max(0, -int((file_min_x - min_x) / pixel_width))
        data_start_row = max(0, -int((max_y - file_max_y) / abs(pixel_height)))
        data_end_col = data_start_col + (end_col - start_col)
        data_end_row = data_start_row + (end_row - start_row)
        
        data_clip = data[data_start_row:data_end_row, data_start_col:data_end_col]
        
        # 处理 NoData
        if nodata is not None:
            valid_mask = data_clip != nodata
        else:
            valid_mask = ~np.isnan(data_clip)
        
        # 根据方法镶嵌
        if method == 'mean':
            mosaic_sum[start_row:end_row, start_col:end_col][valid_mask] += data_clip[valid_mask]
            mosaic_count[start_row:end_row, start_col:end_col][valid_mask] += 1
        elif method == 'max':
            current = mosaic_data[start_row:end_row, start_col:end_col]
            mosaic_data[start_row:end_row, start_col:end_col] = np.where(
                valid_mask & (np.isnan(current) | (data_clip > current)),
                data_clip, current
            )
        elif method == 'min':
            current = mosaic_data[start_row:end_row, start_col:end_col]
            mosaic_data[start_row:end_row, start_col:end_col] = np.where(
                valid_mask & (np.isnan(current) | (data_clip < current)),
                data_clip, current
            )
        elif method == 'first':
            current = mosaic_data[start_row:end_row, start_col:end_col]
            mosaic_data[start_row:end_row, start_col:end_col] = np.where(
                valid_mask & np.isnan(current),
                data_clip, current
            )
        elif method == 'last':
            mosaic_data[start_row:end_row, start_col:end_col] = np.where(
                valid_mask,
                data_clip, 
                mosaic_data[start_row:end_row, start_col:end_col]
            )
        
        ds = None
    
    # 计算最终结果
    if method == 'mean':
        with np.errstate(divide='ignore', invalid='ignore'):
            mosaic_data = mosaic_sum / mosaic_count
            mosaic_data[mosaic_count == 0] = nodata if nodata is not None else np.nan
    
    # 创建输出文件
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_file, cols, rows, 1, gdalconst.GDT_Float32,
                          options=['COMPRESS=LZW', 'BIGTIFF=YES'])
    
    # 设置地理变换
    out_gt = (min_x, pixel_width, 0, max_y, 0, pixel_height)
    out_ds.SetGeoTransform(out_gt)
    out_ds.SetProjection(projection)
    
    # 写入数据
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(mosaic_data)
    if nodata is not None:
        out_band.SetNoDataValue(nodata)
    
    # 关闭数据集
    out_band.FlushCache()
    out_ds = None
    
    print(f"  ✓ 完成: {os.path.basename(output_file)}")


def main():
    """主函数"""
    # 配置参数
    BASE_DIR = r"/home/data/ql2024/flood-tly/greek/data" # 修改为你的实际路径
    
    # 要处理的顶层文件夹
    #TOP_FOLDERS = ['event_Fx','monthresults','event_Fx_mask']
    TOP_FOLDERS = ['event_Fx']
    
    # 镶嵌方法:
    # 'mean'  - 重叠区域取平均值（推荐）
    # 'max'   - 重叠区域取最大值
    # 'min'   - 重叠区域取最小值
    # 'first' - 重叠区域保留第一个值
    # 'last'  - 重叠区域保留最后一个值
    MOSAIC_METHOD = 'first'
    
    print("=" * 70)
    print("栅格镶嵌工具 - 按日期拼接不同空间范围的栅格")
    print("=" * 70)
    
    for top_folder in TOP_FOLDERS:
        top_path = os.path.join(BASE_DIR, top_folder)
        
        if not os.path.exists(top_path):
            print(f"\n跳过: {top_folder} (目录不存在)")
            continue
        
        print(f"\n{'='*70}")
        print(f"处理文件夹: {top_folder}")
        print('='*70)
        
        # 查找编号文件夹
        numbered_folders = find_numbered_folders(top_path)
        
        if len(numbered_folders) == 0:
            print("  未找到编号子文件夹（如 *_01, *_02）")
            continue
        
        print(f"\n找到 {len(numbered_folders)} 个编号文件夹:")
        for folder in numbered_folders:
            print(f"  - {os.path.basename(folder)}")
        
        # 收集栅格文件并按日期分组
        date_groups = collect_rasters_by_date(numbered_folders)
        
        if len(date_groups) == 0:
            print("  未找到栅格文件")
            continue
        
        print(f"\n找到 {len(date_groups)} 个不同日期")
        
        # 创建输出文件夹
        output_dir = os.path.join(top_path, 'mosaicked_by_date')
        os.makedirs(output_dir, exist_ok=True)
        
        # 镶嵌每个日期的文件
        for date_id, files in sorted(date_groups.items()):
            print(f"\n{'-'*70}")
            print(f"日期: {date_id} ({len(files)} 个文件)")
            print('-'*70)
            
            # 构建输出文件名
            output_filename = f"{top_folder}_{date_id}_mosaic.tif"
            output_path = os.path.join(output_dir, output_filename)
            
            # 镶嵌
            try:
                mosaic_rasters(files, output_path, method=MOSAIC_METHOD)
            except Exception as e:
                print(f"  错误: {str(e)}")
                continue
        
        print(f"\n输出目录: {output_dir}")
    
    print("\n" + "=" * 70)
    print("处理完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()