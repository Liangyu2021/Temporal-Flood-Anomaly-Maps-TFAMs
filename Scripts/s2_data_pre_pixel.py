#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAR → HDF5 pipeline  ·  季节基准版
================================================
流程：
  1. 将所有影像按气象季节分组（春3-5 / 夏6-8 / 秋9-11 / 冬12-2）
  2. 对每个季节：
       → 以该季节【所有年份】的影像为基准影像集
       → 逐网格跑 HDF5 三步流程
  3. 输出目录：BASE_OUTPUT / 季节名 / grid_XX / *.hdf5
     季节名：spring / summer / autumn / winter
"""

import os
import re
import numpy as np
from osgeo import gdal
import h5py
from datetime import datetime
from collections import defaultdict
import traceback

# ──────────────────────────────────────────────────────────────────────────────
#  ★ 用户配置区
# ──────────────────────────────────────────────────────────────────────────────

BASE_INPUT  = r'/home/data/ql2024/flood-tly/australia/data/monthly_SARdata'
BASE_OUTPUT = r'/home/data/ql2024/flood-tly/australia/data/hdf5_seasons'

# dB 过滤阈值
DB_THRESHOLD_MIN = -30.0

# 气象季节定义：月份 → 季节名
# 可按需修改为自然季节或自定义月份分组
SEASON_MAP = {
    3: 'spring', 4: 'spring', 5: 'spring',
    6: 'summer', 7: 'summer', 8: 'summer',
    9: 'autumn', 10: 'autumn', 11: 'autumn',
    12: 'winter', 1: 'winter', 2: 'winter',
}

# 季节显示顺序（用于日志排序）
SEASON_ORDER = ['spring', 'summer', 'autumn', 'winter']

# ──────────────────────────────────────────────────────────────────────────────
#  工具函数
# ──────────────────────────────────────────────────────────────────────────────

def parse_filename(filename: str) -> datetime:
    basename = os.path.splitext(filename)[0]
    parts    = basename.split('_')
    if len(parts) < 2:
        raise ValueError(f"文件名格式不正确: {filename}")
    date_str = parts[1]
    if len(date_str) != 8:
        raise ValueError(f"日期段长度不对: {filename}")
    return datetime.strptime(date_str, '%Y%m%d')


def get_season(d: datetime) -> str:
    """根据月份返回季节名"""
    return SEASON_MAP[d.month]


def read_tif(filepath: str) -> np.ndarray:
    ds = gdal.Open(filepath)
    if ds is None:
        raise IOError(f"无法打开文件: {filepath}")
    data = ds.GetRasterBand(1).ReadAsArray()
    ds   = None
    return data

# ──────────────────────────────────────────────────────────────────────────────
#  扫描影像池：{ grid_name: [(filepath, date_obj, filename), ...] }
# ──────────────────────────────────────────────────────────────────────────────

def collect_all_grids(base_input_dir: str) -> dict:
    """兼容 month_XX/grid_XX 或扁平 grid_XX 两种输入结构"""
    month_pat = re.compile(r'^month_\d{2}$')
    grid_pat  = re.compile(r'^grid_\d+$')
    pool      = defaultdict(list)

    def _scan(grid_name, grid_dir):
        for fname in os.listdir(grid_dir):
            if not fname.lower().endswith('.tif'):
                continue
            try:
                pool[grid_name].append(
                    (os.path.join(grid_dir, fname), parse_filename(fname), fname)
                )
            except ValueError:
                pass

    for entry in sorted(os.listdir(base_input_dir)):
        ep = os.path.join(base_input_dir, entry)
        if not os.path.isdir(ep):
            continue
        if month_pat.match(entry):
            for sub in sorted(os.listdir(ep)):
                sp = os.path.join(ep, sub)
                if grid_pat.match(sub) and os.path.isdir(sp):
                    _scan(sub, sp)
        elif grid_pat.match(entry):
            _scan(entry, ep)

    for gname in pool:
        seen, unique = set(), []
        for item in pool[gname]:
            if item[0] not in seen:
                seen.add(item[0])
                unique.append(item)
        pool[gname] = sorted(unique, key=lambda x: x[1])

    return dict(pool)

# ──────────────────────────────────────────────────────────────────────────────
#  季节分组
# ──────────────────────────────────────────────────────────────────────────────

def group_by_season(grid_files: list) -> dict:
    """
    将 grid_files 按季节分组。
    返回 { season_name: [(filepath, date_obj, filename), ...] }
    """
    groups = defaultdict(list)
    for item in grid_files:
        season = get_season(item[1])
        groups[season].append(item)
    return dict(groups)

# ──────────────────────────────────────────────────────────────────────────────
#  HDF5 三步流程（与第二版保持完全一致的数据结构）
# ──────────────────────────────────────────────────────────────────────────────

def create_negative_hdf5_from_list(file_list, output_path, season_name, verbose=True):
    """
    Step 1: 读取原始 TIF → 取负值 → 过滤 dB 阈值 → 写入 HDF5
    数据结构与洪水事件版完全兼容（attrs 略有差异：season 替代 event_date）
    """
    n = len(file_list)
    if verbose:
        dates = [i[1].strftime('%Y%m%d') for i in file_list]
        print(f"        季节影像数: {n}  ({dates[0]} → {dates[-1]})")

    max_h = max_w = 0
    for fpath, _, _ in file_list:
        h, w = read_tif(fpath).shape
        max_h, max_w = max(max_h, h), max(max_w, w)

    with h5py.File(output_path, 'w') as hf:
        ds = hf.create_dataset('data', shape=(n, max_h, max_w),
                               dtype='float32', fillvalue=np.nan)
        total_masked = 0
        for i, (fpath, date_obj, fname) in enumerate(file_list):
            data = read_tif(fpath)
            h, w = data.shape
            arr  = -data.astype('float32')
            mask = arr > -float(DB_THRESHOLD_MIN)
            total_masked += int(mask.sum())
            arr[mask] = np.nan
            ds[i, :h, :w] = arr

            # ── 与洪水版完全相同的 per-image attrs ──
            hf.attrs[f'date_{i}']           = date_obj.strftime('%Y%m%d')
            hf.attrs[f'filename_{i}']       = fname
            hf.attrs[f'original_shape_{i}'] = f"{h}x{w}"

        # ── 全局 attrs（季节版新增 season / season_months）──
        hf.attrs['max_height']       = max_h
        hf.attrs['max_width']        = max_w
        hf.attrs['total_files']      = n
        hf.attrs['db_threshold_min'] = float(DB_THRESHOLD_MIN)
        hf.attrs['season']           = season_name
        # 记录该季节对应的月份，便于后续读取时查验
        season_months = sorted({m for m, s in SEASON_MAP.items() if s == season_name})
        hf.attrs['season_months']    = str(season_months)

        if verbose:
            print(f"        过滤像元总数（dB < {DB_THRESHOLD_MIN}）: {total_masked:,}")


def create_sorted_and_index_hdf5(original_hdf5_path, sorted_output_path, index_output_path):
    """Step 2: 逐像元排序 + 计算经验 CDF 索引（与洪水版完全一致）"""
    with h5py.File(original_hdf5_path, 'r') as hf:
        original_data = hf['data'][:]
        attrs_dict    = dict(hf.attrs)

    n_times, height, width = original_data.shape
    sorted_data    = np.full_like(original_data, np.nan)
    index_data     = np.full_like(original_data, np.nan)
    sortorder_data = np.full((n_times, height, width), -1, dtype=np.int32)

    for h in range(height):
        for w in range(width):
            ts         = original_data[:, h, w]
            valid_mask = ~np.isnan(ts)
            n_valid    = valid_mask.sum()
            if n_valid == 0:
                continue
            valid_vals = ts[valid_mask]
            valid_idx  = np.where(valid_mask)[0]
            sort_order = np.argsort(valid_vals)
            sorted_data[valid_mask, h, w]    = valid_vals[sort_order]
            index_data[valid_mask, h, w]     = np.arange(1, n_valid + 1) / (n_valid + 1)
            sortorder_data[valid_mask, h, w] = valid_idx[sort_order]

    for out_path, arr in [(sorted_output_path, sorted_data), (index_output_path, index_data)]:
        with h5py.File(out_path, 'w') as hf:
            hf.create_dataset('data',       data=arr,            dtype='float32')
            hf.create_dataset('sort_order', data=sortorder_data, dtype='int32')
            for k, v in attrs_dict.items():
                hf.attrs[k] = v


def create_normalized_hdf5(sorted_hdf5_path, normalized_output_path, verbose=True):
    """Step 3: 逐像元 min-max 归一化（与洪水版完全一致）"""
    with h5py.File(sorted_hdf5_path, 'r') as hf:
        sorted_data    = hf['data'][:]
        sortorder_data = hf['sort_order'][:]
        attrs_dict     = dict(hf.attrs)

    pixel_min    = np.nanmin(sorted_data, axis=0)
    pixel_max    = np.nanmax(sorted_data, axis=0)
    denom        = pixel_max - pixel_min
    invalid_mask = (denom == 0) | np.isnan(denom)
    denom_safe   = np.where(invalid_mask, 1.0, denom)

    normalized_data = (sorted_data - pixel_min[np.newaxis]) / denom_safe[np.newaxis]
    normalized_data[:, invalid_mask] = np.nan

    with h5py.File(normalized_output_path, 'w') as hf:
        hf.create_dataset('data',       data=normalized_data.astype('float32'), dtype='float32')
        hf.create_dataset('sort_order', data=sortorder_data,                    dtype='int32')
        hf.create_dataset('pixel_min',
                          data=np.where(invalid_mask, np.nan, pixel_min).astype('float32'),
                          dtype='float32')
        hf.create_dataset('pixel_max',
                          data=np.where(invalid_mask, np.nan, pixel_max).astype('float32'),
                          dtype='float32')
        for k, v in attrs_dict.items():
            hf.attrs[k] = v
        hf.attrs['norm_method'] = 'per_pixel_minmax_on_sorted'

    if verbose:
        n_inv = int(invalid_mask.sum())
        print(f"        无效像元（常数/全NaN）: {n_inv}/{invalid_mask.size}")
        print(f"        ✓ 归一化写出 → {normalized_output_path}")

# ──────────────────────────────────────────────────────────────────────────────
#  单网格 × 单季节
# ──────────────────────────────────────────────────────────────────────────────

def process_one_grid(season_files, grid_output_dir, season_name):
    os.makedirs(grid_output_dir, exist_ok=True)
    orig = os.path.join(grid_output_dir, 'original_negative_values.hdf5')
    srt  = os.path.join(grid_output_dir, 'sorted_negative_values.hdf5')
    idx  = os.path.join(grid_output_dir, 'index_values.hdf5')
    nrm  = os.path.join(grid_output_dir, 'normalized_negative_values.hdf5')

    print(f"        [1/3] original ...")
    create_negative_hdf5_from_list(season_files, orig, season_name)
    print(f"        [2/3] sorted / index ...")
    create_sorted_and_index_hdf5(orig, srt, idx)
    print(f"        [3/3] normalized ...")
    create_normalized_hdf5(srt, nrm)

# ──────────────────────────────────────────────────────────────────────────────
#  主流程
# ──────────────────────────────────────────────────────────────────────────────

def batch_process(base_input_dir, base_output_dir,
                  target_seasons=None, grid_filter=None):
    """
    Parameters
    ----------
    base_input_dir  : 输入根目录
    base_output_dir : 输出根目录
    target_seasons  : 要处理的季节列表，如 ['spring','summer']；None = 全部四季
    grid_filter     : 要处理的网格列表，如 ['grid_1','grid_3']；None = 全部
    """
    if not os.path.exists(base_input_dir):
        raise FileNotFoundError(f"输入目录不存在: {base_input_dir}")

    seasons_to_run = target_seasons if target_seasons else SEASON_ORDER

    # ── 扫描全量影像池 ────────────────────────────────────────────
    print("=" * 70)
    print(f"扫描输入: {base_input_dir}")
    all_grids = collect_all_grids(base_input_dir)
    if not all_grids:
        raise ValueError("未找到任何 grid_XX 目录或合法 tif 文件")
    if grid_filter:
        all_grids = {k: v for k, v in all_grids.items() if k in grid_filter}

    total_tifs = sum(len(v) for v in all_grids.values())
    print(f"网格数: {len(all_grids)}   总影像数（去重）: {total_tifs}")
    print(f"处理季节: {seasons_to_run}")
    print(f"dB 过滤阈值:  < {DB_THRESHOLD_MIN} dB")
    print("=" * 70)

    grid_names = sorted(all_grids.keys(),
                        key=lambda g: int(re.search(r'\d+', g).group()))

    success_list, failed_list = [], []

    for grid_name in grid_names:
        grid_files = all_grids[grid_name]

        # ── 按季节分组 ────────────────────────────────────────────
        season_groups = group_by_season(grid_files)

        print(f"\n{'─'*70}")
        print(f"[{grid_name}]  各季节影像数: " +
              "  ".join(f"{s}={len(season_groups.get(s,[]))}"
                        for s in SEASON_ORDER))

        for season_name in seasons_to_run:
            task_tag     = f"{grid_name}/{season_name}"
            season_files = season_groups.get(season_name, [])

            print(f"\n  季节: {season_name}  影像数: {len(season_files)}")

            if not season_files:
                msg = "该季节无影像"
                print(f"    ⚠ {msg}，跳过")
                failed_list.append((task_tag, msg))
                continue

            grid_output_dir = os.path.join(base_output_dir, season_name, grid_name)
            try:
                process_one_grid(season_files, grid_output_dir, season_name)
                success_list.append(task_tag)
                print(f"    ✓ 完成 → {grid_output_dir}")
            except Exception as e:
                failed_list.append((task_tag, str(e)))
                print(f"    ✗ 失败: {e}")
                traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"完成  成功 {len(success_list)}  失败 {len(failed_list)}  "
          f"共 {len(success_list)+len(failed_list)} 个任务")
    if failed_list:
        print("失败任务:")
        for tag, err in failed_list:
            print(f"  ✗ {tag}: {err}")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    batch_process(
        base_input_dir  = BASE_INPUT,
        base_output_dir = BASE_OUTPUT,
        target_seasons  = None,       # None=全四季；['spring','summer'] 指定季节
        grid_filter     = None,       # None=全部；['grid_1','grid_3'] 指定网格
    )