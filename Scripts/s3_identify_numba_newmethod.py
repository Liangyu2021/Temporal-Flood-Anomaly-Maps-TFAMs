#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
季节×网格 TCEV 批量分析脚本

依赖上游脚本的输出：
  sar_to_hdf5_seasonal.py  → hdf5_seasonal/季节名/grid_XX/*.hdf5
                             季节名：spring / summer / autumn / winter

输出目录结构：
  tcev_seasonal/
  ├── spring/
  │   ├── grid_01/
  │   │   ├── R_index.tif
  │   │   ├── L1_alpha.tif  L1_beta.tif
  │   │   ├── L2_alpha.tif  L2_beta.tif
  │   │   ├── breakpoint_DB.tif  breakpoint_FX.tif
  │   │   ├── piecewise_residual.tif
  │   │   ├── linear_residual.tif
  │   │   ├── n_valid.tif
  │   │   └── sample_pixel.xlsx
  │   ├── grid_02/
  │   │   └── ...
  │   └── mosaic/
  │       ├── R_index.tif  ...
  │       └── n_valid.tif
  ├── summer/  autumn/  winter/
  └── ...

季节定义（气象季节，与上游 hdf5 生成脚本保持一致）：
  spring : 3-5月   summer : 6-8月
  autumn : 9-11月  winter : 12-2月
"""

import os
import re
import traceback
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import h5py
import pandas as pd
from osgeo import gdal, osr
from multiprocessing import Pool, Manager
from numba import jit


# ============================================================
# 季节定义（与上游 hdf5 脚本保持一致）
# ============================================================

SEASON_MAP = {
    3: 'spring', 4: 'spring', 5: 'spring',
    6: 'summer', 7: 'summer', 8: 'summer',
    9: 'autumn', 10: 'autumn', 11: 'autumn',
    12: 'winter', 1: 'winter', 2: 'winter',
}

# 季节 → 对应月份列表（用于在 monthly_SARdata 中寻找参考TIF）
SEASON_TO_MONTHS = {
    'spring': [3, 4, 5],
    'summer': [6, 7, 8],
    'autumn': [9, 10, 11],
    'winter': [12, 1, 2],
}

SEASON_ORDER = ['spring', 'summer', 'autumn', 'winter']


# ============================================================
# Numba 加速核心函数（与月份版完全一致，无需修改）
# ============================================================

@jit(nopython=True)
def filter_valid_data(db_series, fx_series, db_threshold):
    n = len(db_series)
    valid_count = 0
    for i in range(n):
        if not np.isnan(db_series[i]) and not np.isnan(fx_series[i]):
            if 0 < fx_series[i] < 1:
                if db_series[i] <= db_threshold:
                    valid_count += 1
    if valid_count < 8:
        return np.array([0.0]), np.array([0.0]), False
    db_valid = np.zeros(valid_count)
    fx_valid = np.zeros(valid_count)
    idx = 0
    for i in range(n):
        if not np.isnan(db_series[i]) and not np.isnan(fx_series[i]):
            if 0 < fx_series[i] < 1:
                if db_series[i] <= db_threshold:
                    db_valid[idx] = db_series[i]
                    fx_valid[idx] = fx_series[i]
                    idx += 1
    return db_valid, fx_valid, True


@jit(nopython=True)
def gumbel_transform_fast(F):
    return -np.log(-np.log(F))


@jit(nopython=True)
def simple_linregress(x, y):
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = 0.0
    denominator = 0.0
    for i in range(n):
        numerator   += (x[i] - x_mean) * (y[i] - y_mean)
        denominator += (x[i] - x_mean) ** 2
    slope = numerator / denominator if denominator != 0 else 0.0
    intercept = y_mean - slope * x_mean
    return slope, intercept


@jit(nopython=True)
def find_breakpoint_by_slope_difference(x, y, min_points=4):
    n = len(x)
    start_idx    = n // 2
    end_idx      = n - min_points
    search_start = max(start_idx, end_idx - 20)
    best_bp_idx  = -1
    best_slope_diff = -np.inf
    for i in range(end_idx - 1, search_start - 1, -1):
        if i < min_points or (n - i) < min_points:
            continue
        slope_left,  _ = simple_linregress(x[:i], y[:i])
        slope_right, _ = simple_linregress(x[i:], y[i:])
        slope_diff = np.abs(slope_right - slope_left)
        if slope_diff > best_slope_diff:
            best_slope_diff = slope_diff
            best_bp_idx = i
    if best_bp_idx == -1:
        best_bp_idx = (start_idx + end_idx) // 2
    return best_bp_idx, best_slope_diff


@jit(nopython=True)
def linear_to_tcev_params_fast(a1, c1, a2, c2):
    alpha1 = a1
    beta1  = -c1 / a1 if a1 != 0 else 0.0
    alpha2 = a2
    beta2  = -c2 / a2 if a2 != 0 else 0.0
    return alpha1, beta1, alpha2, beta2


@jit(nopython=True)
def compute_piecewise_residual(x, y, bp_idx, a1, c1, a2, c2):
    n = len(x)
    sse = 0.0
    for i in range(n):
        y_pred = a1 * x[i] + c1 if i < bp_idx else a2 * x[i] + c2
        sse += (y[i] - y_pred) ** 2
    return np.sqrt(sse / n) if n > 0 else np.nan


@jit(nopython=True)
def compute_linear_residual(x, y):
    slope, intercept = simple_linregress(x, y)
    n = len(x)
    sse = 0.0
    for i in range(n):
        y_pred = slope * x[i] + intercept
        sse += (y[i] - y_pred) ** 2
    return np.sqrt(sse / n) if n > 0 else np.nan


# ============================================================
# 像元级处理（与月份版完全一致）
# ============================================================

def fit_piecewise_linear_with_slope_search(x, y, min_points=4):
    n = len(x)
    if n < 8:
        raise ValueError(f"数据点不足: 只有{n}个点")
    bp_idx, _ = find_breakpoint_by_slope_difference(x, y, min_points)
    bp = x[bp_idx]
    a1, c1 = simple_linregress(x[:bp_idx], y[:bp_idx])
    a2, _  = simple_linregress(x[bp_idx:], y[bp_idx:])
    c2 = a1 * bp + c1 - a2 * bp
    for val, name in [(a1,'a1'),(a2,'a2'),(c1,'c1'),(c2,'c2'),(bp,'bp')]:
        if np.isnan(val) or np.isinf(val):
            raise ValueError(f"拟合产生无效参数 {name}={val}")
    if abs(a1) < 1e-10 or abs(a2) < 1e-10:
        raise ValueError(f"斜率接近零: a1={a1:.2e}, a2={a2:.2e}")
    return a1, a2, c1, c2, bp, bp_idx


def process_pixel_timeseries_fast(db_abs, fx, db_threshold):
    db_valid, fx_valid, is_valid = filter_valid_data(db_abs, fx, db_threshold)
    if not is_valid:
        raise ValueError("有效数据点不足8个")
    y = gumbel_transform_fast(fx_valid)
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("Gumbel变换产生无效值")

    a1, a2, c1, c2, bp, bp_idx = fit_piecewise_linear_with_slope_search(db_valid, y)
    alpha1, beta1, alpha2, beta2 = linear_to_tcev_params_fast(a1, c1, a2, c2)
    R = alpha2 / alpha1 if alpha1 != 0 else 0.0
    bp_fx = fx_valid[np.argmin(np.abs(db_valid - bp))]

    piecewise_rmse = compute_piecewise_residual(db_valid, y, bp_idx, a1, c1, a2, c2)
    linear_rmse    = compute_linear_residual(db_valid, y)
    n_valid        = len(db_valid)

    return R, alpha1, beta1, alpha2, beta2, bp, bp_fx, piecewise_rmse, linear_rmse, n_valid


# ============================================================
# 多进程块处理（与月份版完全一致）
# ============================================================

def process_block(args):
    (block_id, h_start, h_end, w_start, w_end,
     db_data, fx_data, db_threshold,
     counter, lock, total_pixels) = args

    hb, wb = h_end - h_start, w_end - w_start
    nan2d = lambda: np.full((hb, wb), np.nan, dtype=np.float32)
    R_b, a1_b, b1_b, a2_b, b2_b, bp_db_b, bp_fx_b = (nan2d() for _ in range(7))
    pw_rmse_b  = nan2d()
    lin_rmse_b = nan2d()
    n_valid_b  = nan2d()

    err_count = suc_count = proc = 0
    error_types = {}

    for h in range(hb):
        for w in range(wb):
            gh, gw = h_start + h, w_start + w
            try:
                (R, a1, b1, a2, b2, bp_db, bp_fx,
                 pw_rmse, lin_rmse, n_valid_px) = process_pixel_timeseries_fast(
                    db_data[:, gh, gw], fx_data[:, gh, gw], db_threshold)
                R_b[h,w]=R;       a1_b[h,w]=a1;          b1_b[h,w]=b1
                a2_b[h,w]=a2;     b2_b[h,w]=b2
                bp_db_b[h,w]=bp_db; bp_fx_b[h,w]=bp_fx
                pw_rmse_b[h,w]=pw_rmse
                lin_rmse_b[h,w]=lin_rmse
                n_valid_b[h,w]=float(n_valid_px)
                suc_count += 1
            except Exception as e:
                err = str(e)
                error_types[err] = error_types.get(err, 0) + 1
                err_count += 1

            proc += 1
            if proc % 500 == 0:
                with lock:
                    counter.value += 500
                    cur = counter.value
                    print(f"\r  总进度: {cur:,}/{total_pixels:,} "
                          f"({100*cur/total_pixels:.1f}%)", end='', flush=True)

    rem = proc % 500
    if rem:
        with lock:
            counter.value += rem

    rate = 100 * suc_count / proc if proc else 0
    print(f"\n  进程{block_id}: 成功{suc_count}/{proc} ({rate:.1f}%)")
    if error_types:
        for msg, cnt in sorted(error_types.items(), key=lambda x: -x[1])[:3]:
            print(f"    - {msg}: {cnt}次")

    return (block_id, h_start, h_end, w_start, w_end,
            R_b, a1_b, b1_b, a2_b, b2_b, bp_db_b, bp_fx_b,
            pw_rmse_b, lin_rmse_b, n_valid_b,
            err_count, suc_count)


def process_hdf5_data_parallel(sorted_hdf5, index_hdf5, db_threshold, n_processes=4):
    with h5py.File(sorted_hdf5, 'r') as hf:
        db_data = hf['data'][:]
    with h5py.File(index_hdf5, 'r') as hf:
        fx_data = hf['data'][:]

    _, height, width = db_data.shape
    total_pixels = height * width

    nan2d = lambda: np.full((height, width), np.nan, dtype=np.float32)
    R, a1, b1, a2, b2, bp_db, bp_fx = (nan2d() for _ in range(7))
    pw_rmse  = nan2d()
    lin_rmse = nan2d()
    n_valid  = nan2d()

    manager = Manager()
    counter = manager.Value('i', 0)
    lock    = manager.Lock()
    rpp     = height // n_processes

    block_args = []
    for i in range(n_processes):
        hs = i * rpp
        he = height if i == n_processes - 1 else (i + 1) * rpp
        block_args.append((i, hs, he, 0, width,
                           db_data, fx_data, db_threshold,
                           counter, lock, total_pixels))

    t0 = time.time()
    with Pool(processes=n_processes) as pool:
        results = pool.map(process_block, block_args)

    total_suc = total_err = 0
    for res in results:
        (bid, hs, he, ws, we,
         Rb, a1b, b1b, a2b, b2b, bpdb, bpfb,
         pw_b, lin_b, nv_b,
         ec, sc) = res
        R[hs:he]        = Rb
        a1[hs:he]       = a1b
        b1[hs:he]       = b1b
        a2[hs:he]       = a2b
        b2[hs:he]       = b2b
        bp_db[hs:he]    = bpdb
        bp_fx[hs:he]    = bpfb
        pw_rmse[hs:he]  = pw_b
        lin_rmse[hs:he] = lin_b
        n_valid[hs:he]  = nv_b
        total_err += ec
        total_suc += sc

    elapsed = time.time() - t0
    print(f"  耗时 {elapsed/60:.1f} min | "
          f"成功 {total_suc:,}/{total_pixels:,} ({100*total_suc/total_pixels:.1f}%) | "
          f"速度 {total_pixels/elapsed:.0f} px/s")

    return R, a1, b1, a2, b2, bp_db, bp_fx, pw_rmse, lin_rmse, n_valid


# ============================================================
# 栅格保存 / 样本导出（与月份版完全一致）
# ============================================================

LAYER_NAMES = ['R_index', 'L1_alpha', 'L1_beta',
               'L2_alpha', 'L2_beta', 'breakpoint_DB', 'breakpoint_FX',
               'piecewise_residual', 'linear_residual', 'n_valid']


def save_raster_as_tif(data, output_path, reference_tif=None):
    driver = gdal.GetDriverByName('GTiff')
    geotransform = projection = None
    if reference_tif and os.path.exists(reference_tif):
        ref_ds = gdal.Open(reference_tif)
        if ref_ds:
            geotransform = ref_ds.GetGeoTransform()
            projection   = ref_ds.GetProjection()
            ref_ds = None

    out_ds = driver.Create(output_path, data.shape[1], data.shape[0],
                           1, gdal.GDT_Float32, options=['COMPRESS=LZW'])
    if geotransform:
        out_ds.SetGeoTransform(geotransform)
    if projection:
        out_ds.SetProjection(projection)

    band = out_ds.GetRasterBand(1)
    band.WriteArray(data)
    band.SetNoDataValue(-9999)
    band.FlushCache()
    out_ds = None


def export_sample_pixel(sorted_hdf5, index_hdf5,
                        R, a1, b1, a2, b2, bp_db, bp_fx,
                        pw_rmse, lin_rmse, n_valid, output_excel):
    with h5py.File(sorted_hdf5, 'r') as hf:
        db_arr = hf['data'][:]
    with h5py.File(index_hdf5, 'r') as hf:
        fx_arr = hf['data'][:]

    height, width = R.shape
    sh, sw = height // 2, width // 2
    if np.isnan(R[sh, sw]):
        valid = np.where(~np.isnan(R))
        if len(valid[0]):
            dist = (valid[0]-sh)**2 + (valid[1]-sw)**2
            sh, sw = valid[0][np.argmin(dist)], valid[1][np.argmin(dist)]
        else:
            return

    df_ts = pd.DataFrame({
        '时间索引':  range(1, db_arr.shape[0] + 1),
        'DB绝对值': db_arr[:, sh, sw],
        'F(x)指数': fx_arr[:, sh, sw],
    })
    df_param = pd.DataFrame({
        '参数': ['R指数','L1_alpha','L1_beta','L2_alpha','L2_beta',
                 '断点DB','断点FX',
                 '分段拟合RMSE（Gumbel-DB）',
                 '全段线性RMSE（Gumbel-DB）',
                 '有效样本量 n'],
        '值':   [R[sh,sw], a1[sh,sw], b1[sh,sw],
                 a2[sh,sw], b2[sh,sw], bp_db[sh,sw], bp_fx[sh,sw],
                 pw_rmse[sh,sw], lin_rmse[sh,sw], n_valid[sh,sw]],
    })
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        df_ts.to_excel(writer,    sheet_name='时间序列', index=False)
        df_param.to_excel(writer, sheet_name='拟合参数', index=False)


# ============================================================
# 辅助函数：读取单个 TIF 的空间元信息
# ============================================================

def _get_raster_extent(tif_path):
    ds = gdal.Open(tif_path)
    if ds is None:
        return None
    gt   = ds.GetGeoTransform()
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    proj = ds.GetProjection()
    ds   = None
    min_x = gt[0]
    max_y = gt[3]
    max_x = gt[0] + cols * gt[1]
    min_y = gt[3] + rows * gt[5]
    return min_x, max_x, min_y, max_y, gt[1], gt[5], proj, cols, rows


# ============================================================
# 镶嵌函数（与月份版完全一致，仅日志文字改为"季节"）
# ============================================================

def mosaic_season_results(grid_output_dirs, mosaic_output_dir, method='first'):
    """将同一季节各网格的 TCEV TIF 镶嵌为整幅影像。"""
    NODATA = -9999.0
    os.makedirs(mosaic_output_dir, exist_ok=True)
    print(f"\n  [镶嵌] 开始合并 {len(grid_output_dirs)} 个网格 → {mosaic_output_dir}")
    print(f"  [镶嵌] 重叠处理方式: {method}")

    for layer in LAYER_NAMES:
        src_tifs = []
        for gdir in grid_output_dirs:
            p = os.path.join(gdir, f'{layer}.tif')
            if os.path.exists(p):
                src_tifs.append(p)

        if not src_tifs:
            print(f"    [跳过] {layer}: 无可用TIF")
            continue

        out_path = os.path.join(mosaic_output_dir, f'{layer}.tif')
        extents  = []
        for p in src_tifs:
            ext = _get_raster_extent(p)
            if ext is not None:
                extents.append((p, ext))

        if not extents:
            print(f"    [跳过] {layer}: 无法读取空间信息")
            continue

        _, ref = extents[0]
        px_w, px_h, projection = ref[4], ref[5], ref[6]

        union_min_x = min(e[1][0] for e in extents)
        union_max_x = max(e[1][1] for e in extents)
        union_min_y = min(e[1][2] for e in extents)
        union_max_y = max(e[1][3] for e in extents)

        out_cols = int(round((union_max_x - union_min_x) / px_w))
        out_rows = int(round((union_max_y - union_min_y) / abs(px_h)))

        print(f"    {layer}: 并集范围 X=[{union_min_x:.4f}, {union_max_x:.4f}] "
              f"Y=[{union_min_y:.4f}, {union_max_y:.4f}]  "
              f"输出尺寸 {out_cols}×{out_rows}")

        if method == 'mean':
            buf_sum   = np.zeros((out_rows, out_cols), dtype=np.float64)
            buf_count = np.zeros((out_rows, out_cols), dtype=np.int32)
            buf       = None
        else:
            buf = np.full((out_rows, out_cols), np.nan, dtype=np.float32)

        for tif_path, ext in extents:
            ds = gdal.Open(tif_path)
            if ds is None:
                print(f"      警告: 无法打开 {tif_path}")
                continue
            data = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
            ds   = None
            data[data == NODATA] = np.nan

            col_off = int(round((ext[0] - union_min_x) / px_w))
            row_off = int(round((union_max_y - ext[3]) / abs(px_h)))
            tile_rows, tile_cols = data.shape

            r0 = max(row_off, 0);  c0 = max(col_off, 0)
            r1 = min(row_off + tile_rows, out_rows)
            c1 = min(col_off + tile_cols, out_cols)
            dr0 = r0 - row_off;  dc0 = c0 - col_off
            dr1 = dr0 + (r1 - r0);  dc1 = dc0 + (c1 - c0)

            if r1 <= r0 or c1 <= c0:
                print(f"      警告: {os.path.basename(tif_path)} 超出并集范围，跳过")
                continue

            tile  = data[dr0:dr1, dc0:dc1]
            valid = ~np.isnan(tile)

            if method == 'mean':
                buf_sum  [r0:r1, c0:c1][valid] += tile[valid]
                buf_count[r0:r1, c0:c1][valid] += 1
            elif method == 'first':
                dst = buf[r0:r1, c0:c1]
                dst[valid & np.isnan(dst)] = tile[valid & np.isnan(dst)]
                buf[r0:r1, c0:c1] = dst
            elif method == 'last':
                dst = buf[r0:r1, c0:c1]; dst[valid] = tile[valid]
                buf[r0:r1, c0:c1] = dst
            elif method == 'max':
                dst = buf[r0:r1, c0:c1]
                mask = valid & (np.isnan(dst) | (tile > dst))
                dst[mask] = tile[mask]; buf[r0:r1, c0:c1] = dst
            elif method == 'min':
                dst = buf[r0:r1, c0:c1]
                mask = valid & (np.isnan(dst) | (tile < dst))
                dst[mask] = tile[mask]; buf[r0:r1, c0:c1] = dst
            else:
                raise ValueError(f"不支持的 method: {method}")

        if method == 'mean':
            with np.errstate(divide='ignore', invalid='ignore'):
                buf = np.where(buf_count > 0,
                               buf_sum / buf_count, np.nan).astype(np.float32)

        buf[np.isnan(buf)] = NODATA
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(out_path, out_cols, out_rows, 1, gdal.GDT_Float32,
                               options=['COMPRESS=LZW', 'BIGTIFF=YES'])
        out_ds.SetGeoTransform((union_min_x, px_w, 0, union_max_y, 0, px_h))
        out_ds.SetProjection(projection)
        band = out_ds.GetRasterBand(1)
        band.WriteArray(buf)
        band.SetNoDataValue(NODATA)
        band.FlushCache()
        out_ds = None

        valid_n = int(np.sum(buf != NODATA))
        print(f"    ✓ {layer}.tif  "
              f"有效像元 {valid_n:,}/{buf.size:,} ({100*valid_n/buf.size:.1f}%)")

    print(f"  [镶嵌] 完成 → {mosaic_output_dir}\n")


# ============================================================
# ★ 目录结构探测：扫描 hdf5_seasonal/季节名/grid_XX
# ============================================================

def detect_seasonal_grid_tasks(base_hdf5_dir):
    """
    扫描 hdf5_seasonal/ 下的 季节名/grid_XX 两级结构。

    返回
    ----
    tasks : [(season_name, grid_name, grid_hdf5_path), ...]
            按 SEASON_ORDER × grid 编号排序
    """
    grid_pat = re.compile(r'^grid_(\d+)$')
    tasks = []

    for season_name in SEASON_ORDER:
        season_path = os.path.join(base_hdf5_dir, season_name)
        if not os.path.isdir(season_path):
            continue
        for grid_name in sorted(os.listdir(season_path),
                                key=lambda g: int(re.search(r'\d+', g).group())
                                if re.search(r'\d+', g) else 0):
            if not grid_pat.match(grid_name):
                continue
            grid_path = os.path.join(season_path, grid_name)
            if os.path.isdir(grid_path):
                tasks.append((season_name, grid_name, grid_path))

    return tasks


# ============================================================
# ★ 参考TIF查找：在 monthly_SARdata 中按季节月份搜索
# ============================================================

def _extract_date_from_filename(fname):
    stem = os.path.splitext(os.path.basename(fname))[0]
    for part in re.split(r'[_\-]', stem):
        if re.fullmatch(r'\d{8}', part):
            try:
                return datetime.strptime(part, '%Y%m%d')
            except ValueError:
                pass
    return None


def find_reference_tif_for_season(base_sar_monthly_dir, season_name, grid_name):
    """
    在 monthly_SARdata 中，按季节对应月份（SEASON_TO_MONTHS）搜索
    指定网格的参考 TIF，取最早日期的文件。

    目录结构支持：
      base_sar_monthly_dir/month_XX/grid_XX/*.tif
    """
    months = SEASON_TO_MONTHS.get(season_name, [])
    candidates = []

    for month_num in months:
        month_label = f'month_{month_num:02d}'
        grid_dir    = os.path.join(base_sar_monthly_dir, month_label, grid_name)
        if not os.path.isdir(grid_dir):
            continue
        for fname in os.listdir(grid_dir):
            if not fname.lower().endswith('.tif'):
                continue
            d = _extract_date_from_filename(fname)
            candidates.append((d or datetime.max, os.path.join(grid_dir, fname)))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    chosen_date, chosen_path = candidates[0]
    date_str = chosen_date.strftime('%Y%m%d') if chosen_date != datetime.max else '未知日期'
    print(f"    [参考TIF] {os.path.basename(chosen_path)}"
          f"  (日期: {date_str}，季节 {season_name} 共 {len(candidates)} 个候选)")
    return chosen_path


# ============================================================
# 单网格 TCEV 分析（与月份版逻辑一致）
# ============================================================

def process_grid_season(grid_hdf5_dir, reference_tif, output_dir,
                        db_threshold, n_processes=4):
    sorted_hdf5 = os.path.join(grid_hdf5_dir, 'sorted_negative_values.hdf5')
    index_hdf5  = os.path.join(grid_hdf5_dir, 'index_values.hdf5')
    for f in [sorted_hdf5, index_hdf5]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"缺少文件: {f}")

    os.makedirs(output_dir, exist_ok=True)

    (R, a1, b1, a2, b2, bp_db, bp_fx,
     pw_rmse, lin_rmse, n_valid) = process_hdf5_data_parallel(
        sorted_hdf5, index_hdf5,
        db_threshold=db_threshold,
        n_processes=n_processes)

    arrays = [R, a1, b1, a2, b2, bp_db, bp_fx, pw_rmse, lin_rmse, n_valid]
    for name, arr in zip(LAYER_NAMES, arrays):
        save_raster_as_tif(arr, os.path.join(output_dir, f'{name}.tif'), reference_tif)

    export_sample_pixel(sorted_hdf5, index_hdf5,
                        R, a1, b1, a2, b2, bp_db, bp_fx,
                        pw_rmse, lin_rmse, n_valid,
                        os.path.join(output_dir, 'sample_pixel.xlsx'))


# ============================================================
# ★ 批量处理主函数（月份版 → 季节版）
# ============================================================

def batch_tcev_seasonal_grids(base_hdf5_dir,
                               base_sar_monthly_dir,
                               base_output_dir,
                               db_threshold,
                               season_filter=None,
                               grid_filter=None,
                               n_processes=4,
                               mosaic_method='first'):
    """
    遍历所有 季节名 × grid_XX，执行 TCEV 分析，完成后对每季节做镶嵌。

    参数
    ----
    base_hdf5_dir        : hdf5_seasonal/ 根目录（含 spring/summer/autumn/winter 子目录）
    base_sar_monthly_dir : monthly_SARdata/ 根目录（含 month_XX/grid_XX 子目录，用于参考TIF）
    base_output_dir      : TCEV 结果输出根目录
    db_threshold         : DB 值上限阈值（float）。超过此值的点将被剔除后再拟合。
                           设为 np.inf 可关闭此过滤。
    season_filter        : 仅处理指定季节，字符串或列表，None=全部四季
                           示例: 'autumn'  或  ['spring', 'summer']
    grid_filter          : 仅处理指定网格，字符串或列表，None=全部
    n_processes          : 每个网格任务的并行进程数
    mosaic_method        : 镶嵌重叠处理方式 'first'|'mean'|'max'|'min'|'last'
    """
    if not os.path.exists(base_hdf5_dir):
        raise FileNotFoundError(f"HDF5目录不存在: {base_hdf5_dir}")
    if not os.path.exists(base_sar_monthly_dir):
        raise FileNotFoundError(f"SAR monthly目录不存在: {base_sar_monthly_dir}")

    # ── 扫描所有任务 ──────────────────────────────────────────────────
    all_tasks = detect_seasonal_grid_tasks(base_hdf5_dir)
    if not all_tasks:
        raise ValueError(f"在 {base_hdf5_dir} 中未找到 季节名/grid_XX 结构")

    # ── 过滤 ──────────────────────────────────────────────────────────
    if season_filter is not None:
        if isinstance(season_filter, str):
            season_filter = [season_filter]
        # 校验输入的季节名是否合法
        invalid = [s for s in season_filter if s not in SEASON_ORDER]
        if invalid:
            raise ValueError(f"不合法的季节名: {invalid}，合法值为 {SEASON_ORDER}")
        all_tasks = [t for t in all_tasks if t[0] in season_filter]

    if grid_filter is not None:
        if isinstance(grid_filter, str):
            grid_filter = [grid_filter]
        all_tasks = [t for t in all_tasks if t[1] in grid_filter]

    if not all_tasks:
        print("过滤后无任务，请检查 season_filter / grid_filter。")
        return

    season_to_tasks = defaultdict(list)
    for t in all_tasks:
        season_to_tasks[t[0]].append(t)

    threshold_str = str(db_threshold) if not np.isinf(db_threshold) else "无限制（关闭）"

    print("=" * 70)
    print("批量 TCEV 分析：季节 × 网格  +  季节镶嵌")
    print("=" * 70)
    print(f"HDF5 根目录   : {base_hdf5_dir}")
    print(f"SAR月份目录   : {base_sar_monthly_dir}")
    print(f"输出根目录    : {base_output_dir}")
    print(f"DB 阈值过滤   : DB > {threshold_str} 的点将被剔除")
    print(f"总任务数      : {len(all_tasks)}  ({len(season_to_tasks)} 个季节)")
    print(f"并行进程数    : {n_processes}")
    print(f"镶嵌方式      : {mosaic_method}")
    print("-" * 70)
    for s in SEASON_ORDER:
        if s in season_to_tasks:
            grids = [t[1] for t in season_to_tasks[s]]
            months_str = str(SEASON_TO_MONTHS[s])
            print(f"  {s:6s} (月份{months_str}): {grids}")
    print("=" * 70 + "\n")

    success_list, failed_list = [], []
    season_success_dirs = defaultdict(list)

    total = len(all_tasks)
    for idx, (season_label, grid_label, grid_hdf5_path) in enumerate(all_tasks, 1):
        tag = f"{season_label}/{grid_label}"
        print(f"\n{'#'*70}")
        print(f"# [{idx:>3}/{total}]  {tag}")
        print(f"{'#'*70}")

        try:
            # ── 在 monthly_SARdata 中按季节月份找参考TIF ────────────
            ref_tif = find_reference_tif_for_season(
                base_sar_monthly_dir, season_label, grid_label)
            if ref_tif is None:
                raise FileNotFoundError(
                    f"在 {base_sar_monthly_dir}/month_XX/{grid_label} "
                    f"中找不到季节 '{season_label}' 对应月份 "
                    f"{SEASON_TO_MONTHS[season_label]} 的任何TIF文件。"
                )

            output_dir = os.path.join(base_output_dir, season_label, grid_label)
            print(f"  HDF5 目录 : {grid_hdf5_path}")
            print(f"  输出目录  : {output_dir}")

            done_tifs = [os.path.join(output_dir, f'{n}.tif') for n in LAYER_NAMES]
            if all(os.path.exists(p) for p in done_tifs):
                success_list.append(tag)
                season_success_dirs[season_label].append(output_dir)
                print(f"  ⏭ 已存在，跳过计算")
            else:
                process_grid_season(
                    grid_hdf5_dir = grid_hdf5_path,
                    reference_tif = ref_tif,
                    output_dir    = output_dir,
                    db_threshold  = db_threshold,
                    n_processes   = n_processes,
                )
                success_list.append(tag)
                season_success_dirs[season_label].append(output_dir)
                print(f"  ✓ 完成")

        except Exception as e:
            failed_list.append((tag, str(e)))
            print(f"  ✗ 失败: {e}")
            traceback.print_exc()

        # ── 当季所有网格处理完后立即触发镶嵌 ────────────────────────
        expected_grids = [t[1] for t in season_to_tasks[season_label]]
        done_grids     = [t[1] for t in all_tasks[:idx] if t[0] == season_label]
        if set(done_grids) == set(expected_grids):
            if season_success_dirs[season_label]:
                mosaic_dir  = os.path.join(base_output_dir, season_label, 'mosaic')
                mosaic_tifs = [os.path.join(mosaic_dir, f'{n}.tif') for n in LAYER_NAMES]
                if all(os.path.exists(p) for p in mosaic_tifs):
                    print(f"\n  [镶嵌跳过] {season_label}/mosaic 已存在\n")
                else:
                    sorted_dirs = sorted(
                        season_success_dirs[season_label],
                        key=lambda p: int(re.search(r'\d+', os.path.basename(p)).group())
                        if re.search(r'\d+', os.path.basename(p)) else 0
                    )
                    mosaic_season_results(sorted_dirs, mosaic_dir, method=mosaic_method)
            else:
                print(f"\n  [镶嵌跳过] {season_label}: 当季无成功网格\n")

    print("\n" + "=" * 70)
    print("全部完成！")
    print("=" * 70)
    print(f"成功: {len(success_list)}/{len(all_tasks)}")
    if failed_list:
        print(f"\n失败任务 ({len(failed_list)} 个):")
        for tag, err in failed_list:
            print(f"  - {tag}: {err}")
    else:
        print("所有任务均成功！")
    print("=" * 70)


# ============================================================
# 入口 —— 所有路径与参数在此统一定义
# ============================================================

if __name__ == '__main__':

    # ── 路径配置 ─────────────────────────────────────────────────────
    BASE_HDF5        = r'/home/data/ql2024/flood-tly/australia/data/hdf5_seasons'
    BASE_SAR_MONTHLY = r'/home/data/ql2024/flood-tly/australia/data/monthly_SARdata'
    BASE_OUTPUT      = r'/home/data/ql2024/flood-tly/australia/data/tcev_seasonal'

    # ── DB 阈值过滤 ───────────────────────────────────────────────────
    DB_THRESHOLD = 30.0   # 设为 np.inf 可关闭过滤

    # ── 任务过滤 ──────────────────────────────────────────────────────
    # None = 处理全部四季；字符串或列表 = 仅处理指定季节
    # 合法值：'spring' | 'summer' | 'autumn' | 'winter'
    # 示例:  SEASON_FILTER = 'autumn'
    #        SEASON_FILTER = ['spring', 'autumn']
    SEASON_FILTER = 'spring'
    GRID_FILTER   = None

    # ── 并行与镶嵌配置 ────────────────────────────────────────────────
    N_PROCESSES   = 2
    MOSAIC_METHOD = 'mean'   # 无缓冲区重叠推荐 'first'；有重叠推荐 'mean'

    # ── 启动 ──────────────────────────────────────────────────────────
    batch_tcev_seasonal_grids(
        base_hdf5_dir        = BASE_HDF5,
        base_sar_monthly_dir = BASE_SAR_MONTHLY,
        base_output_dir      = BASE_OUTPUT,
        db_threshold         = DB_THRESHOLD,
        season_filter        = SEASON_FILTER,
        grid_filter          = GRID_FILTER,
        n_processes          = N_PROCESSES,
        mosaic_method        = MOSAIC_METHOD,
    )