#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TabPFN 洪水栅格预测脚本 v3（多子网格 · 分辨率重采样 · 模型持久化 · 效率优化）
=============================================================================

新增 / 改动：
  ① TARGET_RESOLUTION_M  — 指定推理与输出分辨率（单位：米），自动判断 CRS 单位
                           None = 保持原始分辨率；设为 30 = 30 m 重采样
  ② MODEL_SAVE_PATH      — fit 后把模型序列化到磁盘
     LOAD_MODEL_PATH      — 非 None 时直接加载已有模型，跳过训练（适用于其他研究区）
  ③ 效率提升：
     · Numba JIT 预热（避免第一次调用延迟）
     · GDAL VSI 内存重采样（不落盘，速度快）
     · 推理改用 multiprocessing Pool，绕过 GIL（仅 predict_proba 并行）
     · canvas 填充改为逐网格向量化写入（原逻辑保持，去掉冗余判断）
     · 所有 float 运算统一 float32

目录结构（输入）
  tcev_base/autumn/  grid_01/ R_index.tif  L1_alpha.tif … breakpoint_FX.tif
  hdf5_base/autumn/  grid_01/ sorted_negative_values.hdf5

依赖
  pip install numpy h5py gdal tabpfn pandas numba torch joblib
"""

import os
import io
import re
import sys
import time
import pickle
import warnings
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import h5py
from numba import jit, prange
from osgeo import gdal, ogr, osr

warnings.filterwarnings('ignore')
gdal.UseExceptions()


# ══════════════════════════════════════════════════════════════
# ▌用户配置区
# ══════════════════════════════════════════════════════════════

TCEV_BASE   = r'/home/data/ql2024/flood-tly/australia/data/tcev_seasonal/spring'
HDF5_BASE   = r'/home/data/ql2024/flood-tly/australia/data/hdf5_seasons/spring'
POINTS_SHP  = r'/home/data/ql2024/flood-tly/australia/data/tcev/label.shp'
OUTPUT_DIR  = r'/home/data/ql2024/flood-tly/australia/data/flood_result_30m'

# 本地 TabPFN 权重目录（已从 HuggingFace 下载）
TABPFN_WEIGHTS_DIR = r'/home/data/ql2024/tabpfn_2_6'

LABEL_FIELD = 'id'
LABEL_MAP   = {1: 0, 2: 1}          # 字段值 → 内部标签（0=非洪水，1=洪水）

THRESHOLD      = 0.5
BATCH_SIZE     = 10000             # 单批像元数
N_LOAD_WORKERS = 2                   # 并行加载网格的线程数
N_INFER_WORKERS = 2                  # 并行推理 Worker 数（进程池）

# ──────────────────────────────────────────────────────────────
# ① 分辨率设置
#   · None  → 保持原始像素尺寸，不重采样
#   · 数值  → 目标分辨率（单位：米）
#             程序自动判断 CRS 单位：地理坐标系(°) 按纬度近似换算，投影坐标系直接使用
# ──────────────────────────────────────────────────────────────
TARGET_RESOLUTION_M = 30           # 例：30  表示 30 m；None 不重采样

# ──────────────────────────────────────────────────────────────
# ② 模型持久化
#   MODEL_SAVE_PATH  — fit 后自动保存模型（None = 不保存）
#   LOAD_MODEL_PATH  — 直接加载已有模型（None = 重新训练）
#     当 LOAD_MODEL_PATH 非 None 时，仍需训练点来验证数据完整性；
#     若只做推理无需标签，可将 POINTS_SHP 留空并跳过 Step 3。
# ──────────────────────────────────────────────────────────────
#MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, 'tabpfn_model.pkl')  # None = 不保存
MODEL_SAVE_PATH = None  # None = 不保存
LOAD_MODEL_PATH = '/home/data/ql2024/flood-tly/guangxi/data/tcev/flood_result_all_100m/tabpfn_model.pkl'   # 例：r'/other_area/tabpfn_model.pkl'

# ══════════════════════════════════════════════════════════════
# 常量
# ══════════════════════════════════════════════════════════════
TCEV_LAYERS   = [
    'R_index', 'L1_alpha', 'L1_beta',
    'L2_alpha', 'L2_beta', 'breakpoint_DB', 'breakpoint_FX'
]
FEATURE_NAMES = TCEV_LAYERS + ['db_range', 'top10_mean']
NODATA_OUT    = -9999.0
GRID_PAT      = re.compile(r'^grid_\d+$')
N_FEATURES    = len(FEATURE_NAMES)     # 9


# ══════════════════════════════════════════════════════════════
# HDF5 衍生特征（Numba 并行版，cache=True 只编译一次）
# ══════════════════════════════════════════════════════════════

@jit(nopython=True, parallel=True, cache=True)
def _extra_features_numba(sd):
    T, H, W = sd.shape
    db  = np.full((H, W), np.nan, dtype=np.float32)
    t10 = np.full((H, W), np.nan, dtype=np.float32)
    for h in prange(H):
        for w in range(W):
            nv = 0
            for t in range(T):
                if not np.isnan(sd[t, h, w]):
                    nv += 1
            if nv < 2:
                continue
            fv = np.nan; lv = np.nan; seen = 0
            for t in range(T):
                v = sd[t, h, w]
                if not np.isnan(v):
                    if seen == 0:
                        fv = v
                    lv = v; seen += 1
            db[h, w] = lv - fv
            start = int(0.9 * nv)
            tot = 0.0; cnt = 0; vs = 0
            for t in range(T):
                v = sd[t, h, w]
                if not np.isnan(v):
                    if vs >= start:
                        tot += v; cnt += 1
                    vs += 1
            if cnt > 0:
                t10[h, w] = tot / cnt
    return db, t10


def warmup_numba():
    """预热 Numba JIT，避免第一次真实调用时的编译延迟（~10s）"""
    dummy = np.random.rand(3, 4, 4).astype(np.float32)
    dummy[dummy < 0.2] = np.nan
    _extra_features_numba(dummy)
    print("  ✓ Numba JIT 预热完成")


def load_hdf5_features(hdf5_dir):
    p = os.path.join(hdf5_dir, 'sorted_negative_values.hdf5')
    if not os.path.exists(p):
        raise FileNotFoundError(f"缺少 HDF5: {p}")
    with h5py.File(p, 'r') as hf:
        data = hf['data'][:]
    return _extra_features_numba(data)   # (H,W), (H,W)


# ══════════════════════════════════════════════════════════════
# 单个网格特征立方体读取
# ══════════════════════════════════════════════════════════════

def load_grid_cube(tcev_dir, hdf5_dir):
    """返回 (cube[H,W,N_FEATURES], geotransform, projection) 或 None"""
    tcev_arrs = {}
    grid_gt = None; grid_proj = None
    for layer in TCEV_LAYERS:
        p = os.path.join(tcev_dir, f'{layer}.tif')
        if not os.path.exists(p):
            return None
        ds   = gdal.Open(p)
        arr  = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
        nd   = ds.GetRasterBand(1).GetNoDataValue()
        gt   = ds.GetGeoTransform()
        proj = ds.GetProjection()
        ds   = None
        if nd is not None:
            arr[arr == nd] = np.nan
        tcev_arrs[layer] = arr
        if grid_gt is None:
            grid_gt = gt; grid_proj = proj

    try:
        db_range, top10 = load_hdf5_features(hdf5_dir)
    except FileNotFoundError:
        return None

    H = min(tcev_arrs['R_index'].shape[0], db_range.shape[0], top10.shape[0])
    W = min(tcev_arrs['R_index'].shape[1], db_range.shape[1], top10.shape[1])
    for k in tcev_arrs:
        tcev_arrs[k] = tcev_arrs[k][:H, :W]
    db_range = db_range[:H, :W]; top10 = top10[:H, :W]

    cube = np.full((H, W, N_FEATURES), np.nan, dtype=np.float32)
    for i, layer in enumerate(TCEV_LAYERS):
        cube[:, :, i] = tcev_arrs[layer]
    cube[:, :, 7] = db_range
    cube[:, :, 8] = top10
    return cube, grid_gt, grid_proj


def all_grid_names(base_dir):
    return sorted(
        n for n in os.listdir(base_dir)
        if GRID_PAT.match(n) and os.path.isdir(os.path.join(base_dir, n))
    )


# ══════════════════════════════════════════════════════════════
# 并行加载所有网格
# ══════════════════════════════════════════════════════════════

def load_all_grids(tcev_base, hdf5_base, grid_names, n_workers=N_LOAD_WORKERS):
    results = {}

    def _load(gname):
        td = os.path.join(tcev_base, gname)
        hd = os.path.join(hdf5_base, gname)
        return gname, load_grid_cube(td, hd)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_load, g): g for g in grid_names}
        done = 0
        for fut in as_completed(futs):
            gname, res = fut.result()
            results[gname] = res
            done += 1
            status = '✓' if res is not None else '✗ 缺失'
            print(f"\r  加载网格 [{done}/{len(grid_names)}] {gname} {status}      ",
                  end='')
    print()
    return results


# ══════════════════════════════════════════════════════════════
# ① 分辨率工具函数
# ══════════════════════════════════════════════════════════════

def crs_is_geographic(proj_wkt: str) -> bool:
    """判断投影是否为地理坐标系（单位：度）"""
    srs = osr.SpatialReference()
    srs.ImportFromWkt(proj_wkt)
    return bool(srs.IsGeographic())


def meters_to_pixel_size(target_m: float, proj_wkt: str,
                         center_lat: float = 25.0) -> float:
    """
    将目标分辨率（米）转换为该 CRS 的像素单位。
    - 地理坐标系（°）：纬度方向 1°≈111320 m，不随纬度变化；
      此处统一用纬度等效（推荐投影坐标系使用）
    - 投影坐标系（m）：直接返回 target_m
    注：地理坐标系下建议先投影再操作；此处提供近似换算供快速使用。
    """
    if crs_is_geographic(proj_wkt):
        # 1° 纬度 ≈ 111320 m（不受纬度影响）
        deg = target_m / 111320.0
        print(f"  [分辨率] 地理坐标系：{target_m} m ≈ {deg:.6f}°（按纬度方向换算）")
        return deg
    else:
        print(f"  [分辨率] 投影坐标系（m）：直接使用 {target_m} m")
        return float(target_m)


def resample_canvas_gdal(canvas: np.ndarray,
                         gt: tuple,
                         proj: str,
                         target_m: float) -> tuple:
    """
    用 GDAL 内存 VSI 对画布进行双线性重采样。
    返回 (resampled_canvas[H',W',C], new_gt)
    不写磁盘，速度快。
    """
    H, W, C = canvas.shape
    new_px = meters_to_pixel_size(target_m, proj)
    new_py = -new_px          # 负值（向下）

    # 计算新尺寸
    x_extent = W * gt[1]     # gt[1] > 0
    y_extent = H * abs(gt[5])
    new_W = max(1, int(round(x_extent / new_px)))
    new_H = max(1, int(round(y_extent / new_px)))
    new_gt = (gt[0], new_px, 0, gt[3], 0, new_py)

    mem_drv = gdal.GetDriverByName('MEM')

    # 源数据集（各波段 = 各特征）
    src_ds = mem_drv.Create('', W, H, C, gdal.GDT_Float32)
    src_ds.SetGeoTransform(gt)
    src_ds.SetProjection(proj)
    for b in range(C):
        band = src_ds.GetRasterBand(b + 1)
        layer = canvas[:, :, b].copy()
        layer[np.isnan(layer)] = NODATA_OUT
        band.WriteArray(layer)
        band.SetNoDataValue(NODATA_OUT)

    # 目标数据集
    dst_ds = mem_drv.Create('', new_W, new_H, C, gdal.GDT_Float32)
    dst_ds.SetGeoTransform(new_gt)
    dst_ds.SetProjection(proj)
    for b in range(C):
        dst_ds.GetRasterBand(b + 1).SetNoDataValue(NODATA_OUT)

    gdal.ReprojectImage(
        src_ds, dst_ds, proj, proj,
        gdal.GRA_Bilinear   # 双线性插值；分类特征可改 GRA_NearestNeighbour
    )

    # 读回 numpy
    new_canvas = np.full((new_H, new_W, C), np.nan, dtype=np.float32)
    for b in range(C):
        arr = dst_ds.GetRasterBand(b + 1).ReadAsArray().astype(np.float32)
        arr[arr == NODATA_OUT] = np.nan
        new_canvas[:, :, b] = arr

    src_ds = dst_ds = None

    old_mem = H * W * C * 4 / 1024 / 1024
    new_mem = new_H * new_W * C * 4 / 1024 / 1024
    print(f"  重采样: {H}×{W} → {new_H}×{new_W}  "
          f"({old_mem:.0f} MB → {new_mem:.0f} MB)")
    return new_canvas, new_gt


# ══════════════════════════════════════════════════════════════
# ② 模型持久化
# ══════════════════════════════════════════════════════════════

def save_model(clf, path: str):
    """将训练好的 TabPFN 模型序列化到磁盘（pickle）"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(clf, f, protocol=5)
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  ✓ 模型已保存 → {path}  ({size_mb:.1f} MB)")


def load_model(path: str):
    """从磁盘加载已有 TabPFN 模型，直接用于推理"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"模型文件不存在: {path}")
    with open(path, 'rb') as f:
        clf = pickle.load(f)
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  ✓ 模型已加载 ← {path}  ({size_mb:.1f} MB)")
    return clf


# ══════════════════════════════════════════════════════════════
# 空间工具
# ══════════════════════════════════════════════════════════════

def raster_bbox(gt, cols, rows):
    minX = gt[0];  maxX = gt[0] + cols * gt[1]
    maxY = gt[3];  minY = gt[3] + rows * gt[5]
    return minX, maxX, minY, maxY


def transform_point(x, y, src_wkt, dst_wkt):
    src = osr.SpatialReference(); src.ImportFromWkt(src_wkt)
    dst = osr.SpatialReference(); dst.ImportFromWkt(dst_wkt)
    if src.IsSame(dst):
        return x, y
    t = osr.CoordinateTransformation(src, dst)
    pt = t.TransformPoint(x, y)
    return pt[0], pt[1]


# ══════════════════════════════════════════════════════════════
# 训练点采样（向量化版）
# ══════════════════════════════════════════════════════════════

def sample_training_points(xs, ys, pt_proj, cube_cache):
    n = len(xs)
    X = np.full((n, N_FEATURES), np.nan, dtype=np.float32)
    xs_arr  = np.asarray(xs, np.float64)
    ys_arr  = np.asarray(ys, np.float64)
    assigned = np.zeros(n, dtype=bool)

    for gname, res in cube_cache.items():
        if res is None:
            continue
        cube, g_gt, g_proj = res
        H, W, _ = cube.shape

        try:
            src = osr.SpatialReference(); src.ImportFromWkt(pt_proj)
            dst = osr.SpatialReference(); dst.ImportFromWkt(g_proj)
            if src.IsSame(dst):
                xg, yg = xs_arr.copy(), ys_arr.copy()
            else:
                t = osr.CoordinateTransformation(src, dst)
                pts = np.array([t.TransformPoint(x, y)
                                for x, y in zip(xs_arr, ys_arr)])
                xg, yg = pts[:, 0], pts[:, 1]
        except Exception:
            xg, yg = xs_arr.copy(), ys_arr.copy()

        bbox = raster_bbox(g_gt, W, H)
        in_box = (
            (xg >= bbox[0]) & (xg <= bbox[1]) &
            (yg >= bbox[2]) & (yg <= bbox[3]) &
            (~assigned)
        )
        if not in_box.any():
            continue

        cols_idx = ((xg[in_box] - g_gt[0]) / g_gt[1]).astype(int)
        rows_idx = ((yg[in_box] - g_gt[3]) / g_gt[5]).astype(int)
        valid_rc = (rows_idx >= 0) & (rows_idx < H) & (cols_idx >= 0) & (cols_idx < W)
        idx_all  = np.where(in_box)[0]
        for k, (r, c, ok) in enumerate(zip(rows_idx, cols_idx, valid_rc)):
            if ok:
                X[idx_all[k]] = cube[r, c, :]
                assigned[idx_all[k]] = True

    hit = int(assigned.sum())
    print(f"  采样结果: {hit}/{n} 个点成功命中网格并取到特征")
    return X


# ══════════════════════════════════════════════════════════════
# 全量预测画布构建
# ══════════════════════════════════════════════════════════════

def build_full_canvas(cube_cache):
    valid_grids = {g: r for g, r in cube_cache.items() if r is not None}
    if not valid_grids:
        raise RuntimeError("所有网格均加载失败，无法构建预测画布。")

    ref_name        = next(iter(valid_grids))
    _, ref_gt, ref_proj = valid_grids[ref_name]
    px_w = ref_gt[1]; px_h = ref_gt[5]

    all_minX, all_maxX = [], []
    all_minY, all_maxY = [], []

    src_crs = osr.SpatialReference()
    dst_crs = osr.SpatialReference(); dst_crs.ImportFromWkt(ref_proj)

    for gname, res in valid_grids.items():
        cube, g_gt, g_proj = res
        H, W, _ = cube.shape
        g_bbox = raster_bbox(g_gt, W, H)
        src_crs.ImportFromWkt(g_proj)
        if src_crs.IsSame(dst_crs):
            b = g_bbox
        else:
            t = osr.CoordinateTransformation(src_crs, dst_crs)
            corners = [(g_bbox[0], g_bbox[2]), (g_bbox[0], g_bbox[3]),
                       (g_bbox[1], g_bbox[2]), (g_bbox[1], g_bbox[3])]
            pts = [t.TransformPoint(x, y) for x, y in corners]
            b = (min(p[0] for p in pts), max(p[0] for p in pts),
                 min(p[1] for p in pts), max(p[1] for p in pts))
        all_minX.append(b[0]); all_maxX.append(b[1])
        all_minY.append(b[2]); all_maxY.append(b[3])

    global_minX = min(all_minX); global_maxX = max(all_maxX)
    global_minY = min(all_minY); global_maxY = max(all_maxY)

    def snap(val, origin, step):
        return origin + round((val - origin) / step) * step

    minX = snap(global_minX, ref_gt[0], px_w)
    maxX = snap(global_maxX, ref_gt[0], px_w)
    maxY = snap(global_maxY, ref_gt[3], px_h)
    minY = snap(global_minY, ref_gt[3], px_h)

    out_cols = max(1, int(round((maxX - minX) / px_w)))
    out_rows = max(1, int(round((maxY - minY) / abs(px_h))))
    canvas_gt = (minX, px_w, 0, maxY, 0, px_h)

    total_px = out_rows * out_cols
    mem_mb   = total_px * N_FEATURES * 4 / 1024 / 1024
    print(f"  全局画布: {out_rows} 行 × {out_cols} 列  "
          f"（≈{mem_mb:.0f} MB，{len(valid_grids)} 个网格）")

    canvas   = np.full((out_rows, out_cols, N_FEATURES), np.nan, dtype=np.float32)
    src_crs2 = osr.SpatialReference()

    for gname, res in valid_grids.items():
        cube, g_gt, g_proj = res
        H, W, _ = cube.shape
        src_crs2.ImportFromWkt(g_proj)
        if src_crs2.IsSame(dst_crs):
            ox, oy = g_gt[0], g_gt[3]
        else:
            t = osr.CoordinateTransformation(src_crs2, dst_crs)
            pt = t.TransformPoint(g_gt[0], g_gt[3])
            ox, oy = pt[0], pt[1]

        col_off = int(round((ox   - minX)  / px_w))
        row_off = int(round((maxY - oy)    / abs(px_h)))
        r0 = max(row_off, 0);            c0 = max(col_off, 0)
        r1 = min(row_off + H, out_rows); c1 = min(col_off + W, out_cols)
        sr0 = r0 - row_off;              sc0 = c0 - col_off
        sr1 = sr0 + (r1 - r0);          sc1 = sc0 + (c1 - c0)
        if r1 <= r0 or c1 <= c0:
            print(f"    [{gname}] 偏移后无有效区域，跳过")
            continue

        tile  = cube[sr0:sr1, sc0:sc1, :]
        # 向量化：仅写入目标区域当前全NaN且源有效的像元
        dst_slice  = canvas[r0:r1, c0:c1, :]
        src_valid  = ~np.isnan(tile).all(axis=2)    # (h,w)
        dst_empty  = np.isnan(dst_slice).all(axis=2) # (h,w)
        fill_mask  = src_valid & dst_empty           # (h,w)
        dst_slice[fill_mask] = tile[fill_mask]
        canvas[r0:r1, c0:c1, :] = dst_slice
        print(f"    [{gname}] 填充像元={int(fill_mask.sum()):,}  "
              f"画布位置=[{r0}:{r1}, {c0}:{c1}]")

    nan_rate = 100 * np.isnan(canvas).all(axis=2).mean()
    print(f"  画布全NaN像元率={nan_rate:.1f}%")
    return canvas, canvas_gt, ref_proj


# ══════════════════════════════════════════════════════════════
# 点矢量读取
# ══════════════════════════════════════════════════════════════

def load_points_shp(shp_path, label_field, label_map):
    ds  = ogr.Open(shp_path)
    if ds is None:
        raise IOError(f"无法打开: {shp_path}")
    lyr  = ds.GetLayer()
    defn = lyr.GetLayerDefn()
    fields = [defn.GetFieldDefn(i).GetName() for i in range(defn.GetFieldCount())]
    if label_field not in fields:
        raise ValueError(
            f"点矢量中找不到字段 '{label_field}'。\n"
            f"  已有字段: {fields}\n"
            f"  请修改脚本顶部的 LABEL_FIELD。"
        )
    xs, ys, labels = [], [], []
    for feat in lyr:
        geom = feat.GetGeometryRef()
        if geom is None:
            continue
        val = feat.GetField(label_field)
        if val not in label_map:
            print(f"  [警告] FID={feat.GetFID()} {label_field}={val} "
                  f"不在映射表 {list(label_map.keys())}，跳过")
            continue
        xs.append(geom.GetX()); ys.append(geom.GetY())
        labels.append(label_map[int(val)])
    pt_proj = lyr.GetSpatialRef().ExportToWkt() if lyr.GetSpatialRef() else ''
    ds = None
    xs = np.array(xs, np.float64); ys = np.array(ys, np.float64)
    labels = np.array(labels, np.int8)
    print(f"  洪水(→1)={int((labels==1).sum())}  "
          f"非洪水(→0)={int((labels==0).sum())}  合计={len(labels)}")
    return xs, ys, labels, pt_proj


# ══════════════════════════════════════════════════════════════
# 栅格输出工具
# ══════════════════════════════════════════════════════════════

def save_tif(data, path, gt, proj, nodata=NODATA_OUT):
    r, c = data.shape
    ds   = gdal.GetDriverByName('GTiff').Create(
        path, c, r, 1, gdal.GDT_Float32,
        options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=IF_SAFER'])
    ds.SetGeoTransform(gt); ds.SetProjection(proj)
    arr = data.astype(np.float32); arr[np.isnan(arr)] = nodata
    b   = ds.GetRasterBand(1)
    b.WriteArray(arr); b.SetNoDataValue(nodata); b.FlushCache()
    ds  = None


# ══════════════════════════════════════════════════════════════
# TabPFN 本地初始化
# ══════════════════════════════════════════════════════════════

def build_classifier():
    import torch
    os.environ['HF_HUB_OFFLINE']   = '1'
    os.environ['TABPFN_CACHE_DIR'] = TABPFN_WEIGHTS_DIR
    torch.set_num_threads(max(1, os.cpu_count() // N_INFER_WORKERS))

    try:
        from tabpfn import TabPFNClassifier
    except ImportError:
        print("  未安装 tabpfn，请执行: python -m pip install tabpfn")
        sys.exit(1)

    clf = TabPFNClassifier(device='cpu')
    print("  ✓ TabPFN 本地权重加载完成（CPU 模式）")
    return clf


# ══════════════════════════════════════════════════════════════
# ③ 高效并行推理（多线程 + 预分配输出数组）
#    TabPFN 的 predict_proba 内部使用 PyTorch，会释放 GIL，
#    多线程并行真实有效。
# ══════════════════════════════════════════════════════════════

def parallel_infer(clf, X_all: np.ndarray,
                   n_workers: int = N_INFER_WORKERS,
                   batch_size: int = BATCH_SIZE) -> np.ndarray:
    """
    多线程分批推理，返回 prob_all (n_px,) float32。
    每个线程处理轮询分配的批次列表，避免负载不均。
    """
    n_total    = len(X_all)
    prob_all   = np.empty(n_total, dtype=np.float32)
    done_count = [0]
    lock       = threading.Lock()

    # 生成批次
    batch_ranges = [
        (s, min(s + batch_size, n_total))
        for s in range(0, n_total, batch_size)
    ]
    # 轮询分配给各 Worker
    worker_tasks = [[] for _ in range(n_workers)]
    for i, rng in enumerate(batch_ranges):
        worker_tasks[i % n_workers].append(rng)

    def _infer(tasks):
        for s, e in tasks:
            prob_all[s:e] = clf.predict_proba(X_all[s:e])[:, 1]
            with lock:
                done_count[0] += (e - s)
                pct = 100 * done_count[0] / n_total
                bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
                print(f"\r  [{bar}] {pct:5.1f}%  "
                      f"已处理 {done_count[0]:,}/{n_total:,}",
                      end='', flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = [pool.submit(_infer, tasks) for tasks in worker_tasks]
        for fut in as_completed(futs):
            fut.result()   # 捕获线程内异常
    print(f"\n  推理耗时: {time.time()-t0:.1f}s")
    return prob_all


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    SEP = "=" * 70
    print(SEP)
    print("TabPFN 洪水栅格预测 v3（分辨率重采样 · 模型持久化 · 效率优化）")
    print(SEP)

    # ── 预热 Numba JIT ───────────────────────────────────────
    print("\n【预热】Numba JIT 编译（只发生一次）")
    warmup_numba()

    # ── Step 1  读取训练点 ───────────────────────────────────
    skip_train = (LOAD_MODEL_PATH is not None)
    if not skip_train:
        print(f"\n【Step 1】读取训练点矢量")
        xs, ys, labels, pt_proj = load_points_shp(
            POINTS_SHP, LABEL_FIELD, LABEL_MAP)
    else:
        print(f"\n【Step 1】已指定 LOAD_MODEL_PATH，跳过训练点读取（直接推理模式）")
        xs = ys = labels = None; pt_proj = ''

    # ── Step 2  并行加载所有网格 ─────────────────────────────
    grid_names = all_grid_names(TCEV_BASE)
    if not grid_names:
        raise RuntimeError(f"在 {TCEV_BASE} 下未找到任何 grid_XX 子目录")
    print(f"\n【Step 2】并行加载 {len(grid_names)} 个子网格（{N_LOAD_WORKERS} 线程）")
    cube_cache = load_all_grids(TCEV_BASE, HDF5_BASE, grid_names, N_LOAD_WORKERS)
    n_ok  = sum(1 for v in cube_cache.values() if v is not None)
    n_bad = len(cube_cache) - n_ok
    print(f"  成功={n_ok}  失败/缺失={n_bad}")
    if n_ok == 0:
        raise RuntimeError("所有网格均加载失败！")

    # ── Step 3  训练点采样 & 训练 or 加载模型 ────────────────
    print(f"\n【Step 3】{'加载已有模型' if skip_train else '训练点采样 + TabPFN fit'}")

    if skip_train:
        # ── ② 直接加载已有模型 ──────────────────────────────
        clf = load_model(LOAD_MODEL_PATH)
    else:
        X_raw  = sample_training_points(xs, ys, pt_proj, cube_cache)
        valid  = ~np.isnan(X_raw).any(axis=1)
        X_tr, y_tr = X_raw[valid], labels[valid]
        n_drop = int((~valid).sum())
        if n_drop:
            print(f"  [提示] {n_drop} 个点因落在网格外或 NaN 像元被丢弃")

        n_pos = int((y_tr == 1).sum()); n_neg = int((y_tr == 0).sum())
        print(f"  有效训练样本: 洪水={n_pos}  非洪水={n_neg}  合计={len(y_tr)}")
        if n_pos == 0 or n_neg == 0:
            raise RuntimeError(
                "有效训练样本缺少某一类别！\n"
                "  常见原因：坐标系/范围不匹配，或特征全为 NaN。\n"
                f"  当前 LABEL_MAP={LABEL_MAP}"
            )
        if len(y_tr) < 6:
            raise RuntimeError(f"有效样本仅 {len(y_tr)} 个，太少。")

        csv_path = os.path.join(OUTPUT_DIR, 'training_samples.csv')
        pd.DataFrame(X_tr, columns=FEATURE_NAMES).assign(label=y_tr) \
          .to_csv(csv_path, index=False)
        print(f"  训练集 CSV → {csv_path}")

        clf = build_classifier()
        print("  正在 fit …")
        t_fit = time.time()
        clf.fit(X_tr, y_tr)
        print(f"  ✓ fit 完成（{time.time()-t_fit:.1f}s）")

        # ── ② 保存模型 ──────────────────────────────────────
        if MODEL_SAVE_PATH:
            save_model(clf, MODEL_SAVE_PATH)

    # ── Step 4  构建全量预测画布 ─────────────────────────────
    print(f"\n【Step 4】拼合所有网格为全量预测画布")
    canvas, canvas_gt, canvas_proj = build_full_canvas(cube_cache)

    # ── ① 分辨率重采样（可选）──────────────────────────────
    if TARGET_RESOLUTION_M is not None:
        print(f"\n【Step 4b】将画布重采样到 {TARGET_RESOLUTION_M} m")
        canvas, canvas_gt = resample_canvas_gdal(
            canvas, canvas_gt, canvas_proj, TARGET_RESOLUTION_M)

    H, W, _ = canvas.shape

    # ── Step 5  生成预测掩膜 ─────────────────────────────────
    print(f"\n【Step 5】生成预测掩膜（排除全特征含 NaN 的像元）")
    feat_ok = ~np.isnan(canvas).any(axis=2)
    n_px    = int(feat_ok.sum())
    print(f"  待预测像元: {n_px:,} / {H*W:,}  ({100*n_px/(H*W):.1f}%)")
    if n_px == 0:
        raise RuntimeError("画布内无有效特征像元！请检查 Step 4 填充日志。")

    # ── Step 6  并行推理 ─────────────────────────────────────
    print(f"\n【Step 6】分批并行推理（{N_INFER_WORKERS} 线程，"
          f"批={BATCH_SIZE:,}，阈值={THRESHOLD}）")

    ri, ci = np.where(feat_ok)
    X_all  = np.ascontiguousarray(canvas[ri, ci, :], dtype=np.float32)  # (n_px, 9)

    prob_all = parallel_infer(clf, X_all, N_INFER_WORKERS, BATCH_SIZE)

    # 写回地图
    prob_map = np.full((H, W), np.nan, np.float32)
    pred_map = np.full((H, W), np.nan, np.float32)
    prob_map[ri, ci] = prob_all
    pred_map[ri, ci] = (prob_all >= THRESHOLD).astype(np.float32)

    n_flood = int(np.nansum(pred_map == 1))
    n_valid = int(np.nansum(~np.isnan(pred_map)))
    print(f"  洪水={n_flood:,}  非洪水={n_valid-n_flood:,}  "
          f"洪水比例={100*n_flood/max(n_valid,1):.2f}%")

    # ── Step 7  保存结果 ─────────────────────────────────────
    print(f"\n【Step 7】保存结果")
    res_suffix = f'_{int(TARGET_RESOLUTION_M)}m' if TARGET_RESOLUTION_M else ''
    pred_path = os.path.join(OUTPUT_DIR, f'flood_prediction{res_suffix}.tif')
    prob_path = os.path.join(OUTPUT_DIR, f'flood_probability{res_suffix}.tif')
    save_tif(pred_map, pred_path, canvas_gt, canvas_proj)
    save_tif(prob_map, prob_path, canvas_gt, canvas_proj)
    print(f"  ✓ flood_prediction{res_suffix}.tif  → {pred_path}")
    print(f"  ✓ flood_probability{res_suffix}.tif → {prob_path}")
    print(f"\n{SEP}\n  全部完成！\n{SEP}\n")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n✗ 运行出错: {e}")
        traceback.print_exc()
        sys.exit(1)