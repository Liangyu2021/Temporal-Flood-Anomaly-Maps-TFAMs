#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从事件影像DB值反推F(x)值
给定事件影像DB栅格,基于TCEV参数,计算每个像元的F(x)值
超级优化版本 - Numba JIT + 多进程 + 栅格对齐 + NoData修复 + 批量处理 + 跳过已生成

SAR文件命名格式：Image_20141008_VH_42.tif  （日期在第2段，YYYYMMDD）
"""

import re
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from pathlib import Path
from tqdm import tqdm
import warnings
from multiprocessing import Pool, cpu_count
from numba import jit
import os
from datetime import datetime
from collections import Counter

os.environ['NUMBA_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════
#  ★ 配置区 ★
# ══════════════════════════════════════════════════════════════
base_tcev_dir   = "/home/data/ql2024/flood-tly/guangxi/data/tcev/tcev_seasonal/autumn"
base_sar_dir    = "/home/data/ql2024/flood-tly/guangxi/data/SARdata"
base_output_dir = "/home/data/ql2024/flood-tly/guangxi/data/event_Fx"

start_num = 1     # 起始编号
end_num   = 12    # 结束编号

# ── 日期范围过滤 ──────────────────────────────────────────────
# 只处理文件名中日期在此范围内的影像（格式 'YYYYMMDD'，None = 不限）
DATE_START = '20151001'   # 起始日期（含）
DATE_END   = '20151101'   # 结束日期（含）
# ─────────────────────────────────────────────────────────────

# ── ★ 基准栅格日期 ────────────────────────────────────────────
# 每个SAR子文件夹中，自动找到该日期的影像作为空间基准（范围+分辨率）
# 所有其他影像和TCEV参数均对齐到该基准
# 设为 None 则以各影像自身空间参考为准（原有行为）
REFERENCE_DATE = '20150220'   # 例：'20141008'，或 None
# ─────────────────────────────────────────────────────────────

N_PROCESSES   = 2
CHUNK_SIZE    = 100
REVERSE_DB    = True
SKIP_EXISTING = True
# ══════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════
#  自动探测前缀工具函数
# ══════════════════════════════════════════════════════════════

def detect_sar_prefix(base_sar_dir):
    """
    扫描 base_sar_dir，自动探测 SAR 子目录前缀。
    例如 Sentinel_1_POYANGHU_01 → 前缀 Sentinel_1_POYANGHU_
    例如 Sentinel_1_YILANG_01   → 前缀 Sentinel_1_YILANG_

    返回:
        str: 探测到的前缀
    异常:
        ValueError: 探测失败时抛出，并打印实际目录列表辅助排查
    """
    candidates = []
    if os.path.exists(base_sar_dir):
        for name in os.listdir(base_sar_dir):
            m = re.match(r'^(.+_)(\d{2})$', name)
            if m and os.path.isdir(os.path.join(base_sar_dir, name)):
                candidates.append(m.group(1))

    if not candidates:
        actual = [
            n for n in os.listdir(base_sar_dir)
            if os.path.isdir(os.path.join(base_sar_dir, n))
        ] if os.path.exists(base_sar_dir) else []
        raise ValueError(
            f"无法在 {base_sar_dir} 中自动探测到 '*_NN' 格式的子目录。\n"
            f"实际找到的子目录: {actual}\n"
            "请确认目录命名格式。"
        )

    prefix = Counter(candidates).most_common(1)[0][0]
    return prefix


# ══════════════════════════════════════════════════════════════
#  日期解析工具
# ══════════════════════════════════════════════════════════════

def extract_date_from_filename(filename):
    """
    从文件名提取日期。
    支持格式：Image_20141008_VH_42.tif → '20141008'
    规则：找第一个 8 位纯数字段。
    """
    stem = Path(filename).stem
    parts = re.split(r'[_\-]', stem)
    for part in parts:
        if re.fullmatch(r'\d{8}', part):
            return part
    return None


def in_date_range(filename, date_start, date_end):
    """判断文件名中的日期是否在指定范围内。"""
    date_str = extract_date_from_filename(filename)
    if date_str is None:
        print(f"  [警告] 无法从文件名解析日期，跳过: {filename}")
        return False
    try:
        dt = datetime.strptime(date_str, '%Y%m%d')
        if date_start and dt < datetime.strptime(date_start, '%Y%m%d'):
            return False
        if date_end   and dt > datetime.strptime(date_end,   '%Y%m%d'):
            return False
        return True
    except ValueError:
        print(f"  [警告] 日期格式异常，跳过: {filename}")
        return False


# ══════════════════════════════════════════════════════════════
#  核心计算（Numba JIT）
# ══════════════════════════════════════════════════════════════

@jit(nopython=True, fastmath=True)
def tcef_function(x, alpha1, beta1, alpha2, beta2):
    term1 = np.exp(-np.exp(-alpha1 * (x - beta1)))
    term2 = np.exp(-np.exp(-alpha2 * (x - beta2)))
    return term1 * term2


@jit(nopython=True, fastmath=True)
def process_chunk_numba(db_chunk, alpha1_chunk, beta1_chunk, alpha2_chunk, beta2_chunk):
    n_rows, n_cols = db_chunk.shape
    fx_chunk = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    for i in range(n_rows):
        for j in range(n_cols):
            db_value = db_chunk[i, j]
            a1 = alpha1_chunk[i, j]
            b1 = beta1_chunk[i, j]
            a2 = alpha2_chunk[i, j]
            b2 = beta2_chunk[i, j]
            if (np.isnan(db_value) or np.isnan(a1) or np.isnan(b1) or
                    np.isnan(a2) or np.isnan(b2)):
                continue
            fx_chunk[i, j] = tcef_function(db_value, a1, b1, a2, b2)
    return fx_chunk


def process_chunk_wrapper(args):
    row_indices, db_chunk, alpha1_chunk, beta1_chunk, alpha2_chunk, beta2_chunk = args
    fx_chunk = process_chunk_numba(db_chunk, alpha1_chunk, beta1_chunk,
                                   alpha2_chunk, beta2_chunk)
    return (row_indices, fx_chunk)


def resample_to_reference(src_path, ref_profile):
    """将 src_path 栅格重采样/重投影到 ref_profile 定义的空间参考。"""
    with rasterio.open(src_path) as src:
        src_data = src.read(1).astype(np.float32)
        src_transform = src.transform
        src_crs = src.crs

        # 已完全一致，直接返回
        if (src_data.shape == (ref_profile['height'], ref_profile['width']) and
                src_transform == ref_profile['transform'] and
                src_crs == ref_profile['crs']):
            return src_data.astype(np.float64)

        dst_data = np.full(
            (ref_profile['height'], ref_profile['width']),
            np.nan, dtype=np.float32
        )
        reproject(
            source=src_data, destination=dst_data,
            src_transform=src_transform, src_crs=src_crs,
            dst_transform=ref_profile['transform'], dst_crs=ref_profile['crs'],
            resampling=Resampling.bilinear
        )
        return dst_data.astype(np.float64)


# ══════════════════════════════════════════════════════════════
#  单影像处理
# ══════════════════════════════════════════════════════════════

def process_event_image(event_db_path, alpha1_path, beta1_path, alpha2_path, beta2_path,
                        output_dir, reference_profile=None,
                        n_processes=None, chunk_size=100,
                        reverse_db=True, skip_existing=True):
    """
    处理单景事件影像。

    Parameters
    ----------
    reference_profile : dict or None
        rasterio profile 字典，定义输出的空间范围与分辨率。
        为 None 时以事件影像自身空间参考为准（原有行为）。
    """
    if n_processes is None:
        n_processes = cpu_count()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    event_name  = Path(event_db_path).stem
    output_path = output_dir / f"{event_name}_Fx.tif"

    if skip_existing and output_path.exists():
        print(f"⏭️  跳过(已存在): {event_name}_Fx.tif")
        return output_path

    print(f"\n{'='*70}")
    print(f"处理事件影像: {event_name}")
    print(f"{'='*70}")

    # ── 读取事件影像，按需对齐到基准 ─────────────────────────
    with rasterio.open(event_db_path) as src:
        if reference_profile is None:
            db_raster = src.read(1).astype(np.float64)
            profile   = src.profile.copy()
            print(f"  空间基准: 自身影像")
        else:
            db_raw = src.read(1).astype(np.float32)
            dst_db = np.full(
                (reference_profile['height'], reference_profile['width']),
                np.nan, dtype=np.float32
            )
            reproject(
                source=db_raw, destination=dst_db,
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=reference_profile['transform'],
                dst_crs=reference_profile['crs'],
                resampling=Resampling.bilinear
            )
            db_raster = dst_db.astype(np.float64)
            profile   = reference_profile.copy()
            print(f"  空间基准: 外部基准栅格（已对齐）")

    height, width = db_raster.shape

    if reverse_db:
        print("对DB值取相反数...")
        db_raster = -db_raster

    print(f"  影像尺寸: {height} × {width}")
    valid_db = ~np.isnan(db_raster)
    print(f"  有效像元: {np.sum(valid_db):,}")
    if np.sum(valid_db) > 0:
        print(f"  DB范围: [{np.nanmin(db_raster):.2f}, {np.nanmax(db_raster):.2f}]")

    # ── 读取并对齐 TCEV 参数栅格 ──────────────────────────────
    print("\n读取并对齐TCEV参数栅格...")
    alpha1 = resample_to_reference(alpha1_path, profile)
    beta1  = resample_to_reference(beta1_path,  profile)
    alpha2 = resample_to_reference(alpha2_path, profile)
    beta2  = resample_to_reference(beta2_path,  profile)

    valid_mask = (~np.isnan(db_raster) & ~np.isnan(alpha1) &
                  ~np.isnan(beta1) & ~np.isnan(alpha2) & ~np.isnan(beta2))
    total_valid_pixels = np.sum(valid_mask)
    print(f"可计算像元: {total_valid_pixels:,}")

    profile.update(dtype=rasterio.float32, count=1, compress='lzw', nodata=-9999.0)

    print("\n预热Numba JIT编译器...")
    dummy = np.array([[1.0, 2.0], [3.0, 4.0]])
    _ = process_chunk_numba(dummy, dummy, dummy, dummy, dummy)
    print("✓ JIT编译完成\n")

    fx_raster = np.full((height, width), -9999.0, dtype=np.float32)

    chunks = []
    for start_row in range(0, height, chunk_size):
        end_row = min(start_row + chunk_size, height)
        chunks.append((
            list(range(start_row, end_row)),
            db_raster[start_row:end_row, :],
            alpha1[start_row:end_row, :],
            beta1[start_row:end_row, :],
            alpha2[start_row:end_row, :],
            beta2[start_row:end_row, :],
        ))

    print(f"数据已分为 {len(chunks)} 块，开始并行计算...\n")
    with Pool(processes=n_processes) as pool:
        results = list(tqdm(
            pool.imap(process_chunk_wrapper, chunks),
            total=len(chunks), desc="计算F(x)", unit="chunk", ncols=100
        ))

    print("\n合并结果...")
    success_count = 0
    for row_indices, fx_chunk in results:
        start_row = row_indices[0]
        end_row   = row_indices[-1] + 1
        fx_chunk_clean = np.where(np.isnan(fx_chunk), -9999.0, fx_chunk)
        fx_raster[start_row:end_row, :] = fx_chunk_clean
        success_count += int(np.sum(~np.isnan(fx_chunk)))

    print(f"  成功: {success_count:,} ({100*success_count/max(total_valid_pixels,1):.2f}%)")
    if success_count > 0:
        valid_fx = fx_raster[fx_raster != -9999.0]
        print(f"  F(x)范围: [{valid_fx.min():.6f}, {valid_fx.max():.6f}]")

    print(f"\n保存栅格: {output_path}")
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(fx_raster, 1)
    print("✓ 已保存\n")
    return output_path


# ══════════════════════════════════════════════════════════════
#  文件夹处理
# ══════════════════════════════════════════════════════════════

def process_event_folder(event_folder, alpha1_path, beta1_path, alpha2_path, beta2_path,
                         output_dir, file_pattern="*.tif",
                         date_start=None, date_end=None,
                         reference_date=None,
                         n_processes=None, chunk_size=100,
                         reverse_db=True, skip_existing=True):

    event_folder = Path(event_folder)
    output_dir   = Path(output_dir)

    all_files = sorted(event_folder.glob(file_pattern))
    if not all_files:
        print(f"错误: 在 {event_folder} 中未找到匹配 {file_pattern} 的文件!")
        return []

    # ── 在当前子文件夹中自动查找基准影像 ─────────────────────
    reference_profile = None
    if reference_date:
        matched = [f for f in all_files
                   if extract_date_from_filename(f.name) == reference_date]
        if matched:
            ref_path = matched[0]
            with rasterio.open(ref_path) as ref:
                reference_profile = ref.profile.copy()
            print(f"[基准栅格] {ref_path.name}  "
                  f"({reference_profile['width']} × {reference_profile['height']}, "
                  f"res={reference_profile['transform'].a:.6f})")
        else:
            print(f"[警告] 在 {event_folder} 中未找到日期 {reference_date} 的文件，"
                  f"将以各影像自身空间参考为准。")

    if date_start or date_end:
        filtered = [f for f in all_files if in_date_range(f.name, date_start, date_end)]
        print(f"日期过滤: {len(all_files)} → {len(filtered)} 个文件"
              f"（{date_start or '不限'} ~ {date_end or '不限'}）")
        event_files = filtered
    else:
        event_files = all_files

    if not event_files:
        print("过滤后无可处理文件，跳过此文件夹。")
        return []

    print("="*70)
    print(f"事件影像文件夹: {event_folder}")
    print(f"待处理: {len(event_files)} 个影像")
    print(f"输出目录: {output_dir}")
    print("="*70)

    output_paths  = []
    processed_cnt = 0
    skipped_cnt   = 0

    for i, event_file in enumerate(event_files, 1):
        print(f"\n[{i}/{len(event_files)}] {event_file.name}")
        output_path = output_dir / f"{event_file.stem}_Fx.tif"

        if skip_existing and output_path.exists():
            print(f"⏭️  跳过(已存在)")
            skipped_cnt += 1
            output_paths.append(output_path)
            continue

        output_path = process_event_image(
            event_db_path=event_file,
            alpha1_path=alpha1_path, beta1_path=beta1_path,
            alpha2_path=alpha2_path, beta2_path=beta2_path,
            output_dir=output_dir,
            reference_profile=reference_profile,
            n_processes=n_processes, chunk_size=chunk_size,
            reverse_db=reverse_db, skip_existing=skip_existing
        )
        output_paths.append(output_path)
        processed_cnt += 1

    print(f"\n本文件夹完成: 新处理 {processed_cnt} 个，跳过 {skipped_cnt} 个")
    return output_paths


# ══════════════════════════════════════════════════════════════
#  批量处理
# ══════════════════════════════════════════════════════════════

def batch_process_folders(base_tcev_dir, base_sar_dir, base_output_dir,
                          start_num, end_num,
                          date_start=None, date_end=None,
                          reference_date=None,
                          n_processes=None, chunk_size=100,
                          reverse_db=True, skip_existing=True):

    # ---- 自动探测 SAR 子目录前缀（只探测一次）----
    sar_prefix = detect_sar_prefix(base_sar_dir)
    print(f"[自动探测] SAR子目录前缀: {sar_prefix}")

    print("\n" + "="*80)
    print("批量处理：事件影像 DB → F(x)")
    print("="*80)
    print(f"TCEV根目录    : {base_tcev_dir}")
    print(f"SAR根目录     : {base_sar_dir}")
    print(f"SAR子目录前缀 : {sar_prefix}")
    print(f"输出根目录    : {base_output_dir}")
    print(f"处理编号      : {start_num:02d} ~ {end_num:02d}")
    print(f"日期范围      : {date_start or '不限'} ~ {date_end or '不限'}")
    print(f"空间基准日期  : {reference_date or '不指定（各影像自身）'}")
    print("="*80)

    success_list = []
    failed_list  = []

    for i in range(start_num, end_num + 1):
        idx_str = f"{i:02d}"

        print("\n" + "#"*80)
        print(f"# 开始处理编号: {idx_str}")
        print("#"*80 + "\n")

        try:
            tcev_folder = Path(base_tcev_dir) / f"grid_{idx_str}"

            alpha1_path = tcev_folder / "L1_alpha.tif"
            beta1_path  = tcev_folder / "L1_beta.tif"
            alpha2_path = tcev_folder / "L2_alpha.tif"
            beta2_path  = tcev_folder / "L2_beta.tif"

            event_folder = Path(base_sar_dir)    / f"{sar_prefix}{idx_str}"
            output_dir   = Path(base_output_dir) / f"Event_Fx_{idx_str}_Results"

            # 检查必要路径
            for path in [alpha1_path, beta1_path, alpha2_path, beta2_path]:
                if not path.exists():
                    raise FileNotFoundError(f"未找到文件: {path}")
            if not event_folder.exists():
                raise FileNotFoundError(f"SAR目录不存在: {event_folder}")

            print(f"配置信息:")
            print(f"  - TCEV目录   : {tcev_folder}")
            print(f"  - SAR目录    : {event_folder}")
            print(f"  - 输出目录   : {output_dir}\n")

            process_event_folder(
                event_folder=event_folder,
                alpha1_path=str(alpha1_path), beta1_path=str(beta1_path),
                alpha2_path=str(alpha2_path), beta2_path=str(beta2_path),
                output_dir=output_dir,
                file_pattern="*.tif",
                date_start=date_start,
                date_end=date_end,
                reference_date=reference_date,
                n_processes=n_processes,
                chunk_size=chunk_size,
                reverse_db=reverse_db,
                skip_existing=skip_existing
            )

            success_list.append(idx_str)
            print(f"\n✓ 编号 {idx_str} 处理成功!\n")

        except Exception as e:
            failed_list.append((idx_str, str(e)))
            print(f"\n✗ 编号 {idx_str} 处理失败: {e}\n")
            import traceback
            traceback.print_exc()
            continue

    # ---- 汇总报告 ----
    total = end_num - start_num + 1
    print("\n" + "="*80)
    print("全部编号处理完成!")
    print("="*80)
    print(f"成功: {len(success_list)}/{total} 个  →  {success_list}")
    if failed_list:
        print(f"\n失败的编号:")
        for idx_str, err in failed_list:
            print(f"  - {idx_str}: {err}")
    else:
        print("\n所有编号均处理成功!")
    print("="*80 + "\n")


# ══════════════════════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════════════════════

def main():
    batch_process_folders(
        base_tcev_dir   = base_tcev_dir,
        base_sar_dir    = base_sar_dir,
        base_output_dir = base_output_dir,
        start_num       = start_num,
        end_num         = end_num,
        date_start      = DATE_START,
        date_end        = DATE_END,
        reference_date  = REFERENCE_DATE,
        n_processes     = N_PROCESSES,
        chunk_size      = CHUNK_SIZE,
        reverse_db      = REVERSE_DB,
        skip_existing   = SKIP_EXISTING
    )


if __name__ == "__main__":
    main()