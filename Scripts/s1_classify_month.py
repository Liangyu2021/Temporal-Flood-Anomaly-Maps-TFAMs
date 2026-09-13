#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
栅格文件月份分类脚本
将多个网格子文件夹中的 .tif 文件，按月份重新组织到新目录结构：
    output/
    ├── month_01/
    │   ├── grid_01/   ← 网格编号自动探测，数量不限
    │   ├── grid_02/
    │   └── ...
    ├── month_02/
    │   └── ...
    └── month_12/
        └── ...
"""

import os
import re
import shutil
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime


# ============================================================
# 工具函数
# ============================================================

def parse_month(filename: str) -> int | None:
    """
    从文件名中提取月份，兼容两种命名格式：
        Image_20150617_VH_40.tif   → 6
        Image_20150617_...         → 6
    返回 1~12 的整数，无法解析则返回 None。
    """
    basename = os.path.splitext(filename)[0]
    parts = basename.split('_')

    # 取第二段作为日期字段（与原始 parse_filename 保持一致）
    if len(parts) >= 2:
        date_str = parts[1]
        if len(date_str) == 8 and date_str.isdigit():
            try:
                return datetime.strptime(date_str, '%Y%m%d').month
            except ValueError:
                pass

    # 备用：正则兜底，匹配任意位置的 YYYYMMDD
    m = re.search(r'(\d{4})(\d{2})(\d{2})', filename)
    if m:
        month = int(m.group(2))
        if 1 <= month <= 12:
            return month

    return None


def detect_grid_subfolders(base_dir: str,
                            manual_prefix: str | None = None
                            ) -> list[tuple[str, str]]:
    """
    自动探测 base_dir 下形如 PREFIX_NN 的网格子目录。

    返回：按编号排序的 [(grid_label, full_path), ...] 列表
        grid_label 示例：'01', '02', ..., '24'
    """
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"输入目录不存在: {base_dir}")

    entries = os.listdir(base_dir)

    if manual_prefix:
        prefix = manual_prefix
    else:
        # 自动探测：匹配末尾为两位数字的子目录
        candidates = []
        for name in entries:
            m = re.match(r'^(.+_)(\d{2})$', name)
            if m and os.path.isdir(os.path.join(base_dir, name)):
                candidates.append(m.group(1))

        if not candidates:
            raise ValueError(
                f"无法在 {base_dir} 中探测到 '*_NN' 格式的子目录，"
                "请通过 grid_prefix 手动指定前缀。"
            )
        prefix = Counter(candidates).most_common(1)[0][0]
        print(f"[自动探测] 网格目录前缀: '{prefix}'")

    # 收集所有匹配该前缀的子目录
    result = []
    for name in entries:
        if name.startswith(prefix):
            suffix = name[len(prefix):]
            if re.fullmatch(r'\d{2,}', suffix):          # 两位或更多数字编号
                full_path = os.path.join(base_dir, name)
                if os.path.isdir(full_path):
                    result.append((suffix, full_path))

    result.sort(key=lambda x: int(x[0]))
    return result


# ============================================================
# 核心分类函数
# ============================================================

def classify_by_month(base_input_dir: str,
                      base_output_dir: str,
                      grid_prefix: str | None = None,
                      copy: bool = True) -> None:
    """
    将网格子文件夹中的 .tif 文件按月份分类复制到新目录。

    参数
    ----
    base_input_dir  : 包含多个网格子文件夹的根目录
    base_output_dir : 输出根目录（不存在则自动创建）
    grid_prefix     : 网格子文件夹前缀，为 None 时自动探测
    copy            : True=复制文件；False=仅打印（演练模式）
    """

    grids = detect_grid_subfolders(base_input_dir, grid_prefix)

    if not grids:
        raise ValueError("未找到任何网格子目录，请检查路径和前缀设置。")

    print(f"\n共探测到 {len(grids)} 个网格: "
          f"{[g[0] for g in grids]}\n")

    # 统计信息
    stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    skipped = []

    for grid_label, grid_path in grids:

        tif_files = [f for f in os.listdir(grid_path) if f.lower().endswith('.tif')]

        if not tif_files:
            print(f"  [跳过] grid_{grid_label}: 目录为空或无 .tif 文件")
            continue

        print(f"  处理 grid_{grid_label}  ({len(tif_files)} 个文件) ...")

        for filename in tif_files:
            month = parse_month(filename)

            if month is None:
                skipped.append((grid_label, filename))
                continue

            month_str  = f"month_{month:02d}"
            grid_str   = f"grid_{grid_label}"

            dest_dir = Path(base_output_dir) / month_str / grid_str
            dest_file = dest_dir / filename

            if copy:
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(os.path.join(grid_path, filename), dest_file)

            stats[month_str][grid_str] += 1

    # ---- 汇总报告 ----
    print("\n" + "=" * 60)
    print("分类完成！目录结构统计：")
    print("=" * 60)

    total_files = 0
    for month_str in sorted(stats):
        month_total = sum(stats[month_str].values())
        total_files += month_total
        grids_in_month = len(stats[month_str])
        print(f"  {month_str}/  ({grids_in_month} 个网格，共 {month_total} 个文件)")

    print(f"\n  合计处理文件: {total_files}")

    if skipped:
        print(f"\n  ⚠ 无法解析日期，已跳过 {len(skipped)} 个文件：")
        for grid_label, fname in skipped[:10]:
            print(f"     grid_{grid_label}/{fname}")
        if len(skipped) > 10:
            print(f"     ... 还有 {len(skipped)-10} 个")

    print("=" * 60)

    if not copy:
        print("\n（演练模式：未实际复制文件，去掉 dry_run=True 后重新运行）")


# ============================================================
# 入口
# ============================================================

if __name__ == '__main__':

    # ---- 配置路径 ----
    BASE_INPUT  = r'/home/data/ql2024/flood-tly/australia/data/SARdata'
    BASE_OUTPUT = r'/home/data/ql2024/flood-tly/australia/data/monthly_SARdata'

    classify_by_month(
        base_input_dir  = BASE_INPUT,
        base_output_dir = BASE_OUTPUT,
        grid_prefix     = None,     # None = 自动探测；手动示例: 'Sentinel_1_POYANGHU_'
        copy            = True,     # False = 演练模式，只打印不复制
    )