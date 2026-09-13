#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
月份文件夹 → 季节文件夹 重组脚本
================================================
输入结构：
    base_input/
    ├── month_01/ grid_01/ *.tif
    ├── month_03/ grid_01/ *.tif
    └── ...

输出结构：
    base_output/
    ├── spring/ grid_01/ *.tif   （3、4、5 月）
    ├── summer/ grid_01/ *.tif   （6、7、8 月）
    ├── autumn/ grid_01/ *.tif   （9、10、11 月）
    └── winter/ grid_01/ *.tif   （12、1、2 月）
"""

import os
import re
import shutil
from pathlib import Path
from collections import defaultdict


# ============================================================
# 配置区
# ============================================================

BASE_INPUT  = r'/home/data/ql2024/flood-tly/jingjinji/data/monthly_SARdata'
BASE_OUTPUT = r'/home/data/ql2024/flood-tly/jingjinji/data/SARdata_seasonal'

SEASON_MAP = {
    3: 'spring', 4: 'spring', 5: 'spring',
    6: 'summer', 7: 'summer', 8: 'summer',
    9: 'autumn', 10: 'autumn', 11: 'autumn',
    12: 'winter', 1: 'winter', 2: 'winter',
}

COPY = True   # False = 演练模式，只打印不复制


# ============================================================
# 核心逻辑
# ============================================================

def parse_month_from_folder(name: str):
    """从 month_01 ~ month_12 文件夹名中提取月份整数，失败返回 None。"""
    m = re.search(r'month[_\-]?(\d{1,2})', name, re.IGNORECASE)
    if m:
        month = int(m.group(1))
        if 1 <= month <= 12:
            return month
    return None


def classify_by_season(base_input: str, base_output: str,
                        season_map: dict, copy: bool = True):

    stats = defaultdict(lambda: defaultdict(int))
    skipped_folders = []

    month_folders = sorted(os.listdir(base_input))

    for mfolder in month_folders:
        mfolder_path = os.path.join(base_input, mfolder)
        if not os.path.isdir(mfolder_path):
            continue

        month = parse_month_from_folder(mfolder)
        if month is None:
            skipped_folders.append(mfolder)
            continue

        season = season_map.get(month)
        if season is None:
            skipped_folders.append(mfolder)
            continue

        # 遍历网格子文件夹
        for gfolder in sorted(os.listdir(mfolder_path)):
            gfolder_path = os.path.join(mfolder_path, gfolder)
            if not os.path.isdir(gfolder_path):
                continue

            tif_files = [f for f in os.listdir(gfolder_path)
                         if f.lower().endswith('.tif')]
            if not tif_files:
                continue

            dest_dir = Path(base_output) / season / gfolder

            print(f"  {mfolder}/{gfolder} → {season}/{gfolder}  "
                  f"({len(tif_files)} 个文件)")

            for filename in tif_files:
                src = os.path.join(gfolder_path, filename)
                dst = dest_dir / filename
                if copy:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                stats[season][gfolder] += 1

    # 汇总
    print("\n" + "=" * 60)
    print("季节分类完成！")
    print("=" * 60)
    total = 0
    for season in ('spring', 'summer', 'autumn', 'winter'):
        if season not in stats:
            continue
        season_total = sum(stats[season].values())
        total += season_total
        print(f"  {season:8s}: {len(stats[season]):3d} 个网格，"
              f"共 {season_total:5d} 个文件")

    print(f"\n  合计: {total} 个文件")

    if skipped_folders:
        print(f"\n  ⚠ 跳过 {len(skipped_folders)} 个无法识别的文件夹:")
        for f in skipped_folders:
            print(f"     {f}")

    print("=" * 60)
    if not copy:
        print("\n（演练模式：未实际复制，将 COPY 改为 True 后重新运行）")


# ============================================================
# 入口
# ============================================================

if __name__ == '__main__':
    classify_by_season(
        base_input  = BASE_INPUT,
        base_output = BASE_OUTPUT,
        season_map  = SEASON_MAP,
        copy        = COPY,
    )