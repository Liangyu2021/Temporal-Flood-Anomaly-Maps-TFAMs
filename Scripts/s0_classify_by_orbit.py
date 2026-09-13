#!/usr/bin/env python3
import os
import shutil
import re

INPUT_DIR  = "/home/data/ql2024/flood-tly/xinjiang/data/SARdata/Sentinel_1_Xinjiang_01"
OUTPUT_DIR = "/home/data/ql2024/flood-tly/xinjiang/data/SARdata/"


def parse_orbit(filename):
    """从文件名中提取轨道号，如 Image_20150716_VH_113.tif -> '113'"""
    m = re.search(r'_(\d+)\.(tif|tiff)$', filename, re.IGNORECASE)
    return m.group(1) if m else None


def classify_by_orbit(input_dir, output_dir):
    for fname in os.listdir(input_dir):
        orbit = parse_orbit(fname)
        if orbit is None:
            continue
        dest = os.path.join(output_dir, f"orbit_{orbit}")
        os.makedirs(dest, exist_ok=True)
        shutil.copy2(os.path.join(input_dir, fname), os.path.join(dest, fname))
        print(f"  {fname}  ->  orbit_{orbit}/")


if __name__ == "__main__":
    classify_by_orbit(INPUT_DIR, OUTPUT_DIR)
