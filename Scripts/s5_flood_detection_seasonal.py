#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
flood_detection_tcev.py
────────────────────────────────────────────────────────────────────
基于 TCEV 5 通道特征的洪水检测训练与推理脚本。

与 flood_detection_seasonal.py 的核心区别：
  · 特征通道由 6 通道（归一化版）改为 5 通道（TCEV 版），与
    build_dataset_from_points.py 的输出完全对齐：
      ch0  raw_db        原始物理 dB（不归一化）
      ch1  gumbel        Gumbel(Fx) = -log(-log(Fx))
      ch2  fx            超越概率 F(x) ∈ (0, 1)
      ch3  slope_ratio   L2_slope / L1_slope（断点后 / 前段斜率比，标量广播）
      ch4  bp_db         像元级断点 dB（来自 breakpoint_DB.tif，标量广播）

  · 推理时需额外传入 tcev_root（TCEV 断点栅格根目录），
    SeasonalTargetSpec 新增 tcev_root 字段。

训练数据源（build_dataset_from_points.py 输出）
  output_dir/
    dataset_cache.hdf5   ← 含 curves(N,seq_len,5) / labels / grid_index

推理数据源（sar_seasonal_hdf5.py 输出）
  hdf5_root/season/grid_XX/
    sorted_negative_values.hdf5
    index_values.hdf5
    normalized_negative_values.hdf5   ← 仅用于读 height/width/pixel_min/max

TCEV 断点栅格（tcev_seasonal 目录，与 build_dataset_from_points.py 一致）
  tcev_root/season/grid_XX/
    breakpoint_DB.tif
    breakpoint_FX.tif（备用，推理暂不使用）

输出结构：
  target.output_dir /
    source.alias /
      season /             ← spring / summer / autumn / winter
        grid_XX /
          flat_label_<season>_grid_XX.hdf5
          flat_label_<season>_grid_XX.npy
"""

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import os, json, logging, re, warnings, traceback
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict

import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

warnings.filterwarnings('ignore', category=RuntimeWarning)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# ── 常量 ──────────────────────────────────────────────────────────
ALL_SEASONS    = ('spring', 'summer', 'autumn', 'winter')
GRID_PAT       = re.compile(r'^grid_(\d+)$')
REQUIRED_HDF5  = [
    'sorted_negative_values.hdf5',
    'index_values.hdf5',
    'normalized_negative_values.hdf5',
]

# ── 通道描述（与 build_dataset_from_points.py 完全一致） ──────────
CHANNEL_DESC = [
    'raw_db',       # ch0  原始物理 dB（不归一化）
    'gumbel',       # ch1  Gumbel(Fx)
    'fx',           # ch2  超越概率 F(x)
    'slope_ratio',  # ch3  L2_slope / L1_slope（标量广播）
    'bp_db',        # ch4  断点 dB 值（标量广播）
]
IN_CHANNELS = len(CHANNEL_DESC)   # 5


# ================================================================
# ★ 公共工具函数（训练 / 推理共用）
# ================================================================

def pad_or_clip_uniform(arr: np.ndarray, length: int) -> np.ndarray:
    """
    均匀采样到 length 个点（与 build_dataset_from_points.py 完全相同）：
      · len >= length → np.linspace 均匀抽取
      · len <  length → 已有数据放头部，剩余补零
    """
    n = len(arr)
    if n == 0:
        return np.zeros(length, dtype=np.float32)
    if n >= length:
        idx = np.round(np.linspace(0, n - 1, length)).astype(int)
        return arr[idx].copy().astype(np.float32)
    out = np.zeros(length, dtype=np.float32)
    out[:n] = arr
    return out


def _linregress_slope(x: np.ndarray, y: np.ndarray) -> float:
    """最小二乘线性斜率；点数 < 2 或分母为 0 时返回 np.nan。"""
    if len(x) < 2:
        return np.nan
    xm = x.mean()
    ym = y.mean()
    denom = ((x - xm) ** 2).sum()
    if denom < 1e-12:
        return np.nan
    return float(((x - xm) * (y - ym)).sum() / denom)


def build_pixel_features_tcev(
    sorted_db:  np.ndarray,   # 全时序排序 dB（可含 NaN），升序
    fx_raw:     np.ndarray,   # 全时序经验频率 F(x)（可含 NaN）
    bp_db_val:  float,        # 像元级断点 dB（来自 breakpoint_DB.tif）
    seq_len:    int,
    min_valid:  int = 4,
    min_seg:    int = 2,
) -> Optional[np.ndarray]:
    """
    ★ 与 build_dataset_from_points.build_curve_features_tcev 完全对齐的
      五通道特征构建函数，训练和推理统一调用此处。

    通道定义（共 5 个）：
      ch0  raw_db        原始物理 dB（不归一化）
      ch1  gumbel        Gumbel(Fx) = -log(-log(Fx))
      ch2  fx            超越概率 F(x) ∈ (0, 1)
      ch3  slope_ratio   L2_slope / L1_slope（标量广播）
      ch4  bp_db         断点 dB 值（标量广播）

    返回 None 表示该像元无效。
    返回 shape (seq_len, 5) float32 数组。
    """
    if np.isnan(bp_db_val):
        return None

    # ── 有效点过滤 ────────────────────────────────────────────────
    valid = (
        ~np.isnan(sorted_db) & ~np.isnan(fx_raw)
        & (fx_raw > 0) & (fx_raw < 1)
    )
    n_valid = int(valid.sum())
    if n_valid < min_valid:
        return None

    db_v  = sorted_db[valid]   # 升序（sorted_negative_values 已升序）
    fx_v  = fx_raw[valid]
    gum_v = -np.log(-np.log(np.clip(fx_v, 1e-6, 1 - 1e-6))).astype(np.float32)

    # ── 断点分割，计算 slope_ratio ─────────────────────────────────
    mask_l1 = db_v <= bp_db_val
    mask_l2 = db_v >  bp_db_val

    l1_slope = (_linregress_slope(db_v[mask_l1], gum_v[mask_l1])
                if mask_l1.sum() >= min_seg else np.nan)
    l2_slope = (_linregress_slope(db_v[mask_l2], gum_v[mask_l2])
                if mask_l2.sum() >= min_seg else np.nan)

    if np.isnan(l1_slope) or np.isnan(l2_slope) or abs(l1_slope) < 1e-12:
        slope_ratio = np.nan
    else:
        slope_ratio = float(l2_slope / l1_slope)

    # ── 均匀采样到 seq_len ────────────────────────────────────────
    db_seq  = pad_or_clip_uniform(db_v.astype(np.float32), seq_len)
    gum_seq = pad_or_clip_uniform(gum_v,                   seq_len)
    fx_seq  = pad_or_clip_uniform(fx_v.astype(np.float32), seq_len)

    # 标量通道广播
    ratio_seq = np.full(seq_len, slope_ratio, dtype=np.float32)
    bp_seq    = np.full(seq_len, bp_db_val,   dtype=np.float32)

    return np.stack([db_seq, gum_seq, fx_seq, ratio_seq, bp_seq], axis=1)
    # shape: (seq_len, 5)


# ================================================================
# TCEV 断点读取工具
# ================================================================

def read_tif_pixel(tif_path: str, row: int, col: int) -> float:
    """
    从单波段 GeoTIFF 读取 (row, col) 像元值。
    文件不存在 / 越界 / NoData → np.nan。
    """
    try:
        from osgeo import gdal
    except ImportError:
        log.warning("GDAL 未安装，无法读取断点 TIF，推理将跳过所有像元")
        return np.nan

    if not os.path.exists(tif_path):
        return np.nan
    ds = gdal.Open(tif_path)
    if ds is None:
        return np.nan
    if not (0 <= row < ds.RasterYSize and 0 <= col < ds.RasterXSize):
        ds = None
        return np.nan
    band = ds.GetRasterBand(1)
    val  = band.ReadAsArray(col, row, 1, 1)
    nd   = band.GetNoDataValue()
    ds   = None
    v    = float(val[0, 0])
    if nd is not None and abs(v - nd) < 1e-3:
        return np.nan
    return v


def get_breakpoint_db(tcev_grid_dir: str, row: int, col: int) -> float:
    """读取 tcev_grid_dir/breakpoint_DB.tif 中 (row, col) 的断点 dB 值。"""
    return read_tif_pixel(
        os.path.join(tcev_grid_dir, 'breakpoint_DB.tif'), row, col
    )


# ================================================================
# 目录扫描（季节化结构）
# ================================================================

def _scan_seasonal_grids(
    hdf5_root:   str,
    season:      str,
    grid_filter: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    season_dir = os.path.join(hdf5_root, season)
    if not os.path.isdir(season_dir):
        log.warning(f"季节目录不存在: {season_dir}")
        return []

    result = []
    for grid_name in sorted(
        os.listdir(season_dir),
        key=lambda g: int(re.search(r'\d+', g).group()) if re.search(r'\d+', g) else 0,
    ):
        if not GRID_PAT.match(grid_name):
            continue
        if grid_filter and grid_name not in grid_filter:
            continue
        grid_path = os.path.join(season_dir, grid_name)
        if not os.path.isdir(grid_path):
            continue
        if all(os.path.exists(os.path.join(grid_path, f)) for f in REQUIRED_HDF5):
            result.append((grid_name, grid_path))
        else:
            log.warning(f"  跳过 {season}/{grid_name}：缺少必要 HDF5 文件")
    return result


def _available_seasons_in_root(
    root:          str,
    season_filter: Optional[List[str]] = None,
) -> List[str]:
    return [
        s for s in ALL_SEASONS
        if (season_filter is None or s in season_filter)
        and os.path.isdir(os.path.join(root, s))
    ]


# ================================================================
# 配置
# ================================================================

@dataclass
class Config:
    # ── 模型结构 ──────────────────────────────────────────────────
    in_channels:   int       = IN_CHANNELS          # ★ 固定为 5
    conv_channels: List[int] = field(default_factory=lambda: [32, 64, 32])
    kernel_size:   int       = 3
    dropout:       float     = 0.3
    fc_hidden:     int       = 32
    # ── 训练超参 ──────────────────────────────────────────────────
    epochs:        int       = 100
    batch_size:    int       = 512
    lr:            float     = 1e-3
    weight_decay:  float     = 1e-4
    val_ratio:     float     = 0.2
    patience:      int       = 20
    focal_gamma:   float     = 2.0
    # ── 推理 ──────────────────────────────────────────────────────
    prob_threshold: float    = 0.5
    # ── 运行时 ──────────────────────────────────────────────────
    device:        str       = 'auto'
    num_workers:   int       = 0
    pin_memory:    bool      = True

    def resolve_device(self) -> torch.device:
        return _resolve_device(self.device)


def _resolve_device(device_str: str) -> torch.device:
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)


# ================================================================
# 数据类：季节化训练源 / 推理目标
# ================================================================

@dataclass
class SeasonalSourceSpec:
    """
    训练源指向 build_dataset_from_points.py 的输出目录。
    dataset_root 下按季节组织：
      dataset_root/
        autumn/
          dataset_cache.hdf5   ← curves(N,seq_len,5) + labels + grid_index
    """
    dataset_root:  str
    model_root:    Optional[str]       = None
    alias:         str                 = ''
    season_filter: Optional[List[str]] = None
    enabled:       bool                = True
    chunk_train:   bool                = True
    force_retrain: bool                = False

    def display_name(self) -> str:
        return self.alias or os.path.basename(self.dataset_root)

    def dataset_path(self, season: str) -> str:
        return os.path.join(self.dataset_root, season, 'dataset_cache.hdf5')

    def model_save_dir(self, season: str) -> str:
        base = self.model_root or os.path.join(
            os.path.dirname(self.dataset_root), 'trained_models_tcev')
        return os.path.join(base, self.display_name(), season)

    def model_save_path(self, season: str) -> str:
        return os.path.join(self.model_save_dir(season), 'flat_model_tcev.pth')


@dataclass
class SeasonalTargetSpec:
    """
    推理目标。

    ★ 新增字段 tcev_root：
       指向 TCEV 断点栅格目录（与 build_dataset_from_points.py 中
       BASE_TCEV_DIR 一致），结构为：
         tcev_root/season/grid_XX/breakpoint_DB.tif
       若为 None，则尝试从 hdf5_root 同级的 tcev 子目录自动推断。
    """
    hdf5_root:     str
    output_dir:    str
    tcev_root:     Optional[str]       = None   # ★ TCEV 断点栅格根目录
    season_filter: Optional[List[str]] = None
    grid_filter:   Optional[List[str]] = None
    alias:         str                 = ''

    def display_name(self) -> str:
        return self.alias or os.path.basename(self.hdf5_root)

    def resolve_tcev_root(self) -> Optional[str]:
        """返回有效的 tcev_root，找不到则返回 None。"""
        if self.tcev_root and os.path.isdir(self.tcev_root):
            return self.tcev_root
        # 自动推断：hdf5_root 同级的 tcev/tcev_seasonal 目录
        parent = os.path.dirname(self.hdf5_root)
        for candidate in ('tcev_seasonal', 'tcev'):
            p = os.path.join(parent, candidate)
            if os.path.isdir(p):
                log.info(f"  自动推断 tcev_root = {p}")
                return p
        log.warning(
            f"  未找到 tcev_root（传入值={self.tcev_root}），"
            "推理将跳过无断点像元（slope_ratio / bp_db 均为 NaN）"
        )
        return None


# ================================================================
# HDF5 元信息加载
# ================================================================

def load_dataset_meta(path: str) -> Tuple[List[str], Dict]:
    with h5py.File(path, 'r') as hf:
        grid_keys = json.loads(bytes(hf['grid_keys'][:].tobytes()).decode('utf-8'))
        meta = dict(hf.attrs)
    # 验证通道数与本脚本期望一致
    ds_in_ch = int(meta.get('in_channels', IN_CHANNELS))
    if ds_in_ch != IN_CHANNELS:
        raise ValueError(
            f"数据集 in_channels={ds_in_ch}，"
            f"本脚本期望 {IN_CHANNELS}（TCEV 5通道版）。"
            "请检查是否使用了正确的 build_dataset_from_points.py 输出。"
        )
    log.info(
        f"缓存元信息 ← {path}\n"
        f"  网格={len(grid_keys)}  像元={meta.get('n_pixels','?')}  "
        f"洪水比例={meta.get('flood_ratio_pct','?')}%  "
        f"季节={meta.get('month_label', meta.get('season','?'))}  "
        f"通道数={ds_in_ch}"
    )
    return grid_keys, meta


# ================================================================
# ★ 推理侧网格曲线加载（TCEV 5 通道，对齐训练特征）
# ================================================================

def load_grid_curves(
    grid_dir:      str,
    seq_len:       int,
    tcev_grid_dir: Optional[str] = None,
    min_valid:     int = 4,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    从 grid_dir 中读取三个 HDF5 文件，结合 tcev_grid_dir 中的
    breakpoint_DB.tif，提取 TCEV 五通道曲线特征。

    通道定义与训练侧（build_pixel_features_tcev）完全一致：
      ch0  raw_db       原始物理 dB（不归一化）
      ch1  gumbel       Gumbel(Fx)
      ch2  fx           超越概率 F(x)
      ch3  slope_ratio  L2/L1 斜率比（标量广播）
      ch4  bp_db        断点 dB（标量广播）

    参数
    ────
    tcev_grid_dir : 对应网格的 TCEV 断点目录，
                    即 tcev_root/season/grid_XX/
                    若为 None 或目录不存在，像元的 bp_db 全为 NaN，
                    所有像元均被跳过（slope_ratio 无法计算）。

    返回
    ────
    curves : (N, seq_len, 5)  float32
    coords : (N, 2)           int32   (row, col)
    meta   : dict  含 height / width / n_times
    """
    # ── 读取 HDF5 ──────────────────────────────────────────────────
    with h5py.File(
        os.path.join(grid_dir, 'normalized_negative_values.hdf5'), 'r'
    ) as hf:
        height = int(hf.attrs['max_height'])
        width  = int(hf.attrs['max_width'])

    with h5py.File(
        os.path.join(grid_dir, 'sorted_negative_values.hdf5'), 'r'
    ) as hf:
        sorted_db_all = hf['data'][:]   # (T, H, W)

    with h5py.File(
        os.path.join(grid_dir, 'index_values.hdf5'), 'r'
    ) as hf:
        fx_all = hf['data'][:]          # (T, H, W)

    # ── TCEV 断点栅格路径检查 ──────────────────────────────────────
    bp_tif = None
    if tcev_grid_dir and os.path.isdir(tcev_grid_dir):
        candidate = os.path.join(tcev_grid_dir, 'breakpoint_DB.tif')
        if os.path.exists(candidate):
            bp_tif = candidate
        else:
            log.warning(f"  breakpoint_DB.tif 不存在: {tcev_grid_dir}")
    else:
        log.warning(
            f"  tcev_grid_dir 无效或不存在: {tcev_grid_dir}，"
            "该网格所有像元将因 bp_db=NaN 被跳过"
        )

    # ── 如果有 bp_tif，一次性读取整张断点图（避免逐像元 GDAL 开销） ─
    bp_map = None
    if bp_tif is not None:
        try:
            from osgeo import gdal
            ds   = gdal.Open(bp_tif)
            band = ds.GetRasterBand(1)
            bp_map = band.ReadAsArray().astype(np.float32)  # (H_tif, W_tif)
            nd   = band.GetNoDataValue()
            if nd is not None:
                bp_map[np.abs(bp_map - nd) < 1e-3] = np.nan
            ds = None
            log.debug(f"  断点图已加载: {bp_tif}  shape={bp_map.shape}")
        except Exception as e:
            log.warning(f"  断点图读取失败 ({bp_tif}): {e}，像元全部跳过")
            bp_map = None

    # ── 逐像元构建特征 ─────────────────────────────────────────────
    curves_list, coords_list = [], []
    skipped_bp = 0

    for h in range(height):
        for w in range(width):
            # 获取断点值
            if bp_map is not None:
                if h < bp_map.shape[0] and w < bp_map.shape[1]:
                    bp_db_val = float(bp_map[h, w])
                else:
                    bp_db_val = np.nan
            else:
                bp_db_val = np.nan

            if np.isnan(bp_db_val):
                skipped_bp += 1
                continue

            feat = build_pixel_features_tcev(
                sorted_db  = sorted_db_all[:, h, w],
                fx_raw     = fx_all[:, h, w],
                bp_db_val  = bp_db_val,
                seq_len    = seq_len,
                min_valid  = min_valid,
            )
            if feat is None:
                continue

            curves_list.append(feat)
            coords_list.append((h, w))

    if skipped_bp > 0:
        log.debug(f"  跳过 bp=NaN 像元: {skipped_bp}")

    if not curves_list:
        raise ValueError(
            f"网格 {grid_dir} 无有效像元（可能断点 TIF 全部无效）"
        )

    return (
        np.array(curves_list, dtype=np.float32),   # (N, seq_len, 5)
        np.array(coords_list, dtype=np.int32),      # (N, 2)
        dict(height=height, width=width, n_times=sorted_db_all.shape[0]),
    )


# ================================================================
# Dataset / 模型 / 损失
# ================================================================

class FloodCurveDataset(Dataset):
    def __init__(self, curves: np.ndarray, labels: np.ndarray):
        # curves: (N, T, C) → 转置为 (N, C, T) 供 Conv1d 使用
        self.curves = torch.from_numpy(curves.transpose(0, 2, 1))
        self.labels = torch.from_numpy(labels)

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.curves[idx], self.labels[idx]


class FloodNet1D(nn.Module):
    """
    一维卷积分类网络。
    in_channels 从 Config 读取（此处应为 5）。
    """
    def __init__(self, cfg: Config):
        super().__init__()
        layers, in_ch = [], cfg.in_channels
        for out_ch in cfg.conv_channels:
            layers += [
                nn.Conv1d(in_ch, out_ch,
                          kernel_size=cfg.kernel_size,
                          padding=cfg.kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
        self.conv_block = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Linear(in_ch * 2, cfg.fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fc_hidden, 1),
        )

    def forward(self, x):
        feat = self.conv_block(x)
        return self.fc(
            torch.cat([feat.mean(-1), feat.max(-1).values], 1)
        ).squeeze(-1)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=1.0):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, target):
        prob = torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7)
        bce  = -(target * torch.log(prob) + (1 - target) * torch.log(1 - prob))
        fw   = (1 - (prob * target + (1 - prob) * (1 - target))) ** self.gamma
        return ((target * self.pos_weight + (1 - target)) * fw * bce).mean()


# ================================================================
# 评估工具
# ================================================================

def compute_auc_roc(probs, labels):
    pos = labels.sum(); neg = len(labels) - pos
    if pos == 0 or neg == 0: return 0.5
    tprs, fprs = [0.0], [0.0]
    for t in np.linspace(0, 1, 101)[::-1]:
        pp = probs >= t
        tprs.append((pp & (labels == 1)).sum() / pos)
        fprs.append((pp & (labels == 0)).sum() / neg)
    tprs.append(1.0); fprs.append(1.0)
    return float(np.trapz(tprs, fprs))


def compute_f1(probs, labels, threshold=0.5):
    pred = (probs >= threshold).astype(int)
    tp = ((pred == 1) & (labels == 1)).sum()
    fp = ((pred == 1) & (labels == 0)).sum()
    fn = ((pred == 0) & (labels == 1)).sum()
    prec = tp / (tp + fp + 1e-7)
    rec  = tp / (tp + fn + 1e-7)
    return float(prec), float(rec), float(2 * prec * rec / (prec + rec + 1e-7))


# ================================================================
# 训练工具
# ================================================================

def _grid_boundaries(dataset_path: str) -> Dict[int, Tuple[int, int]]:
    with h5py.File(dataset_path, 'r') as hf:
        gi_arr = hf['grid_index'][:]
    return {
        int(gi): (int(np.where(gi_arr == gi)[0][0]),
                  int(np.where(gi_arr == gi)[0][-1]) + 1)
        for gi in np.unique(gi_arr)
    }


def _run_epoch_val(model, val_loader, device, cfg) -> Tuple[float, float]:
    model.eval()
    vp, vl = [], []
    with torch.no_grad():
        for cb, lb in val_loader:
            probs = torch.sigmoid(model(cb.to(device))).cpu().numpy()
            vp.append(probs); vl.append(lb.numpy())
    vp = np.concatenate(vp)
    vl = np.concatenate(vl)
    auc = compute_auc_roc(vp, (vl >= 0.5).astype(int))
    _, _, f1 = compute_f1(vp, (vl >= 0.5).astype(int), cfg.prob_threshold)
    return auc, f1


def _early_stop_update(val_auc, best_auc, model, no_improve):
    if val_auc > best_auc + 1e-4:
        return val_auc, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
    return best_auc, None, no_improve + 1


# ================================================================
# 训练：分块模式（按 grid_index 分块读取，内存友好）
# ================================================================

def train_model_chunked(
    cfg:          Config,
    dataset_path: str,
    grid_keys:    List[str],
) -> Tuple[nn.Module, List[Dict]]:
    device  = cfg.resolve_device()
    pin_mem = cfg.pin_memory and device.type == 'cuda'
    rng     = np.random.default_rng(42)
    log.info(f"设备: {device}  [分块训练]  网格总数: {len(grid_keys)}")

    boundaries = _grid_boundaries(dataset_path)
    valid_gids = sorted(boundaries)
    if not valid_gids:
        raise ValueError("boundaries 为空，请检查 build_dataset_from_points.py 输出。")

    if len(valid_gids) < 2:
        log.warning(f"有效网格数={len(valid_gids)}，退化为全量训练模式。")
        with h5py.File(dataset_path, 'r') as hf:
            return train_model_full(cfg, hf['curves'][:], hf['labels'][:])

    shuffled   = rng.permutation(valid_gids).tolist()
    n_val_g    = max(1, int(len(valid_gids) * cfg.val_ratio))
    val_gids   = shuffled[:n_val_g]
    train_gids = shuffled[n_val_g:]

    with h5py.File(dataset_path, 'r') as hf:
        all_labels_arr = hf['labels'][:]
    n_pos      = int((all_labels_arr >= 0.5).sum())
    pos_weight = float((len(all_labels_arr) - n_pos) / (n_pos + 1e-7))
    log.info(
        f"洪水像元={n_pos} ({100*n_pos/len(all_labels_arr):.1f}%)  "
        f"pos_weight={pos_weight:.2f}"
    )
    del all_labels_arr

    with h5py.File(dataset_path, 'r') as hf:
        val_curves = np.concatenate([
            hf['curves'][boundaries[gi][0]:boundaries[gi][1]]
            for gi in val_gids if gi in boundaries
        ])
        val_labels = np.concatenate([
            hf['labels'][boundaries[gi][0]:boundaries[gi][1]]
            for gi in val_gids if gi in boundaries
        ])
    val_loader = DataLoader(
        FloodCurveDataset(val_curves, val_labels),
        batch_size=cfg.batch_size * 2, shuffle=False,
        num_workers=0, pin_memory=pin_mem,
    )
    log.info(f"验证集像元={len(val_curves)}  训练网格数={len(train_gids)}")
    del val_curves, val_labels

    model      = FloodNet1D(cfg).to(device)
    optimizer  = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler  = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    focal_loss = FocalLoss(gamma=cfg.focal_gamma, pos_weight=pos_weight)
    best_auc, best_state, no_improve, history = 0.0, None, 0, []

    log.info(
        f"\n{'='*65}\n"
        f"{'Epoch':>6} {'TrainLoss':>10} {'ValAUC':>8} {'ValF1':>7} {'LR':>10}\n"
        f"{'='*65}"
    )
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        loss_sum, n_batches = 0.0, 0
        with h5py.File(dataset_path, 'r') as hf:
            valid_train_gids = [gi for gi in train_gids if gi in boundaries]
            for gi in rng.permutation(valid_train_gids).tolist():
                s, e = boundaries[gi]
                chunk_ds = FloodCurveDataset(hf['curves'][s:e], hf['labels'][s:e])
                for cb, lb in DataLoader(
                    chunk_ds, batch_size=cfg.batch_size, shuffle=True,
                    num_workers=0, pin_memory=pin_mem,
                ):
                    optimizer.zero_grad()
                    loss = focal_loss(model(cb.to(device)), lb.to(device))
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    loss_sum += loss.item(); n_batches += 1
        scheduler.step()
        val_auc, val_f1 = _run_epoch_val(model, val_loader, device, cfg)
        lr = optimizer.param_groups[0]['lr']
        log.info(
            f"{epoch:>6} {loss_sum/max(n_batches,1):>10.4f} "
            f"{val_auc:>8.4f} {val_f1:>7.4f} {lr:>10.2e}"
        )
        history.append(dict(epoch=epoch,
                            train_loss=loss_sum / max(n_batches, 1),
                            val_auc=val_auc, val_f1=val_f1, lr=lr))
        new_auc, new_state, no_improve = _early_stop_update(
            val_auc, best_auc, model, no_improve)
        if new_state:
            best_auc, best_state = new_auc, new_state
        if no_improve >= cfg.patience:
            log.info(f"Early stopping (epoch {epoch})  最佳 AUC={best_auc:.4f}")
            break

    log.info(f"{'='*65}\n训练完成  最佳 AUC={best_auc:.4f}")
    if best_state:
        model.load_state_dict(best_state)
    return model.to(device).eval(), history


# ================================================================
# 训练：全量模式
# ================================================================

def train_model_full(
    cfg:        Config,
    all_curves: np.ndarray,
    all_labels: np.ndarray,
) -> Tuple[nn.Module, List[Dict]]:
    device  = cfg.resolve_device()
    pin_mem = cfg.pin_memory and device.type == 'cuda'
    dataset = FloodCurveDataset(all_curves, all_labels)
    n_val   = max(1, int(len(dataset) * cfg.val_ratio))
    train_ds, val_ds = random_split(
        dataset, [len(dataset) - n_val, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=pin_mem)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size * 2, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=pin_mem)

    n_pos      = int((all_labels >= 0.5).sum())
    pos_weight = float((len(all_labels) - n_pos) / (n_pos + 1e-7))
    log.info(
        f"设备: {device}  [全量训练]  洪水像元={n_pos} "
        f"({100*n_pos/len(all_labels):.1f}%)  pos_weight={pos_weight:.2f}"
    )
    model      = FloodNet1D(cfg).to(device)
    optimizer  = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler  = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    focal_loss = FocalLoss(gamma=cfg.focal_gamma, pos_weight=pos_weight)
    best_auc, best_state, no_improve, history = 0.0, None, 0, []

    log.info(
        f"\n{'='*65}\n"
        f"{'Epoch':>6} {'TrainLoss':>10} {'ValAUC':>8} {'ValF1':>7} {'LR':>10}\n"
        f"{'='*65}"
    )
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        loss_sum, n_batches = 0.0, 0
        for cb, lb in train_loader:
            optimizer.zero_grad()
            loss = focal_loss(model(cb.to(device)), lb.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += loss.item(); n_batches += 1
        scheduler.step()
        val_auc, val_f1 = _run_epoch_val(model, val_loader, device, cfg)
        lr = optimizer.param_groups[0]['lr']
        log.info(
            f"{epoch:>6} {loss_sum/max(n_batches,1):>10.4f} "
            f"{val_auc:>8.4f} {val_f1:>7.4f} {lr:>10.2e}"
        )
        history.append(dict(epoch=epoch,
                            train_loss=loss_sum / max(n_batches, 1),
                            val_auc=val_auc, val_f1=val_f1, lr=lr))
        new_auc, new_state, no_improve = _early_stop_update(
            val_auc, best_auc, model, no_improve)
        if new_state:
            best_auc, best_state = new_auc, new_state
        if no_improve >= cfg.patience:
            log.info(f"Early stopping (epoch {epoch})  最佳 AUC={best_auc:.4f}")
            break

    log.info(f"{'='*65}\n训练完成  最佳 AUC={best_auc:.4f}")
    if best_state:
        model.load_state_dict(best_state)
    return model.to(device).eval(), history


# ================================================================
# 推理：逐像元生成硬分类图
# ================================================================

@torch.no_grad()
def infer_flat_map(
    model:  nn.Module,
    curves: np.ndarray,
    coords: np.ndarray,
    meta:   Dict,
    cfg:    Config,
    device: torch.device,
) -> np.ndarray:
    """
    对一个网格的全部有效像元批量推理，返回 (H, W) int8 数组。
    值域：1=洪水  0=非洪水  -1=无效（无数据 / 断点 NaN）
    """
    model.eval()
    H, W      = meta['height'], meta['width']
    label_map = np.full((H, W), -1, dtype=np.int8)

    curves_t   = torch.from_numpy(curves.transpose(0, 2, 1))
    all_logits = []
    for s in range(0, len(curves_t), cfg.batch_size * 4):
        batch = curves_t[s:s + cfg.batch_size * 4].to(device)
        all_logits.append(model(batch).cpu().numpy())
    all_logits = np.concatenate(all_logits)

    if abs(cfg.prob_threshold - 0.5) < 1e-6:
        all_preds = (all_logits >= 0.0).astype(np.int8)
    else:
        all_preds = (
            1 / (1 + np.exp(-all_logits)) >= cfg.prob_threshold
        ).astype(np.int8)

    for idx, (h, w) in enumerate(coords):
        label_map[h, w] = all_preds[idx]
    return label_map


# ================================================================
# 保存推理结果
# ================================================================

def save_label_map(
    label_map:    np.ndarray,
    output_path:  str,
    meta:         Dict,
    cfg:          Config,
    season:       str,
    grid_label:   str,
    source_alias: str = '',
):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    h5_path = output_path + '.hdf5'
    with h5py.File(h5_path, 'w') as hf:
        ds = hf.create_dataset('flat_label', data=label_map,
                               dtype='int8', compression='gzip', compression_opts=4)
        ds.attrs['nodata']  = -1
        ds.attrs['meaning'] = '1=flood  0=non_flood  -1=nodata'
        for k, v in meta.items():
            hf.attrs[str(k)] = str(v)
        hf.attrs.update({
            'season':         season,
            'grid':           grid_label,
            'prob_threshold': cfg.prob_threshold,
            'source_model':   source_alias,
            'feature_mode':   'tcev_5ch',
            'channels':       json.dumps(CHANNEL_DESC),
        })
    np.save(output_path + '.npy', label_map)
    log.info(f"    标签图 → {h5_path}")


# ================================================================
# 单网格推理（内部）
# ================================================================

def _infer_one_grid(
    model:         nn.Module,
    grid_name:     str,
    grid_path:     str,
    cfg:           Config,
    device:        torch.device,
    season_out:    str,
    season:        str,
    seq_len:       int,
    tcev_root:     Optional[str] = None,   # ★ 新增
    source_alias:  str = '',
):
    tag = f"{season}/{grid_name}"
    log.info(f"    推理 {tag} ...")

    # 推导对应网格的 TCEV 断点目录
    tcev_grid_dir = None
    if tcev_root:
        candidate = os.path.join(tcev_root, season, grid_name)
        if os.path.isdir(candidate):
            tcev_grid_dir = candidate
        else:
            log.warning(f"    断点目录不存在: {candidate}")

    try:
        curves, coords, meta = load_grid_curves(
            grid_path, seq_len, tcev_grid_dir=tcev_grid_dir
        )
        label_map = infer_flat_map(model, curves, coords, meta, cfg, device)
        n_valid = int((label_map >= 0).sum())
        n_flood = int((label_map == 1).sum())
        log.info(
            f"      有效像元={n_valid}  洪水={n_flood} "
            f"({100 * n_flood / max(n_valid, 1):.1f}%)"
        )
        out_base = os.path.join(
            season_out, grid_name,
            f'flat_label_{season}_{grid_name}',
        )
        save_label_map(label_map, out_base, meta, cfg, season, grid_name, source_alias)
    except Exception as e:
        log.warning(f"    推理失败 {tag}: {e}")
        traceback.print_exc()


# ================================================================
# 模型持久化工具
# ================================================================

def _save_model(
    model:        nn.Module,
    history:      List[Dict],
    cfg:          Config,
    seq_len:      int,
    save_path:    str,
    dataset_path: str,
    alias:        str,
    season:       str,
):
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config':           {**asdict(cfg), 'seq_len': seq_len},
        'history':          history,
        'dataset_path':     dataset_path,
        'alias':            alias,
        'season':           season,
        'feature_mode':     'tcev_5ch',
        'channel_desc':     CHANNEL_DESC,
    }, save_path)
    log.info(f"  权重已保存 → {save_path}")
    log_path = os.path.join(os.path.dirname(save_path), 'training_log.json')
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def _load_checkpoint(
    path:   str,
    device: torch.device,
) -> Tuple[nn.Module, int]:
    ckpt     = torch.load(path, map_location='cpu', weights_only=True)
    cfg_ckpt = Config(**{k: v for k, v in ckpt['config'].items()
                         if k in Config.__dataclass_fields__})
    # 验证 in_channels
    if cfg_ckpt.in_channels != IN_CHANNELS:
        raise ValueError(
            f"检查点 in_channels={cfg_ckpt.in_channels}，"
            f"期望 {IN_CHANNELS}（TCEV 5通道）"
        )
    model = FloodNet1D(cfg_ckpt).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    seq_len = int(ckpt['config'].get('seq_len', 16))
    log.info(
        f"  权重已加载 ← {path}  "
        f"(seq_len={seq_len}  in_channels={cfg_ckpt.in_channels})"
    )
    return model, seq_len


# ================================================================
# SeasonalRunner：多源 × 多目标 × 多季节推理执行器
# ================================================================

class SeasonalRunner:
    def __init__(
        self,
        cfg:     Config,
        sources: List[SeasonalSourceSpec],
        targets: List[SeasonalTargetSpec],
    ):
        self.cfg     = cfg
        self.sources = sources
        self.targets = targets

    def run(self):
        enabled = [s for s in self.sources if s.enabled]
        if not enabled:
            log.warning("没有启用的训练源，退出。")
            return
        if not self.targets:
            log.warning("未配置任何推理目标，退出。")
            return
        self._print_plan(enabled)
        device = self.cfg.resolve_device()
        log.info(f"运行设备: {device}\n")
        for src in enabled:
            self._run_one_source(src, device)
        log.info(f"\n{'='*65}\n全部推理完成。\n{'='*65}")

    def run_source(self, alias: str):
        src = self._find_source(alias)
        if src is None:
            log.warning(f"run_source: 未找到 alias='{alias}'")
            return
        self._run_one_source(src, self.cfg.resolve_device())

    def run_season(self, season: str):
        if season not in ALL_SEASONS:
            log.warning(f"run_season: 无效季节 '{season}'")
            return
        device = self.cfg.resolve_device()
        for src in self.sources:
            if not src.enabled:
                continue
            if src.season_filter and season not in src.season_filter:
                continue
            model, seq_len = self._prepare_season_model(src, season, device)
            if model is None:
                continue
            for tgt in self.targets:
                if tgt.season_filter and season not in tgt.season_filter:
                    continue
                self._infer_season(model, seq_len, src, tgt, season, device)

    def _run_one_source(self, src: SeasonalSourceSpec, device: torch.device):
        log.info(f"\n{'='*65}")
        log.info(f"训练源：[{src.display_name()}]  dataset_root={src.dataset_root}")
        log.info(f"{'='*65}")
        seasons_in_root = _available_seasons_in_root(src.dataset_root, src.season_filter)
        if not seasons_in_root:
            log.warning(f"  {src.display_name()} 下无可用季节，跳过。")
            return
        for season in seasons_in_root:
            log.info(f"\n  ── 季节: {season.upper()} ──")
            model, seq_len = self._prepare_season_model(src, season, device)
            if model is None:
                continue
            for tgt in self.targets:
                if tgt.season_filter and season not in tgt.season_filter:
                    log.info(f"    目标 '{tgt.display_name()}' 过滤掉季节 {season}，跳过")
                    continue
                log.info(f"  → 目标：[{tgt.display_name()}]  {tgt.hdf5_root}")
                try:
                    self._infer_season(model, seq_len, src, tgt, season, device)
                except Exception as e:
                    log.error(f"  推理失败: {e}")
                    traceback.print_exc()

    def _prepare_season_model(
        self,
        src:    SeasonalSourceSpec,
        season: str,
        device: torch.device,
    ) -> Tuple[Optional[nn.Module], int]:
        model_path   = src.model_save_path(season)
        dataset_path = src.dataset_path(season)

        if os.path.exists(model_path) and not src.force_retrain:
            log.info(f"  加载已有权重: {model_path}")
            try:
                return _load_checkpoint(model_path, device)
            except Exception as e:
                log.warning(f"  权重加载失败，尝试重新训练: {e}")

        if not os.path.exists(dataset_path):
            log.warning(f"  数据集不存在，跳过: {dataset_path}")
            return None, 0

        log.info(f"  训练数据集: {dataset_path}")
        try:
            grid_keys, ds_meta = load_dataset_meta(dataset_path)
        except Exception as e:
            log.warning(f"  元信息读取失败: {e}，跳过 {season}")
            return None, 0

        seq_len = int(ds_meta.get('seq_len', 16))
        try:
            if src.chunk_train:
                model, history = train_model_chunked(self.cfg, dataset_path, grid_keys)
            else:
                with h5py.File(dataset_path, 'r') as hf:
                    curves = hf['curves'][:]
                    labels = hf['labels'][:]
                model, history = train_model_full(self.cfg, curves, labels)
        except Exception as e:
            log.error(f"  训练失败 {season}: {e}")
            traceback.print_exc()
            return None, 0

        _save_model(model, history, self.cfg, seq_len,
                    model_path, dataset_path, src.display_name(), season)
        return model.to(device).eval(), seq_len

    def _infer_season(
        self,
        model:   nn.Module,
        seq_len: int,
        src:     SeasonalSourceSpec,
        tgt:     SeasonalTargetSpec,
        season:  str,
        device:  torch.device,
    ):
        grid_dirs  = _scan_seasonal_grids(tgt.hdf5_root, season, tgt.grid_filter)
        tcev_root  = tgt.resolve_tcev_root()   # ★ 解析 TCEV 根目录

        if not grid_dirs:
            log.warning(f"    {tgt.display_name()}/{season}：无合法网格，跳过")
            return
        season_out = os.path.join(tgt.output_dir, src.display_name(), season)
        os.makedirs(season_out, exist_ok=True)
        log.info(
            f"    季节={season}  网格数={len(grid_dirs)}  "
            f"tcev_root={tcev_root or '(未配置)'}  "
            f"结果目录={season_out}"
        )
        for grid_name, grid_path in grid_dirs:
            _infer_one_grid(
                model, grid_name, grid_path,
                self.cfg, device, season_out, season, seq_len,
                tcev_root    = tcev_root,      # ★ 传入
                source_alias = src.display_name(),
            )

    def _find_source(self, alias: str) -> Optional[SeasonalSourceSpec]:
        for s in self.sources:
            if s.alias == alias or s.display_name() == alias:
                return s
        return None

    def _print_plan(self, enabled_sources: List[SeasonalSourceSpec]):
        log.info(f"\n{'='*65}")
        log.info(f"SeasonalRunner (TCEV 5通道版) 执行计划")
        log.info(f"  通道: {CHANNEL_DESC}")
        log.info(f"  启用的训练源 ({len(enabled_sources)}):")
        for s in enabled_sources:
            seasons = s.season_filter or list(ALL_SEASONS)
            log.info(f"    · [{s.display_name()}]  季节={seasons}")
            log.info(f"      dataset_root={s.dataset_root}")
            log.info(f"      model_root  ={s.model_root or '(auto)'}  "
                     f"force_retrain={s.force_retrain}  chunk={s.chunk_train}")
        log.info(f"  推理目标 ({len(self.targets)}):")
        for t in self.targets:
            seasons = t.season_filter or list(ALL_SEASONS)
            log.info(f"    · [{t.display_name()}]  季节={seasons}")
            log.info(f"      hdf5_root ={t.hdf5_root}")
            log.info(f"      tcev_root ={t.tcev_root or '(自动推断)'}") # ★
            log.info(f"      output_dir={t.output_dir}")
            if t.grid_filter:
                log.info(f"      网格过滤: {t.grid_filter}")
        log.info(f"{'='*65}\n")


# ================================================================
# 便捷入口：单区域季节训练+就地推理
# ================================================================

def run_seasonal_train_and_infer(
    cfg:           Config,
    dataset_root:  str,
    model_root:    str,
    hdf5_root:     str,
    output_dir:    str,
    tcev_root:     Optional[str]       = None,   # ★ 新增
    alias:         str                 = 'region',
    season_filter: Optional[List[str]] = None,
    grid_filter:   Optional[List[str]] = None,
    chunk_train:   bool                = True,
    force_retrain: bool                = False,
) -> Dict[str, str]:
    runner = SeasonalRunner(
        cfg=cfg,
        sources=[SeasonalSourceSpec(
            dataset_root  = dataset_root,
            model_root    = model_root,
            alias         = alias,
            season_filter = season_filter,
            enabled       = True,
            chunk_train   = chunk_train,
            force_retrain = force_retrain,
        )],
        targets=[SeasonalTargetSpec(
            hdf5_root     = hdf5_root,
            output_dir    = output_dir,
            tcev_root     = tcev_root,     # ★ 传入
            season_filter = season_filter,
            grid_filter   = grid_filter,
            alias         = alias,
        )],
    )
    runner.run()
    src = runner.sources[0]
    return {
        s: src.model_save_path(s)
        for s in (season_filter or list(ALL_SEASONS))
        if os.path.exists(src.model_save_path(s))
    }


# ================================================================
# 入口示例
# ================================================================

if __name__ == '__main__':

    cfg = Config(
        in_channels    = IN_CHANNELS,   # 5  （勿修改）
        epochs         = 100,
        batch_size     = 512,
        lr             = 1e-3,
        weight_decay   = 1e-4,
        val_ratio      = 0.2,
        patience       = 20,
        focal_gamma    = 2.0,
        prob_threshold = 0.5,
        device         = 'auto',
        num_workers    = 0,
        pin_memory     = True,
    )

    runner = SeasonalRunner(
        cfg=cfg,
        sources=[
            SeasonalSourceSpec(
                # ★ 指向 build_dataset_from_points.py 的 OUTPUT_DIR
                #   结构：dataset_root/season/dataset_cache.hdf5
                dataset_root  = r'/home/data/ql2024/flood-tly/guangxi/data/tcev/point_dataset_label',
                model_root    = r'/home/data/ql2024/flood-tly/guangxi/data/tcev/flood_models_tcev',
                alias         = 'tcev-季节',
                season_filter = ['autumn'],
                enabled       = True,
                chunk_train   = True,
                force_retrain = True,
            ),
        ],
        targets=[
            SeasonalTargetSpec(
                hdf5_root  = r'/home/data/ql2024/flood-tly/guangxi/data/hdf5_seasonal',
                output_dir = r'/home/data/ql2024/flood-tly/guangxi/data/tcev/flood_result_tcev',
                # ★ 指向 tcev_seasonal 目录（与 build_dataset_from_points.py 的
                #   BASE_TCEV_DIR 一致），结构：tcev_root/season/grid_XX/breakpoint_DB.tif
                tcev_root  = r'/home/data/ql2024/flood-tly/guangxi/data/tcev/tcev_seasonal',
                season_filter = None,
                grid_filter   = None,
                alias         = '广西',
            ),
        ],
    )

    runner.run_source('tcev-季节')