import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, t as student_t
from PIL import Image
import clip
from tqdm import tqdm

from model.rigde import RidgeCVEstimator
from p_PCA import *

# ==================== 配置 ====================
cfg = {
    # ---- 数据路径 ----
    "caption_csv": r"..\image_text_pairs_test20_cleaned.csv",   # 建议使用清洗后的文件
    "image_dir": r"..\data\fmri\emotion\coco(nsd)",
    # fMRI 数据路径模式，{subject}和{emotion}将在运行时替换
    "fmri_path_pattern": r".\data\12_ROI_emotion\{subject}_{emotion}_fmri.npz",
    "subjects": ['sub01', 'sub02', 'sub03', 'sub04', 'sub05', 'sub06', 'sub07', 'sub08'],
    "emotions": ['positive', 'negative', 'neutral'],   # 三种情绪
    # ---- 模型 ----
    "backbone": "RN50",                 # or "RN50"
    "finetune_stage1": False,
    "stage1_epochs": 30,
    "stage1_lr": 1e-5,
    "stage2_epochs": 50,
    "stage2_lr": 1e-5,
    "ridge_alpha": 1.0,
    "mse_weight": 0.1,
    "bootstrap_repeats": 2000,
    # ---- 训练 ----
    "batch_size": 32,
    "n_folds": 5,
    "seed": 42,
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "output_dir": "../retnet/12_ROI_emotion/results_two_stage",   # 基础输出目录
    "clip_resolution": 224,
    "pca_variance": 0.95,
}

# ----------------- 辅助 -----------------
torch.manual_seed(cfg['seed'])
np.random.seed(cfg['seed'])

def load_image(img_id, image_dir, preprocess):
    base = os.path.join(image_dir, f"{img_id}_face_0")
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        fpath = base + ext
        if os.path.exists(fpath):
            return preprocess(Image.open(fpath).convert("RGB"))
    raise FileNotFoundError(f"Image {img_id} not found in {image_dir}")

class CaptionDataset(Dataset):
    def __init__(self, caption_csv, image_dir, preprocess):
        self.df = pd.read_csv(caption_csv, dtype={'image_id': str})
        self.image_dir = image_dir
        self.preprocess = preprocess
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = load_image(row['image_id'], self.image_dir, self.preprocess)
        caption = row['caption']
        return img, caption

class fMRI_Dataset(Dataset):
    def __init__(self, image_ids, fmri_responses, image_dir, preprocess):
        self.image_ids = image_ids
        self.fmri = fmri_responses
        self.image_dir = image_dir
        self.preprocess = preprocess
    def __len__(self): return len(self.image_ids)
    def __getitem__(self, idx):
        img = load_image(self.image_ids[idx], self.image_dir, self.preprocess)
        fmri = torch.tensor(self.fmri[idx], dtype=torch.float32)
        return img, fmri, self.image_ids[idx]

class RidgeReadout(nn.Module):
    def __init__(self, in_features, out_features, alpha=1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.alpha = alpha
    def forward(self, x):
        return self.linear(x)
    def ridge_loss(self):
        return self.alpha * (self.linear.weight.pow(2).sum())

# ----------------- 阶段一（全局，只要运行一次） -----------------
def stage1_finetune(cfg):
    print("="*50)
    print("Stage 1: Emotion-Aware CLIP Finetuning")
    print("="*50)
    model, preprocess = clip.load(cfg['backbone'], device=cfg['device'], jit=False)
    model = model.float()
    for p in model.token_embedding.parameters(): p.requires_grad = False
    for p in model.transformer.parameters(): p.requires_grad = False
    model.logit_scale.requires_grad = False
    for p in model.visual.parameters(): p.requires_grad = True

    dataset = CaptionDataset(cfg['caption_csv'], cfg['image_dir'], preprocess)
    loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True)
    optimizer = optim.AdamW(model.visual.parameters(), lr=cfg['stage1_lr'])
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(cfg['stage1_epochs']):
        total_loss = 0.0
        for imgs, caps in tqdm(loader, desc=f"Stage1 Epoch {epoch+1}"):
            imgs = imgs.to(cfg['device'])
            caps_text = clip.tokenize(caps).to(cfg['device'])
            optimizer.zero_grad()
            image_features = model.encode_image(imgs)
            text_features = model.encode_text(caps_text)
            image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-8)
            text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-8)
            logit_scale = model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()
            labels = torch.arange(imgs.size(0), device=cfg['device'])
            loss_i = loss_fn(logits_per_image, labels)
            loss_t = loss_fn(logits_per_text, labels)
            loss = (loss_i + loss_t) / 2
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss/len(loader):.4f}")

    os.makedirs(cfg['output_dir'], exist_ok=True)
    backbone_name = cfg['backbone'].replace('/', '_')
    save_path = os.path.join(cfg['output_dir'], f"emotion_aware_clip_{backbone_name}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Stage 1 Checkpoint saved to {save_path}")
    return model, preprocess

from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import ShuffleSplit, PredefinedSplit
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, t as student_t
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

def pearson_r_vectorized(y_true, y_pred):
    """向量化计算所有体素的 Pearson r，返回 (V,) 数组。"""
    y_true_c = y_true - y_true.mean(axis=0, keepdims=True)
    y_pred_c = y_pred - y_pred.mean(axis=0, keepdims=True)
    cov = (y_true_c * y_pred_c).sum(axis=0)
    denom = np.sqrt((y_true_c ** 2).sum(axis=0) * (y_pred_c ** 2).sum(axis=0))
    r = cov / np.where(denom < 1e-12, np.nan, denom)
    return r

def r2_score_np(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    r2 = np.where(ss_tot > 1e-8, 1 - ss_res / ss_tot, np.nan)
    return r2

def r2_score(Real, Pred):
    # print(Real.shape)
    # print(Pred.shape)
    SSres = np.mean((Real - Pred) ** 2, 0)
    # print(SSres.shape)
    SStot = np.var(Real, 0)
    # print(SStot.shape)
    return np.nan_to_num(1 - SSres / SStot)

cfg.update({
        "fmri_dir": r"..\data\extract_fmri",
        "labels_dir": r"..\data\fmri\emotion",
        "hcpex_atlas": r"..\data\HCPex\HCPex_v1.0\HCPex_22Systems.nii",
        "hcpex_labels_csv": r"..\data\HCPex\HCPex_v1.0\HCPex_22Systems_Labels.csv",
        # 显著 ROI 结果目录：与 emotion_extract.py 中 SIG_DIR 保持一致
        "sig_roi_dir": r".\data\emotion_decoding_3c",
        # FDR 阈值
        "fdr_alpha": 0.05,
    })

import os

def safe_filename(name):
    """把 ROI 名称转换成适合保存为文件夹名的字符串。"""
    text = str(name)
    for ch in ['\\', '/', ':', '*', '?', '"', '<', '>', '|', ' ', '\t', '\n', '\r']:
        text = text.replace(ch, '_')
    while '__' in text:
        text = text.replace('__', '_')
    return text.strip('_')[:120]


def fdr_bh(p_values, alpha=0.05):
    """
    Benjamini-Hochberg FDR 校正。
    返回 reject_mask 和 q_values，reject_mask 表示 q < alpha。
    """
    p_values = np.asarray(p_values, dtype=np.float64)
    q_values = np.full_like(p_values, np.nan, dtype=np.float64)
    reject = np.zeros_like(p_values, dtype=bool)

    valid = np.isfinite(p_values)
    p_valid = p_values[valid]
    m = len(p_valid)
    if m == 0:
        return reject, q_values

    order = np.argsort(p_valid)
    ranked_p = p_valid[order]
    ranks = np.arange(1, m + 1, dtype=np.float64)

    q_ordered = ranked_p * m / ranks
    q_ordered = np.minimum.accumulate(q_ordered[::-1])[::-1]
    q_ordered = np.clip(q_ordered, 0.0, 1.0)

    q_valid = np.empty_like(q_ordered)
    q_valid[order] = q_ordered

    q_values[valid] = q_valid
    reject[valid] = q_valid < alpha
    return reject, q_values


def pearson_r_p_vectorized(y_true, y_pred):
    """
    向量化计算每个体素的 Pearson r 和双侧 p 值。
    y_true, y_pred: (N, V)
    """
    r = pearson_r_vectorized(y_true, y_pred)
    n = y_true.shape[0]
    p = np.full_like(r, np.nan, dtype=np.float64)
    if n < 3:
        return r, p

    valid = np.isfinite(r)
    r_safe = np.clip(r[valid], -0.999999999999, 0.999999999999)
    t_stat = r_safe * np.sqrt((n - 2) / np.maximum(1.0 - r_safe ** 2, 1e-12))
    p[valid] = 2.0 * student_t.sf(np.abs(t_stat), df=n - 2)
    return r, p


def load_significant_rois_for_emotion(cfg, emotion):
    """
    读取每种情绪对应的显著 ROI 名称，并映射到 HCPex ROI ID。
    这里复用 emotion_extract.py 的逻辑：sig_rois_{emotion}.csv 中 ROI 列决定拼接顺序。
    """
    sig_csv = os.path.join(cfg.get('sig_roi_dir', r'.\data\emotion_decoding_3c'),
                           f"GROUP_LEVEL_PERMUTATION_UAR.csv")
    if not os.path.exists(sig_csv):
        print(f"  Missing significant ROI file: {sig_csv}")
        return []

    hcpex_df = pd.read_csv(cfg['hcpex_labels_csv'])
    name_to_id = dict(zip(hcpex_df['Name'], hcpex_df['ID']))

    sig_df = pd.read_csv(sig_csv)
    if 'ROI' not in sig_df.columns:
        print(f"  Missing ROI column in {sig_csv}")
        return []

    roi_infos = []
    for order, roi_name in enumerate(sig_df['ROI'].tolist(), start=1):
        if roi_name not in name_to_id:
            print(f"  ROI name not found in HCPex label csv: {roi_name}")
            continue
        roi_infos.append({
            'roi_order': order,
            'roi_name': roi_name,
            'roi_id': int(name_to_id[roi_name]),
        })
    return roi_infos


def build_roi_slices_from_original_extraction(subject_id, emotion, cfg, total_voxels):
    """
    根据 emotion_extract.py 的提取顺序还原合并矩阵中的 ROI 列范围。

    emotion_extract.py 中的逻辑是：
        for roi_id in roi_ids:
            X = extract_roi_features(..., roi_id)
            features_list.append(X)
        combined = np.concatenate(features_list, axis=1)

    因此这里只需要按相同 ROI 顺序重新计算每个 ROI 的体素数，
    即可得到 merged fmri 矩阵中每个 ROI 对应的 [start, end) 列范围。
    """
    roi_infos = load_significant_rois_for_emotion(cfg, emotion)
    if not roi_infos:
        return []

    data = load_subject_data(subject_id, cfg['fmri_dir'], cfg['labels_dir'], cfg['hcpex_atlas'])
    if data is None:
        print(f"  Cannot load original fMRI/atlas data for ROI slicing: {subject_id}")
        return []

    fmri_data = data['fMRI_data']
    atlas_data = data['atlas_data']

    roi_slices = []
    start_col = 0
    for roi_info in roi_infos:
        roi_id = roi_info['roi_id']
        roi_name = roi_info['roi_name']
        X_roi = extract_roi_features(fmri_data, atlas_data, roi_id)
        if X_roi is None or X_roi.ndim != 2 or X_roi.shape[1] == 0:
            print(f"  ROI {roi_id} ({roi_name}) has no voxels, skipping")
            continue

        n_vox = int(X_roi.shape[1])
        end_col = start_col + n_vox
        if start_col >= total_voxels:
            print(f"  Reconstructed ROI columns exceed merged fMRI columns at ROI {roi_id} ({roi_name}); stopping")
            break
        if end_col > total_voxels:
            print(f"  Warning: ROI {roi_id} ({roi_name}) end column {end_col} > total voxels {total_voxels}; truncating")
            end_col = total_voxels

        item = dict(roi_info)
        item.update({
            'start_col': int(start_col),
            'end_col': int(end_col),
            'n_voxels_extracted': int(n_vox),
            'n_voxels_in_merged': int(end_col - start_col),
        })
        roi_slices.append(item)
        start_col += n_vox

    if start_col != total_voxels:
        print(f"  Warning: reconstructed ROI voxel count = {start_col}, merged fMRI voxel count = {total_voxels}")
    return roi_slices


def stage2_encoding_eval_advanced(
    clip_model, preprocess, cfg, subject_id, emotion,
    seed=42,
    test_ratio=0.15,          # 测试集占比
    n_sessions=None,          # 若提供列表（如 [5,10,15,...]），则做递增训练量评估；否则只用全部训练数据评估一次
    bootstrap_repeats=2000,
    save_intermediate=True
):
    """
    单 ROI 编码评估版：
    - 保留原始 CLIP 特征提取、随机打乱、RidgeCVEstimator、bootstrap 等训练设置；
    - 根据 HCPex 显著 ROI 的原始拼接顺序，将合并后的 fMRI 矩阵拆成单 ROI；
    - 每个 ROI 单独训练 Ridge 编码模型；
    - 每个 ROI 内对体素级 Pearson p 值做 FDR 校正，并保存 q < 0.05 的体素。
    """
    print("\n" + "=" * 50)
    print(f"Single-ROI Encoding Model: {emotion} | {subject_id}")
    print("=" * 50)

    # ---- 1. 加载合并后的 fMRI ----
    fmri_path = cfg['fmri_path_pattern'].format(subject=subject_id, emotion=emotion)
    if not os.path.exists(fmri_path):
        print(f"  fMRI npz not found: {fmri_path}, skipping")
        return None
    data_npz = np.load(fmri_path)
    fmri_all = data_npz['fmri'].astype(np.float64)

    # ---- 2. 对齐图像 ID：保留原始读取逻辑 ----
    # ---- 2. 对齐图像 ID：优先使用 npz 中保存的 image_ids/cocoids ----
    if 'image_ids' in data_npz.files:
        image_ids = data_npz['image_ids'].astype(str).tolist()
    elif 'cocoids' in data_npz.files:
        cocoids = data_npz['cocoids'].astype(np.int64)
        image_ids = [str(int(cid)).zfill(12) for cid in cocoids]
    else:
        raise ValueError(f"{fmri_path} 中没有 image_ids 或 cocoids，无法和图片对齐")

    if len(image_ids) != fmri_all.shape[0]:
        raise ValueError(
            f"image_ids 数量和 fMRI trial 数不一致: "
            f"image_ids={len(image_ids)}, fMRI={fmri_all.shape[0]}"
        )

    print(f"  Loaded merged fMRI: {fmri_all.shape[1]} voxels, {fmri_all.shape[0]} trials")
    print(f"  First 3 image IDs: {image_ids[:3]}")

    # ---- 3. 提取 CLIP 特征：保留原始逻辑 ----
    clip_model.eval()
    all_feats = []
    bs = cfg['batch_size']
    with torch.no_grad():
        for i in range(0, len(image_ids), bs):
            batch_ids = image_ids[i:i+bs]
            imgs = [load_image(img_id, cfg['image_dir'], preprocess) for img_id in batch_ids]
            imgs = torch.stack(imgs).to(cfg['device'])
            feats = clip_model.encode_image(imgs).cpu().numpy()
            all_feats.append(feats)
    X = np.concatenate(all_feats, axis=0)
    print(f"  CLIP features: {X.shape}, merged fMRI: {fmri_all.shape}")

    # ---- 4. 还原每个 ROI 在合并矩阵中的列范围 ----
    roi_slices = build_roi_slices_from_original_extraction(
        subject_id=subject_id,
        emotion=emotion,
        cfg=cfg,
        total_voxels=fmri_all.shape[1]
    )
    if not roi_slices:
        print("  No valid ROI slices reconstructed, skipping")
        return None
    print(f"  Reconstructed {len(roi_slices)} ROI slices")

    # ---- 5. 固定种子打乱：所有 ROI 共用完全相同的样本顺序 ----
    rng = np.random.RandomState(seed)
    order = np.arange(len(X))
    rng.shuffle(order)
    X_shuf = X[order].copy()
    print(f"  Data shuffled with seed {seed}")

    # ---- 6. 划分测试集：所有 ROI 共用完全相同的训练/测试划分 ----
    n_total = len(X_shuf)
    n_test = int(n_total * test_ratio)
    X_test = torch.from_numpy(X_shuf[-n_test:]).float()
    X_pool = torch.from_numpy(X_shuf[:-n_test]).float()
    print(f"  Test size: {n_test}, training pool: {len(X_pool)}")

    out_base = os.path.join(cfg['output_dir'], emotion, subject_id)
    os.makedirs(out_base, exist_ok=True)

    def train_and_evaluate_roi(X_tr, y_tr, X_te, y_te, roi_info,
                               kept_local_indices, kept_global_indices,
                               session_label="full"):
        """单个 ROI 的训练、评估、FDR 和保存。"""
        roi_id = roi_info['roi_id']
        roi_name = roi_info['roi_name']
        roi_order = roi_info['roi_order']
        fdr_alpha = cfg.get('fdr_alpha', 0.05)

        # ---- y 标准化（基于训练集）：保留原始逻辑 ----
        y_mean = y_tr.mean(dim=0, keepdim=True)
        y_std = y_tr.std(dim=0, keepdim=True)
        y_std[y_std < 1e-8] = 1.0
        y_tr_scaled = (y_tr - y_mean) / y_std
        y_te_scaled = (y_te - y_mean) / y_std

        # ---- 交叉验证拆分 ----
        # 注意：这里使用 PredefinedSplit 的标准写法：-1 为训练集，0 为验证集。
        # 训练参数、alpha 范围和 scoring 均保持原代码一致。
        inner_cv = ShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
        test_fold = np.full(len(X_tr), -1, dtype=int)
        for _, val_idx in inner_cv.split(X_tr):
            test_fold[val_idx] = 0
        kfold = PredefinedSplit(test_fold)

        # ---- 训练：保留 RidgeCVEstimator 与 alpha 搜索逻辑 ----
        alphas = torch.logspace(-8, 0.5 * np.log10(X_tr.shape[1]) + 8, 100)
        estimator = RidgeCVEstimator(
            alphas, kfold,
            scoring=lambda y, yhat: -torch.nn.functional.mse_loss(yhat, y),
            scale_X=True
        )
        estimator.fit(X_tr, y_tr_scaled)
        y_pred = estimator.predict(X_te).cpu().numpy()
        y_true = y_te_scaled.cpu().numpy()

        weights, biases = estimator.get_model_weights_and_bias()

        # ---- 体素级 Pearson r、p、r² 和 FDR ----
        r_all, p_all = pearson_r_p_vectorized(y_true, y_pred)
        std_mask = (np.std(y_true, axis=0) > 1e-8) & (np.std(y_pred, axis=0) > 1e-8)
        valid_mask = std_mask & np.isfinite(r_all) & np.isfinite(p_all)

        fdr_reject, q_all = fdr_bh(p_all, alpha=fdr_alpha)
        raw_p_mask = np.isfinite(p_all) & (p_all < 0.05)

        r_valid = r_all[valid_mask]
        r2_valid = r_valid ** 2

        median_r = np.nanmedian(r_valid) if len(r_valid) > 0 else np.nan
        mean_r = np.nanmean(r_valid) if len(r_valid) > 0 else np.nan
        median_r2 = np.nanmedian(r2_valid) if len(r2_valid) > 0 else np.nan
        mean_r2 = np.nanmean(r2_valid) if len(r2_valid) > 0 else np.nan
        max_r2 = np.nanmax(r2_valid) if len(r2_valid) > 0 else np.nan
        min_r2 = np.nanmin(r2_valid) if len(r2_valid) > 0 else np.nan
        mse = np.mean((y_true - y_pred) ** 2)

        # ---- Bootstrap R² (r²) 中位数分布：保留原始逻辑 ----
        r2_medians = []
        idx_all = np.arange(len(y_true))
        rng_bs = np.random.RandomState(seed + 1)
        for _ in tqdm(range(bootstrap_repeats), desc=f"Bootstrap ROI {roi_order:02d}", leave=False):
            idx_bs = rng_bs.choice(idx_all, size=len(idx_all), replace=True)
            yt = y_true[idx_bs]
            yp = y_pred[idx_bs]
            r_boot = pearson_r_vectorized(yt, yp)
            r_boot = r_boot[std_mask]
            r_boot = r_boot[np.isfinite(r_boot)]
            if len(r_boot) > 0:
                r2_medians.append(np.nanmedian(r_boot ** 2))
            else:
                r2_medians.append(np.nan)
        r2_medians = np.array(r2_medians)
        boot_mean = np.nanmean(r2_medians)
        boot_std = np.nanstd(r2_medians)

        # ---- 保存：每个 ROI 单独目录 ----
        roi_dir_name = f"roi_{roi_order:02d}_id{roi_id}_{safe_filename(roi_name)}"
        sub_dir = os.path.join(out_base, roi_dir_name, session_label)
        if save_intermediate:
            os.makedirs(sub_dir, exist_ok=True)
            np.save(os.path.join(sub_dir, "y_pred.npy"), y_pred)
            np.save(os.path.join(sub_dir, "y_true.npy"), y_true)
            np.save(os.path.join(sub_dir, "weights.npy"), weights.cpu().numpy())
            np.save(os.path.join(sub_dir, "biases.npy"), biases.cpu().numpy())
            np.save(os.path.join(sub_dir, "best_alpha_indices.npy"), estimator.best_l_idxs.cpu().numpy())
            np.save(os.path.join(sub_dir, "r2_bootstrap_dist.npy"), r2_medians)
            np.save(os.path.join(sub_dir, "kept_local_voxel_indices.npy"), kept_local_indices)
            np.save(os.path.join(sub_dir, "kept_global_voxel_indices_in_merged_fmri.npy"), kept_global_indices)
            np.save(os.path.join(sub_dir, "fdr_q05_local_voxel_indices.npy"), kept_local_indices[fdr_reject])
            np.save(os.path.join(sub_dir, "fdr_q05_global_voxel_indices_in_merged_fmri.npy"), kept_global_indices[fdr_reject])
            np.save(os.path.join(sub_dir, "raw_p05_local_voxel_indices.npy"), kept_local_indices[raw_p_mask])
            np.save(os.path.join(sub_dir, "raw_p05_global_voxel_indices_in_merged_fmri.npy"), kept_global_indices[raw_p_mask])

            voxel_stats = pd.DataFrame({
                'subject': subject_id,
                'emotion': emotion,
                'roi_order': roi_order,
                'roi_id': roi_id,
                'roi_name': roi_name,
                'local_voxel_index_in_roi': kept_local_indices,
                'global_voxel_index_in_merged_fmri': kept_global_indices,
                'pearson_r': r_all,
                'corr_r2': r_all ** 2,
                'p_value': p_all,
                'fdr_q_value': q_all,
                'fdr_q_lt_0_05': fdr_reject,
                'raw_p_lt_0_05': raw_p_mask,
            })
            voxel_stats.to_csv(os.path.join(sub_dir, "voxelwise_stats_with_fdr.csv"), index=False)
            voxel_stats[voxel_stats['fdr_q_lt_0_05']].to_csv(
                os.path.join(sub_dir, "fdr_significant_voxels_q05.csv"), index=False)
            voxel_stats[voxel_stats['raw_p_lt_0_05']].to_csv(
                os.path.join(sub_dir, "raw_p_significant_voxels_p05.csv"), index=False)

        print(f"\n--- ROI {roi_order:02d} | {roi_name} | ID {roi_id} Results ---")
        print(f"  Voxels original/used: {roi_info['n_voxels_in_merged']}/{len(kept_local_indices)}")
        print(f"  Median r = {median_r:.4f}, Mean r = {mean_r:.4f}")
        print(f"  Median R² = {median_r2:.4f}, Mean R² = {mean_r2:.4f}, Max R² = {max_r2:.4f}, Min R² = {min_r2:.4f}")
        print(f"  MSE = {mse:.4f}")
        print(f"  FDR q < {fdr_alpha}: {int(fdr_reject.sum())} voxels | raw p < 0.05: {int(raw_p_mask.sum())} voxels")
        print(f"  Bootstrap R² median: mean = {boot_mean:.4f}, std = {boot_std:.4f}")

        return {
            'subject': subject_id,
            'emotion': emotion,
            'roi_order': roi_order,
            'roi_id': roi_id,
            'roi_name': roi_name,
            'roi_start_col': roi_info['start_col'],
            'roi_end_col': roi_info['end_col'],
            'n_voxels_original_roi': roi_info['n_voxels_in_merged'],
            'n_voxels_used': len(kept_local_indices),
            'n_valid_voxels_for_r': int(valid_mask.sum()),
            'n_fdr_q05_voxels': int(fdr_reject.sum()),
            'n_raw_p05_voxels': int(raw_p_mask.sum()),
            'train_size': len(X_tr),
            'test_size': len(X_te),
            'median_pearson_r': median_r,
            'mean_pearson_r': mean_r,
            'median_r2': median_r2,
            'mean_r2': mean_r2,
            'max_r2': max_r2,
            'min_r2': min_r2,
            'mse': mse,
            'bootstrap_r2_mean': boot_mean,
            'bootstrap_r2_std': boot_std,
            'result_dir': sub_dir,
        }

    # ---- 7. 逐 ROI 单独训练 ----
    results_list = []
    for roi_info in roi_slices:
        roi_name = roi_info['roi_name']
        roi_id = roi_info['roi_id']
        start_col = roi_info['start_col']
        end_col = roi_info['end_col']
        print("\n" + "-" * 50)
        print(f"Processing ROI {roi_info['roi_order']:02d}: {roi_name} | ID {roi_id} | cols [{start_col}, {end_col})")
        print("-" * 50)

        y_roi_raw = fmri_all[:, start_col:end_col]
        if y_roi_raw.shape[1] == 0:
            print(f"  Empty ROI after slicing, skipping: {roi_name}")
            continue

        y_roi_shuf = y_roi_raw[order].copy()

        # 保留原代码“先去除近似零方差体素”的处理，但改为在每个 ROI 内单独执行。
        y_var = y_roi_shuf.var(axis=0)
        keep = y_var > 1e-8
        if keep.sum() == 0:
            print(f"  All voxels are near-zero variance in ROI {roi_name}, skipping")
            continue

        kept_local_indices = np.where(keep)[0].astype(np.int64)
        kept_global_indices = (start_col + kept_local_indices).astype(np.int64)
        y_roi_shuf = y_roi_shuf[:, keep]
        print(f"  Removed {(~keep).sum()} near-zero-variance voxels, remaining: {y_roi_shuf.shape[1]}")

        y_test = torch.from_numpy(y_roi_shuf[-n_test:]).float()
        y_pool = torch.from_numpy(y_roi_shuf[:-n_test]).float()

        if n_sessions is not None:
            piece = len(X_pool) // 40
            for n in n_sessions:
                n_tr = min(n * piece, len(X_pool))
                X_tr = X_pool[:n_tr]
                y_tr = y_pool[:n_tr]
                label = f"session_{n}"
                res = train_and_evaluate_roi(
                    X_tr, y_tr, X_test, y_test, roi_info,
                    kept_local_indices, kept_global_indices,
                    session_label=label
                )
                res['session'] = n
                results_list.append(res)
        else:
            res = train_and_evaluate_roi(
                X_pool, y_pool, X_test, y_test, roi_info,
                kept_local_indices, kept_global_indices,
                session_label="full"
            )
            results_list.append(res)

    if not results_list:
        print("  No ROI produced valid results")
        return None

    df = pd.DataFrame(results_list)
    subject_summary_path = os.path.join(out_base, f"{subject_id}_{emotion}_single_roi_summary.csv")
    df.to_csv(subject_summary_path, index=False)
    print(f"\nSaved subject-level single ROI summary: {subject_summary_path}")

    avg_pearson = df['median_pearson_r'].mean()
    avg_r2 = df['median_r2'].mean()
    return df, avg_pearson, avg_r2

# ==================== 主流程 ====================
if __name__ == "__main__":
    os.makedirs(cfg['output_dir'], exist_ok=True)

    # 阶段一（全局）
    if cfg['finetune_stage1']:
        emotion_clip, preprocess = stage1_finetune(cfg)
    else:
        backbone = cfg['backbone']
        emotion_clip, preprocess = clip.load(backbone, device=cfg['device'], jit=False)
        print("Using original CLIP weights (no finetuning).")
    emotion_clip = emotion_clip.float()

    # 收集所有 ROI 结果用于最终汇总
    summary_rows = []

    for emo in cfg['emotions']:
        print(f"\n{'=' * 60}")
        print(f"Processing emotion: {emo}")
        print(f"{'=' * 60}")
        for subj in cfg['subjects']:
            res = stage2_encoding_eval_advanced(
                emotion_clip, preprocess, cfg,
                subject_id=subj, emotion=emo
            )
            if res is not None:
                results_df, avg_pearson, avg_r2 = res
                summary_rows.extend(results_df.to_dict('records'))

    # 汇总跨被试、跨 ROI 指标
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        subject_level_path = os.path.join(cfg['output_dir'], 'summary_roi_subject_level.csv')
        summary_df.to_csv(subject_level_path, index=False)

        final_by_emotion_roi = summary_df.groupby(
            ['emotion', 'roi_order', 'roi_id', 'roi_name']
        ).agg(
            mean_median_pearson=('median_pearson_r', 'mean'),
            std_median_pearson=('median_pearson_r', 'std'),
            mean_mean_pearson=('mean_pearson_r', 'mean'),
            std_mean_pearson=('mean_pearson_r', 'std'),

            mean_median_r2=('median_r2', 'mean'),
            std_median_r2=('median_r2', 'std'),
            mean_mean_r2=('mean_r2', 'mean'),
            std_mean_r2=('mean_r2', 'std'),

            mean_mse=('mse', 'mean'),
            std_mse=('mse', 'std'),

            mean_n_fdr_q05_voxels=('n_fdr_q05_voxels', 'mean'),
            mean_n_raw_p05_voxels=('n_raw_p05_voxels', 'mean'),
            mean_n_voxels_used=('n_voxels_used', 'mean'),
            n_subjects=('subject', 'nunique')
        ).reset_index()

        final_by_emotion_roi_path = os.path.join(cfg['output_dir'], 'summary_roi_cross_subject_by_emotion.csv')
        final_by_emotion_roi.to_csv(final_by_emotion_roi_path, index=False)

        final_overall_roi = summary_df.groupby(
            ['roi_order', 'roi_id', 'roi_name']
        ).agg(
            mean_mse=('mse', 'mean'),
            std_mse=('mse', 'std'),
            mean_median_pearson=('median_pearson_r', 'mean'),
            std_median_pearson=('median_pearson_r', 'std'),
            mean_median_r2=('median_r2', 'mean'),
            std_median_r2=('median_r2', 'std'),
            mean_n_fdr_q05_voxels=('n_fdr_q05_voxels', 'mean'),
            mean_n_raw_p05_voxels=('n_raw_p05_voxels', 'mean'),
            mean_n_voxels_used=('n_voxels_used', 'mean'),
            n_subjects=('subject', 'nunique'),
            n_emotions=('emotion', 'nunique')
        ).reset_index()

        final_overall_roi_path = os.path.join(cfg['output_dir'], 'summary_roi_overall.csv')
        final_overall_roi.to_csv(final_overall_roi_path, index=False)

        print("\n==================== 单 ROI 跨被试平均指标：按 emotion × ROI ====================")
        print(final_by_emotion_roi.to_string(index=False))
        print(f"\nSaved ROI subject-level summary: {subject_level_path}")
        print(f"Saved ROI cross-subject-by-emotion summary: {final_by_emotion_roi_path}")
        print(f"Saved ROI overall summary: {final_overall_roi_path}")
    else:
        print("\n未产生任何有效 ROI 结果。")
