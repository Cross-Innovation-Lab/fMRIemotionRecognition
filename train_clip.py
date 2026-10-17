import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from PIL import Image
import clip
from tqdm import tqdm

from model.rigde import RidgeCVEstimator
from p_PCA import *

# ==================== 配置 ====================
cfg = {
    # ---- 数据路径 ----
    "caption_csv": r"..\image_text_pairs_test20_cleaned.csv",
    "image_dir": r"..\data\fmri\emotion\coco(nsd)",
    "fmri_path_pattern": r"..\data\12_ROI_emotion\{subject}_{emotion}_fmri.npz",
    "subjects": ['sub01', 'sub02', 'sub03', 'sub04', 'sub05', 'sub06', 'sub07', 'sub08'],
    "emotions": ['positive', 'negative', 'neutral'],
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
    "output_dir": "./retnet/results_two_stage",   # 基础输出目录
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
from scipy.stats import pearsonr
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
    })

import os

def stage2_encoding_eval_advanced(
    clip_model, preprocess, cfg, subject_id, emotion,
    seed=42,
    test_ratio=0.15,          # 测试集占比
    n_sessions=None,          # 若提供列表（如 [5,10,15,...]），则做递增训练量评估；否则只用全部训练数据评估一次
    bootstrap_repeats=2000,
    save_intermediate=True
):

    print("\n" + "=" * 50)
    print(f"Encoding Model (advanced): {emotion} | {subject_id}")
    print("=" * 50)

    # ---- 1. 加载 fMRI ----
    fmri_path = cfg['fmri_path_pattern'].format(subject=subject_id, emotion=emotion)
    if not os.path.exists(fmri_path):
        print(f"  fMRI npz not found: {fmri_path}, skipping")
        return None
    data_npz = np.load(fmri_path)
    fmri_all = data_npz['fmri'].astype(np.float64)

    # ---- 2. 对齐图像 ID ----
    if 'image_ids' in data_npz.files:
        image_ids = data_npz['image_ids'].astype(str).tolist()
    elif 'cocoids' in data_npz.files:
        cocoids = data_npz['cocoids'].astype(np.int64)
        image_ids = [str(int(cid)).zfill(12) for cid in cocoids]
    else:
        raise ValueError(f"{fmri_path} 中没有 image_ids 或 cocoids，无法和图片对齐")

    # 强制检查
    if len(image_ids) != fmri_all.shape[0]:
        raise ValueError(
            f"image_ids 数量和 fMRI trial 数不一致: "
            f"image_ids={len(image_ids)}, fMRI={fmri_all.shape[0]}"
        )

    print(f"  Loaded {fmri_all.shape[1]} voxels, {fmri_all.shape[0]} trials")
    print(f"  First 3 image IDs: {image_ids[:3]}")

    # ---- 3. 提取 CLIP 特征 ----
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
    y_raw = fmri_all
    print(f"  CLIP features: {X.shape}, fMRI: {y_raw.shape}")

    # ---- 4. 固定种子打乱 ----
    rng = np.random.RandomState(seed)
    order = np.arange(len(X))
    rng.shuffle(order)
    X_shuf = X[order].copy()
    y_shuf = y_raw[order].copy()
    print(f"  Data shuffled with seed {seed}")

    # ---- 5. 去除响应方差≈0的体素（基于训练集会变化，但此处先全局去除零方差）----
    y_var = y_shuf.var(axis=0)
    keep = y_var > 1e-8
    y_shuf = y_shuf[:, keep]
    print(f"  Removed {(~keep).sum()} zero-variance voxels, remaining: {y_shuf.shape[1]}")

    # ---- 6. 划分测试集 ----
    n_total = len(X_shuf)
    n_test = int(n_total * test_ratio)
    X_test = torch.from_numpy(X_shuf[-n_test:]).float()
    y_test = torch.from_numpy(y_shuf[-n_test:]).float()
    X_pool = torch.from_numpy(X_shuf[:-n_test]).float()
    y_pool = torch.from_numpy(y_shuf[:-n_test]).float()
    print(f"  Test size: {n_test}, training pool: {len(X_pool)}")

    out_base = os.path.join(cfg['output_dir'], emotion, subject_id)
    os.makedirs(out_base, exist_ok=True)

    # ---- 7. 定义训练 & 评估函数 ----
    def train_and_evaluate(X_tr, y_tr, X_te, y_te, session_label=""):
        # ---- y 标准化（基于训练集） ----
        y_mean = y_tr.mean(dim=0, keepdim=True)
        y_std = y_tr.std(dim=0, keepdim=True)
        y_std[y_std < 1e-8] = 1.0
        y_tr_scaled = (y_tr - y_mean) / y_std
        y_te_scaled = (y_te - y_mean) / y_std

        # ---- 交叉验证拆分 ----
        inner_cv = ShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
        test_fold = np.full(len(X_tr), -1)
        for tr_idx, val_idx in inner_cv.split(X_tr):
            test_fold[val_idx] = 0
        kfold = PredefinedSplit(test_fold)

        # ---- 训练 ----
        alphas = torch.logspace(-8, 0.5 * np.log10(X_tr.shape[1]) + 8, 100)
        estimator = RidgeCVEstimator(
            alphas, kfold,
            scoring=lambda y, yhat: -torch.nn.functional.mse_loss(yhat, y),
            scale_X=True
        )
        estimator.fit(X_tr, y_tr_scaled)
        y_pred = estimator.predict(X_te).cpu().numpy()
        y_true = y_te_scaled.cpu().numpy()

        # ---- 无条件获取权重和偏置 ----
        weights, biases = estimator.get_model_weights_and_bias()

        # ---- 评估指标 ----
        r_values = []
        r_squared_values = []
        for v in range(y_true.shape[1]):
            if np.std(y_true[:, v]) > 1e-8 and np.std(y_pred[:, v]) > 1e-8:
                with np.errstate(invalid='ignore'):
                    r, _ = pearsonr(y_true[:, v], y_pred[:, v])
                if np.isfinite(r):
                    r_values.append(r)
                    r_squared_values.append(r ** 2)  # r²
                else:
                    # 无效的 r 仍然可以添加 nan，但后面用 nanmedian 时会忽略
                    r_values.append(np.nan)
                    r_squared_values.append(np.nan)
                # r_vals.append(r if np.isfinite(r) else np.nan)
        median_r = np.nanmedian(r_values) if len(r_values) > 0 else np.nan
        mean_r = np.nanmean(r_values) if len(r_values) > 0 else np.nan
        # rsqs = r2_score(y_true, y_pred)

        mse = np.mean((y_true - y_pred) ** 2)
        median_r2 = np.nanmedian(r_squared_values) if len(r_squared_values) > 0 else np.nan
        max_r2 = np.nanmax(r_squared_values)
        min_r2 = np.nanmin(r_squared_values)

        # Bootstrap 改为基于 r² 分布（向量化，速度快）
        r2_medians = []
        idx_all = np.arange(len(y_true))
        rng_bs = np.random.RandomState(seed + 1)
        n_vox = y_true.shape[1]

        for _ in tqdm(range(bootstrap_repeats), desc="Bootstrap", leave=False):
            idx_bs = rng_bs.choice(idx_all, size=len(idx_all), replace=True)
            yt = y_true[idx_bs]  # (n_bs, v)
            yp = y_pred[idx_bs]  # (n_bs, v)

            # 向量化计算所有体素的 r
            yt_centered = yt - yt.mean(axis=0)
            yp_centered = yp - yp.mean(axis=0)
            cov = (yt_centered * yp_centered).sum(axis=0) / (len(idx_bs) - 1)  # 无偏协方差
            std_yt = yt.std(ddof=1, axis=0)
            std_yp = yp.std(ddof=1, axis=0)
            # 避免除零
            denom = (std_yt * std_yp) + 1e-12
            r = cov / denom
            r_sq = r ** 2
            r2_medians.append(np.nanmedian(r_sq))

        r2_medians = np.array(r2_medians)
        boot_mean = r2_medians.mean()
        boot_std = r2_medians.std()

        # ---- 保存 ----
        if save_intermediate:
            sub_dir = os.path.join(out_base, session_label)
            os.makedirs(sub_dir, exist_ok=True)
            np.save(os.path.join(sub_dir, "y_pred.npy"), y_pred)
            np.save(os.path.join(sub_dir, "y_true.npy"), y_true)
            np.save(os.path.join(sub_dir, "weights.npy"), weights.cpu().numpy())
            np.save(os.path.join(sub_dir, "biases.npy"), biases.cpu().numpy())
            np.save(os.path.join(sub_dir, "best_alpha_indices.npy"),
                    estimator.best_l_idxs.cpu().numpy())
            np.save(os.path.join(sub_dir, "r2_bootstrap_dist.npy"), r2_medians)

        # ---- 打印 ----
        print(f"\n--- Session {session_label if session_label else 'full'} Results ---")
        print(f"  Training samples: {len(X_tr)}")
        print(f"  Median r = {median_r:.4f}, Mean r = {mean_r:.4f}")
        print(f"  Median R² = {median_r2:.4f}, Max R² = {max_r2:.4f}, Min R² = {min_r2:.4f}")
        print(f"  MSE = {mse:.4f}")
        print(f"  Bootstrap R² median: mean = {boot_mean:.4f}, std = {boot_std:.4f}")

        return {
            'train_size': len(X_tr),
            'median_pearson_r': median_r,
            'mean_pearson_r': mean_r,
            'median_r2': median_r2,
            'max_r2': max_r2,
            'min_r2': min_r2,
            'mse': mse,
            'bootstrap_r2_mean': boot_mean,
            'bootstrap_r2_std': boot_std,
            # 'n_voxels': total_voxels,
            'y_pred': y_pred,
            'y_true': y_true,
            'weights': weights.cpu().numpy(),
            'biases': biases.cpu().numpy(),
            'best_alpha_indices': estimator.best_l_idxs.cpu().numpy(),
        }

    cv_flag = True  # 启用交叉验证选 alpha

    # ---- 8. 递增训练量（可选）或仅用全部训练集 ----
    if n_sessions is not None:
        # 模仿参考代码：按比例划分训练池
        results_list = []
        piece = len(X_pool) // 40
        for n in n_sessions:
            n_tr = min(n * piece, len(X_pool))
            X_tr = X_pool[:n_tr]
            y_tr = y_pool[:n_tr]
            label = f"session_{n}"
            print(f"\n--- Training with {n_tr} samples (session {n}) ---")
            res = train_and_evaluate(X_tr, y_tr, X_test, y_test, name=f"session{n}", session_label=label)
            res['session'] = n
            results_list.append(res)
        # 汇总为 DataFrame
        summary = pd.DataFrame(results_list)
        return summary, None, None   # 可根据需要修改返回
    else:
        # 使用全部训练池
        label = f"full"
        print("\n--- Training with all pool data ---")
        res = train_and_evaluate(X_pool, y_pool, X_test, y_test, session_label=label)
        # 添加到结果字典中附加信息
        res.update({
            'subject': subject_id,
            'emotion': emotion,
            'test_size': n_test,
        })
        df = pd.DataFrame([res])
        return df, res['median_pearson_r'], res['median_r2']


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

    # 收集所有结果用於最終汇总
    summary_rows = []

    for emo in cfg['emotions']:
        print(f"\n{'=' * 60}")
        print(f"Processing emotion: {emo}")
        print(f"{'=' * 60}")
        for subj in cfg['subjects']:
            res = stage2_encoding_eval_advanced(emotion_clip, preprocess, cfg,
                                       subject_id=subj, emotion=emo)
            if res is not None:
                results_df, avg_pearson, avg_r2 = res
                summary_rows.append({
                    'emotion': emo,
                    'subject': subj,
                    'avg_pearson': avg_pearson,
                    'avg_r2': avg_r2
                })

    # 汇总跨被试指标
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        final_summary = summary_df.groupby('emotion').agg(
            mean_pearson=('avg_pearson', 'mean'),
            std_pearson=('avg_pearson', 'std'),
            mean_r2=('avg_r2', 'mean'),
            std_r2=('avg_r2', 'std'),
            n_subjects=('subject', 'nunique')
        ).reset_index()
        print("\n==================== 最终跨被试平均指标 ====================")
        print(final_summary.to_string(index=False))
        final_summary.to_csv(os.path.join(cfg['output_dir'], 'summary_cross_subject.csv'), index=False)
    else:
        print("\n未产生任何有效结果。")