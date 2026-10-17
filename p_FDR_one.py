import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


# =========================================================
# 1. 路径与参数
# =========================================================

RESULT_DIR = r"./12_ROI_emotion/results_two_stage"

SUBJECT_LEVEL_CSV = os.path.join(
    RESULT_DIR,
    "summary_roi_subject_level.csv"
)

OUT_SUBJECT_SUMMARY = os.path.join(
    RESULT_DIR,
    "summary_bootstrap_one_sided_subject_level.csv"
)

OUT_EMOTION_ROI_SUMMARY = os.path.join(
    RESULT_DIR,
    "summary_bootstrap_one_sided_by_emotion_roi.csv"
)

OUT_OVERALL_ROI_SUMMARY = os.path.join(
    RESULT_DIR,
    "summary_bootstrap_one_sided_overall_roi.csv"
)

OUT_FIG_PNG = os.path.join(
    RESULT_DIR,
    "bootstrap_one_sided_fdr_voxel_count_by_emotion.png"
)

# OUT_FIG_PDF = os.path.join(
#     RESULT_DIR,
#     "bootstrap_one_sided_fdr_voxel_count_by_emotion.pdf"
# )

BOOTSTRAP_REPEATS = 2000
FDR_ALPHA = 0.05
SEED = 42

ONLY_FIRST_12_ROI = True


# =========================================================
# 2. 基础函数
# =========================================================

def pearson_r_vectorized(y_true, y_pred):
    """
    向量化计算所有体素的 Pearson r。
    y_true, y_pred: shape = (N, V)
    return: shape = (V,)
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    y_true_c = y_true - y_true.mean(axis=0, keepdims=True)
    y_pred_c = y_pred - y_pred.mean(axis=0, keepdims=True)

    cov = np.sum(y_true_c * y_pred_c, axis=0)
    denom = np.sqrt(
        np.sum(y_true_c ** 2, axis=0) *
        np.sum(y_pred_c ** 2, axis=0)
    )

    r = cov / np.where(denom < 1e-12, np.nan, denom)
    return r


def fdr_bh(p_values, alpha=0.05):
    """
    Benjamini-Hochberg FDR 校正。
    输入：
        p_values: 每个体素的 p 值
    输出：
        reject: q < alpha 的布尔数组
        q_values: FDR q 值
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
    reject[valid] = q_values[valid] < alpha

    return reject, q_values


def bootstrap_one_sided_p_values(y_true, y_pred, n_boot=2000, seed=42):
    """
    对每个体素做 bootstrap 单侧检验。

    原假设近似理解为：该体素的预测相关不稳定，不能稳定大于 0。
    备择假设：
        H1: r > 0

    p_boot 的计算：
        p_boot = (1 + count(r_boot <= 0)) / (B + 1)

    这个 p 值的含义：
        bootstrap 重采样中，有多少比例的相关系数没有大于 0。
        p_boot 越小，说明该体素的正相关越稳定。
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    n_samples, n_voxels = y_true.shape

    r_obs = pearson_r_vectorized(y_true, y_pred)

    std_mask = (
        np.std(y_true, axis=0) > 1e-8
    ) & (
        np.std(y_pred, axis=0) > 1e-8
    ) & np.isfinite(r_obs)

    count_le_zero = np.zeros(n_voxels, dtype=np.int64)

    rng = np.random.RandomState(seed)
    sample_indices = np.arange(n_samples)

    for _ in tqdm(range(n_boot), desc="Bootstrap", leave=False):
        idx = rng.choice(sample_indices, size=n_samples, replace=True)

        yt = y_true[idx]
        yp = y_pred[idx]

        r_boot = pearson_r_vectorized(yt, yp)

        # 单侧 H1: r > 0
        bad = np.isfinite(r_boot) & (r_boot <= 0)
        count_le_zero += bad.astype(np.int64)

    p_boot = (count_le_zero + 1.0) / (n_boot + 1.0)

    # 无效体素设为 nan
    p_boot[~std_mask] = np.nan

    return r_obs, p_boot, std_mask


def find_result_dir(row):
    """
    优先使用 summary_roi_subject_level.csv 中保存的 result_dir。
    如果该路径不存在，则根据 emotion/subject/roi_order 尝试 glob 查找。
    """
    if "result_dir" in row and isinstance(row["result_dir"], str):
        d = row["result_dir"]
        if os.path.exists(d):
            return d

        # 有些情况下 CSV 中是 Windows 路径或相对路径，可以尝试重新拼接
        d2 = os.path.normpath(d)
        if os.path.exists(d2):
            return d2

    emotion = row["emotion"]
    subject = row["subject"]
    roi_order = int(row["roi_order"])

    pattern = os.path.join(
        RESULT_DIR,
        str(emotion),
        str(subject),
        f"roi_{roi_order:02d}_*",
        "full"
    )

    candidates = glob.glob(pattern)
    if len(candidates) > 0:
        return candidates[0]

    return None


# =========================================================
# 3. 主分析：逐 subject × emotion × ROI 重算 bootstrap 单侧 FDR
# =========================================================

summary_df = pd.read_csv(SUBJECT_LEVEL_CSV)

if ONLY_FIRST_12_ROI:
    summary_df = summary_df[summary_df["roi_order"] <= 12].copy()

rows = []

for idx, row in tqdm(
    summary_df.iterrows(),
    total=len(summary_df),
    desc="Reanalyzing subject × emotion × ROI"
):
    subject = row["subject"]
    emotion = row["emotion"]
    roi_order = int(row["roi_order"])
    roi_id = int(row["roi_id"])
    roi_name = row["roi_name"]

    result_dir = find_result_dir(row)

    if result_dir is None:
        print(f"[跳过] 找不到结果目录: {subject} | {emotion} | ROI {roi_order}")
        continue

    y_true_path = os.path.join(result_dir, "y_true.npy")
    y_pred_path = os.path.join(result_dir, "y_pred.npy")
    voxel_stats_path = os.path.join(result_dir, "voxelwise_stats_with_fdr.csv")

    if not os.path.exists(y_true_path) or not os.path.exists(y_pred_path):
        print(f"[跳过] 缺少 y_true.npy 或 y_pred.npy: {result_dir}")
        continue

    y_true = np.load(y_true_path)
    y_pred = np.load(y_pred_path)

    if y_true.shape != y_pred.shape:
        print(f"[跳过] y_true/y_pred shape 不一致: {result_dir}")
        print(f"       y_true={y_true.shape}, y_pred={y_pred.shape}")
        continue

    # 读取原始 voxel 信息，主要为了保留 local/global voxel index
    if os.path.exists(voxel_stats_path):
        voxel_df = pd.read_csv(voxel_stats_path)
    else:
        voxel_df = pd.DataFrame({
            "subject": subject,
            "emotion": emotion,
            "roi_order": roi_order,
            "roi_id": roi_id,
            "roi_name": roi_name,
            "local_voxel_index_in_roi": np.arange(y_true.shape[1]),
            "global_voxel_index_in_merged_fmri": np.arange(y_true.shape[1]),
        })

    r_obs, p_boot, valid_mask = bootstrap_one_sided_p_values(
        y_true=y_true,
        y_pred=y_pred,
        n_boot=BOOTSTRAP_REPEATS,
        seed=SEED + idx
    )

    fdr_reject, q_boot = fdr_bh(p_boot, alpha=FDR_ALPHA)

    raw_boot_p05 = np.isfinite(p_boot) & (p_boot < 0.05)
    fdr_boot_q05 = fdr_reject

    # 合并保存 voxelwise 结果
    out_voxel_df = voxel_df.copy()

    out_voxel_df["bootstrap_one_sided_pearson_r"] = r_obs
    out_voxel_df["bootstrap_one_sided_p_value"] = p_boot
    out_voxel_df["bootstrap_one_sided_fdr_q_value"] = q_boot
    out_voxel_df["bootstrap_one_sided_raw_p_lt_0_05"] = raw_boot_p05
    out_voxel_df["bootstrap_one_sided_fdr_q_lt_0_05"] = fdr_boot_q05
    out_voxel_df["bootstrap_valid_voxel"] = valid_mask

    # out_voxel_csv = os.path.join(
    #     result_dir,
    #     "voxelwise_bootstrap_one_sided_fdr.csv"
    # )
    #
    # out_sig_fdr_csv = os.path.join(
    #     result_dir,
    #     "bootstrap_one_sided_fdr_q05_voxels.csv"
    # )
    #
    # out_sig_raw_csv = os.path.join(
    #     result_dir,
    #     "bootstrap_one_sided_raw_p05_voxels.csv"
    # )

    # out_voxel_df.to_csv(out_voxel_csv, index=False)
    # out_voxel_df[out_voxel_df["bootstrap_one_sided_fdr_q_lt_0_05"]].to_csv(
    #     out_sig_fdr_csv,
    #     index=False
    # )
    # out_voxel_df[out_voxel_df["bootstrap_one_sided_raw_p_lt_0_05"]].to_csv(
    #     out_sig_raw_csv,
    #     index=False
    # )

    n_boot_raw_p05 = int(raw_boot_p05.sum())
    n_boot_fdr_q05 = int(fdr_boot_q05.sum())

    rows.append({
        "subject": subject,
        "emotion": emotion,
        "roi_order": roi_order,
        "roi_id": roi_id,
        "roi_name": roi_name,
        "n_voxels": int(y_true.shape[1]),
        "n_valid_voxels": int(valid_mask.sum()),
        "n_bootstrap_one_sided_raw_p05_voxels": n_boot_raw_p05,
        "n_bootstrap_one_sided_fdr_q05_voxels": n_boot_fdr_q05,
        "mean_bootstrap_one_sided_r": float(np.nanmean(r_obs)),
        "median_bootstrap_one_sided_r": float(np.nanmedian(r_obs)),
        "mean_bootstrap_one_sided_p": float(np.nanmean(p_boot)),
        "median_bootstrap_one_sided_p": float(np.nanmedian(p_boot)),
        "result_dir": result_dir,
    })


# =========================================================
# 4. 汇总保存
# =========================================================

boot_subject_df = pd.DataFrame(rows)
# boot_subject_df.to_csv(OUT_SUBJECT_SUMMARY, index=False)

boot_emotion_roi_df = boot_subject_df.groupby(
    ["emotion", "roi_order", "roi_id", "roi_name"],
    as_index=False
).agg(
    mean_n_bootstrap_one_sided_raw_p05_voxels=(
        "n_bootstrap_one_sided_raw_p05_voxels", "mean"
    ),
    std_n_bootstrap_one_sided_raw_p05_voxels=(
        "n_bootstrap_one_sided_raw_p05_voxels", "std"
    ),
    mean_n_bootstrap_one_sided_fdr_q05_voxels=(
        "n_bootstrap_one_sided_fdr_q05_voxels", "mean"
    ),
    std_n_bootstrap_one_sided_fdr_q05_voxels=(
        "n_bootstrap_one_sided_fdr_q05_voxels", "std"
    ),
    mean_n_valid_voxels=("n_valid_voxels", "mean"),
    mean_r=("mean_bootstrap_one_sided_r", "mean"),
    median_r=("median_bootstrap_one_sided_r", "mean"),
    n_subjects=("subject", "nunique")
)

# boot_emotion_roi_df.to_csv(OUT_EMOTION_ROI_SUMMARY, index=False)

boot_overall_roi_df = boot_subject_df.groupby(
    ["roi_order", "roi_id", "roi_name"],
    as_index=False
).agg(
    mean_n_bootstrap_one_sided_raw_p05_voxels=(
        "n_bootstrap_one_sided_raw_p05_voxels", "mean"
    ),
    std_n_bootstrap_one_sided_raw_p05_voxels=(
        "n_bootstrap_one_sided_raw_p05_voxels", "std"
    ),
    mean_n_bootstrap_one_sided_fdr_q05_voxels=(
        "n_bootstrap_one_sided_fdr_q05_voxels", "mean"
    ),
    std_n_bootstrap_one_sided_fdr_q05_voxels=(
        "n_bootstrap_one_sided_fdr_q05_voxels", "std"
    ),
    mean_n_valid_voxels=("n_valid_voxels", "mean"),
    mean_r=("mean_bootstrap_one_sided_r", "mean"),
    median_r=("median_bootstrap_one_sided_r", "mean"),
    n_subjects=("subject", "nunique"),
    n_emotions=("emotion", "nunique")
)

# boot_overall_roi_df.to_csv(OUT_OVERALL_ROI_SUMMARY, index=False)


# =========================================================
# 5. 打印结果
# =========================================================

print("\n================ Bootstrap 单侧 FDR：按 emotion × ROI 汇总 ================\n")

display_cols = [
    "emotion",
    "roi_order",
    "roi_name",
    "mean_n_bootstrap_one_sided_fdr_q05_voxels",
    "std_n_bootstrap_one_sided_fdr_q05_voxels",
    "mean_n_bootstrap_one_sided_raw_p05_voxels",
    "n_subjects"
]

print(
    boot_emotion_roi_df[display_cols]
    .sort_values(["emotion", "roi_order"])
    .to_string(index=False)
)

print("\n保存完成：")
print(f"  subject-level: {OUT_SUBJECT_SUMMARY}")
print(f"  emotion × ROI: {OUT_EMOTION_ROI_SUMMARY}")
print(f"  overall ROI:   {OUT_OVERALL_ROI_SUMMARY}")


# =========================================================
# 6. 画柱状图：每个情绪类别下 12 个 ROI 的 bootstrap 单侧 FDR 显著体素数量
# =========================================================

plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "Arial Unicode MS",
    "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False

plot_df = boot_emotion_roi_df.copy()
plot_df = plot_df.sort_values(["emotion", "roi_order"])

emotions = ["positive", "negative", "neutral"]
emotions = [e for e in emotions if e in plot_df["emotion"].unique()]

fig, axes = plt.subplots(
    nrows=len(emotions),
    ncols=1,
    figsize=(16, 4.8 * len(emotions)),
    sharey=False
)

if len(emotions) == 1:
    axes = [axes]

for ax, emotion in zip(axes, emotions):
    sub = plot_df[plot_df["emotion"] == emotion].copy()
    sub = sub.sort_values("roi_order")

    x = np.arange(len(sub))
    y = sub["mean_n_bootstrap_one_sided_fdr_q05_voxels"].values
    yerr = sub["std_n_bootstrap_one_sided_fdr_q05_voxels"].fillna(0).values

    labels = [
        f'{int(row["roi_order"])} | {row["roi_name"]}'
        for _, row in sub.iterrows()
    ]

    bars = ax.bar(
        x,
        y,
        yerr=yerr,
        capsize=4,
        width=0.68,
        edgecolor="black",
        linewidth=0.8
    )

    for bar, value in zip(bars, y):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

    ax.set_title(
        f"{emotion}：Bootstrap 单侧 FDR q < 0.05 显著体素数量",
        fontsize=16,
        fontweight="bold",
        pad=12
    )

    ax.set_ylabel("显著体素数量\nbootstrap one-sided FDR q < 0.05", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)

    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    n_subjects = int(sub["n_subjects"].max()) if len(sub) > 0 else 0
    ax.text(
        0.995,
        0.94,
        f"跨被试平均，N={n_subjects}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="gray",
            alpha=0.75
        )
    )

fig.suptitle(
    "每个情绪类别下 12 个显著 ROI 的 Bootstrap 单侧 FDR 显著体素数量",
    fontsize=20,
    fontweight="bold",
    y=0.995
)

plt.tight_layout(rect=[0, 0, 1, 0.97])

plt.savefig(OUT_FIG_PNG, dpi=300, bbox_inches="tight")
# plt.savefig(OUT_FIG_PDF, bbox_inches="tight")
plt.show()

print(f"\n柱状图已保存：")
print(f"  {OUT_FIG_PNG}")
# print(f"  {OUT_FIG_PDF}")