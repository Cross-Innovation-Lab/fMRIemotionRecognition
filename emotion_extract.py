import os
import numpy as np
import pandas as pd
from p_PCA import load_subject_data, extract_roi_features

# ==================== 路径配置 ====================
FMRI_DIR = r"..\data\extract_fmri"
LABELS_DIR = r"..\data\fmri\emotion"            # 注意：这里是目录，不是文件
LABELS_EXCEL_FULL = os.path.join(LABELS_DIR, "output.xlsx")                 # 如果需要读取 image ID
HCPex_ATLAS = r"..\data\HCPex\HCPex_v1.0\HCPex_22Systems.nii"
HCPex_LABELS_CSV = r"..\data\HCPex\HCPex_v1.0\HCPex_22Systems_Labels.csv"

# 显著 ROI 结果目录（第一步生成的 CSV）
SIG_DIR = r".\data\emotion_decoding_3c"

# 输出目录
OUT_DIR = r".\data\12_ROI_emotion"
os.makedirs(OUT_DIR, exist_ok=True)

# 被试列表
SUBJECTS = ['sub01', 'sub02', 'sub03', 'sub04', 'sub05', 'sub06', 'sub07', 'sub08']

# 情绪名称映射（与 CSV 中的情绪名称一致）
EMOTIONS = ['negative', 'neutral', 'positive']


def main():
    hcpex_df = pd.read_csv(HCPex_LABELS_CSV)
    name_to_id = dict(zip(hcpex_df['Name'], hcpex_df['ID']))

    EMOTION_ID_MAP = {
        'negative': 0,
        'positive': 1,
        'neutral': 2,
    }

    for emo in EMOTIONS:
        emo_id = EMOTION_ID_MAP[emo]
        sig_csv = os.path.join(SIG_DIR, f"GROUP_LEVEL_PERMUTATION_UAR.csv")
        if not os.path.exists(sig_csv):
            print(f"⚠ 缺少 {emo} 的显著 ROI 文件: {sig_csv}，跳过")
            continue

        sig_df = pd.read_csv(sig_csv)
        roi_names = sig_df['ROI'].drop_duplicates().tolist()

        roi_ids = []
        valid_roi_names = []
        for name in roi_names:
            if name in name_to_id:
                roi_ids.append(name_to_id[name])
                valid_roi_names.append(name)
            else:
                print(f"⚠ ROI 名称不在 HCPex 标签中: {name}")

        if not roi_ids:
            print(f"⚠ 情绪 {emo}: 无有效 ROI ID，跳过")
            continue

        print(f"\n===== 情绪: {emo} | 显著 ROI: {roi_names} | ID: {roi_ids} =====")

        # 4. 逐被试处理
        for sub in SUBJECTS:
            print(f"  处理被试 {sub} ...")

            data = load_subject_data(sub, FMRI_DIR, LABELS_DIR, HCPex_ATLAS)
            if data is None:
                print(f"    无法加载数据，跳过")
                continue

            fmri_data = data['fMRI_data']
            atlas_data = data['atlas_data']
            cocoids = data['cocoids']
            emotions_3class = data['emotions_3class']

            n_trials = fmri_data.shape[3]
            if not (n_trials == len(cocoids) == len(emotions_3class)):
                print(
                    f"    错误: trial 数不一致 | "
                    f"fMRI={n_trials}, cocoids={len(cocoids)}, emotions={len(emotions_3class)}"
                )
                continue

            features_list = []
            for roi_id in roi_ids:
                X = extract_roi_features(fmri_data, atlas_data, roi_id)
                if X is not None and X.shape[0] > 0:
                    features_list.append(X)
                else:
                    roi_name = [k for k, v in name_to_id.items() if v == roi_id][0] if roi_id in name_to_id.values() else '?'
                    print(f"    ROI {roi_id} ({roi_name}) 无体素或提取失败")

            if not features_list:
                print(f"    被试 {sub} 未提取到任何体素，跳过")
                continue

            combined = np.concatenate(features_list, axis=1)  # (N_images, V)
            if combined.shape[0] != n_trials:
                print(
                    f"    错误: ROI 特征 trial 数 {combined.shape[0]} "
                    f"≠ fMRI trial 数 {n_trials}"
                )
                continue

                # 按当前情绪筛选 trial
            emo_mask = emotions_3class == emo_id

            n_emo = int(np.sum(emo_mask))
            if n_emo == 0:
                print(f"    被试 {sub} 在情绪 {emo} 下无样本，跳过")
                continue

            combined_emo = combined[emo_mask]
            cocoids_emo = cocoids[emo_mask]

            # 保存 cocoid，格式化为 12 位字符串
            image_ids_emo = np.array(
                [str(int(x)).zfill(12) for x in cocoids_emo],
                dtype='<U12'
            )

            out_path = os.path.join(OUT_DIR, f"{sub}_{emo}_fmri.npz")

            np.savez_compressed(
                out_path,
                fmri=combined_emo.astype(np.float32),
                image_ids=image_ids_emo,
                cocoids=cocoids_emo.astype(np.int64),
                emotion=np.array([emo] * n_emo),
                emotion_id=np.array([emo_id] * n_emo),
                roi_names=np.array(valid_roi_names),
                roi_ids=np.array(roi_ids)
            )

            print(
                f"    已保存: {out_path} | "
                f"样本数: {combined_emo.shape[0]} | "
                f"体素数: {combined_emo.shape[1]}"
            )

        print("\n导出完成！")


if __name__ == "__main__":
    main()