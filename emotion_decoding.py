import gc
import os
from datetime import datetime

import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.base import clone
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests
from scipy.stats import combine_pvalues
from tqdm import tqdm
import warnings

from p_PCA import load_subject_data
from p_PCA import extract_roi_features

warnings.filterwarnings("ignore")

# ==================== 路径配置 ====================
FMRI_DIR = r"..\data\extract_fmri"
LABELS_FILE = r"..\data\fmri\emotion"
HCPex_22_ATLAS = r"..\data\HCPex\HCPex_v1.0\HCPex_22Systems.nii"
HCPex_LABELS = r"..\data\HCPex\HCPex_v1.0\HCPex_22Systems_Labels.csv"
OUTPUT_DIR = r".\data\emotion_decoding_3c"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 情绪映射
EMOTION_4CLASS = {'positive': 0, 'negative': 1, 'neutral': 2, 'surprise': 3}
EMOTION_NAMES_4 = {v: k for k, v in EMOTION_4CLASS.items()}

USE_SIG_ROIS_ONLY = True
SIGNIFICANCE_FILE = r".\data\decoding_selection\GROUP_LEVEL_PERMUTATION_UAR_20260429_024557.csv"

# 模型与置换参数
N_SPLITS = 5          # 交叉验证折数
N_PERMS = 3000         # 置换次数（样本内）
N_JOBS = -1            # 并行数
K_BEST = 800           # 特征选择（与多分类一致）

# ==================== 二分类置换检验函数 ====================
def _uar_for_permuted_labels_binary(model, X_test_sel, y_test_perm):
    y_pred = model.predict(X_test_sel)
    return balanced_accuracy_score(y_test_perm, y_pred)

def binary_cv_with_permutation(X, y, model, n_splits=10, n_perms=1000, n_jobs=-1, random_state=42):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = y.astype(int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    folds_info = []
    all_y_true_total = []
    all_y_pred_total = []
    perm_uars_all = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test  = np.nan_to_num(X_test,  nan=0.0, posinf=0.0, neginf=0.0)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        selector = SelectKBest(score_func=f_classif, k=K_BEST)
        selector.fit(X_train_s, y_train)
        X_train_sel             = selector.transform(X_train_s)
        X_test_sel  = selector.transform(X_test_s)

        if X_train_sel.shape[1] == 0:
            continue

        model_fold = clone(model)
        model_fold.fit(X_train_sel, y_train)

        y_pred_real = model_fold.predict(X_test_sel)
        all_y_true_total.extend(y_test)
        all_y_pred_total.extend(y_pred_real)

        # 置换
        rng = np.random.RandomState(random_state + fold)
        y_test_shuffled = np.array([rng.permutation(y_test) for _ in range(n_perms)])
        fold_perm_uars = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(_uar_for_permuted_labels_binary)(model_fold, X_test_sel, y_test_shuffled[i])
            for i in range(n_perms)
        )
        perm_uars_all.extend(fold_perm_uars)

        folds_info.append({
            'fold': fold,
            'y_true': y_test,
            'y_pred': y_pred_real,
            'selected_feature_indices': selector.get_support(indices=True),
            'n_features_original': X_train_s.shape[1],
            'n_features_selected': X_train_sel.shape[1]
        })


    real_uar = balanced_accuracy_score(all_y_true_total, all_y_pred_total)
    perm_uars_all = np.array(perm_uars_all)
    p_value = (np.sum(perm_uars_all >= real_uar) + 1) / (len(perm_uars_all) + 1)
    result = {
        'real_uar': real_uar,
        'perm_uars': perm_uars_all,  # 完整的置换分布
        'perm_uar_mean': np.mean(perm_uars_all),
        'perm_p_value': p_value,
        'folds_info': folds_info,  # 每折的详细数据
        'all_y_true': np.array(all_y_true_total),
        'all_y_pred': np.array(all_y_pred_total),
        'chance_level': 0.5,
        'n_perms': n_perms,
        'n_splits': n_splits
    }
    return result

# ==================== 主体分析 ====================
def run_subject_emotion_binary(subject_id, data, roi_ids, roi_names, output_dir):
    """
    对一名被试的所有 ROI，跑 6 个情绪的二分类，返回该被试的结果列表。
    """
    subject_results = []
    fMRI_data, atlas_data = data['fMRI_data'], data['atlas_data']
    emotions_6 = data['emotions_6class']
    mapping = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2, 5: 3}  # 3 = surprise
    emotions_4 = np.array([mapping[e] for e in emotions_6])

    for roi_id in roi_ids:
        roi_name = roi_names[roi_id]
        X = extract_roi_features(fMRI_data, atlas_data, roi_id)
        if X is None or X.shape[0] < 5:
            continue

        for class_idx in range(4):
            y_binary = (emotions_4 == class_idx).astype(int)
            if np.sum(y_binary) < 5 or np.sum(y_binary == 0) < 5:
                continue

            model = MLPClassifier(random_state=42)
            detail = binary_cv_with_permutation(
                X, y_binary, model, n_splits=N_SPLITS, n_perms=N_PERMS, n_jobs=N_JOBS
            )
            subject_results.append({
                'Subject': subject_id,
                'ROI_ID': roi_id,
                'ROI_Name': roi_name,
                'Emotion': class_idx,
                'Emotion_Name': EMOTION_NAMES_4[class_idx],
                'UAR': detail['real_uar'],
                'Perm_P': detail['perm_p_value'],
                'Perm_UAR_Mean': detail['perm_uar_mean'],
                'binary_details': detail   # 完整字典，将用于保存
            })

    save_subject_results(subject_id, subject_results, output_dir)
    return subject_results


def save_subject_results(subject_id, results_list, output_dir):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    basepath = os.path.join(output_dir, f'{subject_id}_binary_detailed_{timestamp}')

    # 准备一个扁平化的 DF 作为概览（不含数组），方便 Excel 打开
    overview = []
    for res in results_list:
        row = {k: res[k] for k in ['Subject', 'ROI_ID', 'ROI_Name', 'Emotion', 'Emotion_Name',
                                   'UAR', 'Perm_P', 'Perm_UAR_Mean']}
        overview.append(row)
    overview_df = pd.DataFrame(overview)
    overview_df.to_csv(basepath + '_overview.csv', index=False)

    # 保存详细数据到 NPZ：每个 ROI×Emotion 一个键
    npz_dict = {}
    for i, res in enumerate(results_list):
        detail = res.get('binary_details')
        if detail is None:
            continue
        key_prefix = f"roi{res['ROI_ID']}_emo{res['Emotion']}"
        npz_dict[f'{key_prefix}_real_uar'] = detail['real_uar']
        npz_dict[f'{key_prefix}_perm_uars'] = detail['perm_uars']
        npz_dict[f'{key_prefix}_perm_uar_mean'] = detail['perm_uar_mean']
        npz_dict[f'{key_prefix}_perm_p_value'] = detail['perm_p_value']
        npz_dict[f'{key_prefix}_all_y_true'] = detail['all_y_true']
        npz_dict[f'{key_prefix}_all_y_pred'] = detail['all_y_pred']

        # 折叠信息：因为每折有数组，单独扁平化
        for fi, finfo in enumerate(detail['folds_info']):
            fold_key = f'{key_prefix}_fold{fi}'
            npz_dict[f'{fold_key}_y_true'] = finfo['y_true']
            npz_dict[f'{fold_key}_y_pred'] = finfo['y_pred']
            npz_dict[f'{fold_key}_n_feat_orig'] = finfo['n_features_original']
            npz_dict[f'{fold_key}_n_feat_sel'] = finfo['n_features_selected']
            # 特征索引保存为数组
            npz_dict[f'{fold_key}_selected_idx'] = finfo['selected_feature_indices']

    np.savez_compressed(basepath + '_details.npz', **npz_dict)
    print(f"被试 {subject_id} 完整结果已保存: {basepath}_overview.csv 和 _details.npz")
    return basepath

if __name__ == "__main__":
    subjects = ['sub01', 'sub02', 'sub03', 'sub04', 'sub05', 'sub06', 'sub07', 'sub08']

    labels_df = pd.read_csv(HCPex_LABELS)
    if USE_SIG_ROIS_ONLY:
        sig_df = pd.read_csv(SIGNIFICANCE_FILE)
        sig_roi_names = sig_df[sig_df['Significant'] == True]['ROI_Name'].tolist()
        roi_subset = labels_df[labels_df['Name'].isin(sig_roi_names)]
    else:
        roi_subset = labels_df[labels_df['ID'] > 0].copy()

    roi_ids = roi_subset['ID'].astype(int).tolist()
    roi_names = roi_subset.set_index('ID')['Name'].to_dict()
    print(f"分析 ROI 数量: {len(roi_ids)}")

    all_results = []

    for sub in subjects:
        print(f"\n===== 处理被试 {sub} =====")
        data = load_subject_data(sub, FMRI_DIR, LABELS_FILE, HCPex_22_ATLAS)
        if data is None:
            continue
        sub_res = run_subject_emotion_binary(sub, data, roi_ids, roi_names, OUTPUT_DIR)
        all_results.extend(sub_res)
        del data
        gc.collect()

    # 保存个体结果
    df_all = pd.DataFrame(all_results)
    df_all.to_csv(os.path.join(OUTPUT_DIR, 'binary_individual_results.csv'), index=False)

