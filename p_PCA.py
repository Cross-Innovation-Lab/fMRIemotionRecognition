import gc
import os
import sys
import warnings
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectFdr, f_classif, SelectFpr, SelectKBest  # [修改] 引入特征选择
from sklearn.base import clone
from datetime import datetime
from tqdm import tqdm
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests
import warnings

warnings.filterwarnings("ignore", message="The number of unique classes")
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
# 假设 utlies.py 中有 Logger 定义
try:
    from utlies import Logger
except ImportError:
    class Logger:
        def __init__(self, *args, **kwargs): pass

        def write(self, *args): pass

        def flush(self): pass

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# warnings.filterwarnings('ignore')
# sys.stdout = Logger("ML_Train_SelectFpr.log")

# ==================== 配置路径 ====================
FMRI_DIR = r"..\data\extract_fmri"
LABELS_FILE = r"..\data\fmri\emotion"
HCPex_22_ATLAS = r"..\data\HCPex\HCPex_v1.0\HCPex_22Systems.nii"
HCPex_LABELS = r"..\data\HCPex\HCPex_v1.0\HCPex_22Systems_Labels.csv"
OUTPUT_DIR = r"..\data\decoding_selection\all_model"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 情绪标签映射 ====================
EMOTION_6CLASS = {'happy': 0, 'sad': 1, 'angry': 2, 'fear': 3, 'neutral': 4, 'surprise': 5}
EMOTION_3CLASS = {'happy': 0, 'sad': 1, 'angry': 1, 'fear': 1, 'neutral': 2}
EMOTION_3CLASS_NAMES = {0: 'Positive', 1: 'Negative', 2: 'Neutral'}
EMOTION_6CLASS_NAMES = {0: 'Happy', 1: 'Sad', 2: 'Angry', 3: 'Fear', 4: 'Neutral', 5: 'Surprise'}

TASK_CONFIGS = [
    {'task_name': '3Class', 'task_key': '3class', 'label_key': 'emotions_3class', 'chance_level': 1 / 3},
    {'task_name': '6Class', 'task_key': '6class', 'label_key': 'emotions_6class', 'chance_level': 1 / 6},
]

# ==================== 机器学习模型配置 ====================
MODELS = {
    'SVM': {
        'model': SVC(kernel='linear', C=0.1, class_weight='balanced', random_state=42, probability=True),
        'param_name': 'SVC'
    },
    'DT': {
        'model': DecisionTreeClassifier(class_weight='balanced', random_state=42),
        'param_name': 'DecisionTree'
    },
    'RF': {
        'model': RandomForestClassifier(class_weight='balanced', random_state=42),
        'param_name': 'RandomForest'
    },
    'MLP': {
        'model': MLPClassifier(random_state=42),
        'param_name': 'MLP'
    }
}

ALL_RESULTS = []
ALL_RESULTS_3CLASS = []
ALL_RESULTS_6CLASS = []
ALL_PERM_DATA = []


def group_permutation_test(uar_matrix, chance_level, n_perms=1000, random_seed=42):
    """
    对被试×ROI 的 UAR 矩阵进行组水平符号置换检验。

    Parameters
    ----------
    uar_matrix : ndarray of shape (n_subjects, n_rois)
        每个被试在每个ROI上的平衡准确率（UAR），缺失值用 np.nan
    chance_level : float
        理论随机水平（如 1/6）
    n_perms : int
        置换次数
    random_seed : int
        随机种子

    Returns
    -------
    p_values : ndarray of shape (n_rois,)
        每个ROI的单侧置换p值（H1: UAR > chance）
    true_uars : ndarray of shape (n_rois,)
        每个ROI的真实组平均UAR
    """
    n_subj, n_rois = uar_matrix.shape
    rng = np.random.RandomState(random_seed)

    valid_mask = ~np.isnan(uar_matrix).all(axis=0)
    p_values = np.ones(n_rois)
    true_uars = np.nanmean(uar_matrix, axis=0)

    # 差值矩阵：每列是被试的 UAR - chance
    uar_diff = uar_matrix - chance_level  # shape (n_subj, n_rois)

    for i_roi in range(n_rois):
        if not valid_mask[i_roi]:
            continue
        diff_valid = uar_diff[:, i_roi]
        diff_valid = diff_valid[~np.isnan(diff_valid)]
        n_valid = len(diff_valid)
        if n_valid < 2:
            continue

        true_mean_diff = true_uars[i_roi] - chance_level

        null_dist = np.zeros(n_perms)
        for perm in range(n_perms):
            signs = rng.choice([-1, 1], size=n_valid)
            null_dist[perm] = np.mean(diff_valid * signs)

        p_values[i_roi] = (np.sum(null_dist >= true_mean_diff) + 1) / (n_perms + 1)

    return p_values, true_uars


def print_final_summary():
    from scipy.stats import combine_pvalues

    print(f"\n{'=' * 80}")
    print("组水平显著性检验 (FDR校正)")
    print(f"{'=' * 80}")

    if not ALL_RESULTS:
        print("无组水平数据")
        return

    df = pd.DataFrame(ALL_RESULTS)
    if 'Perm_P' not in df.columns:
        print("错误：数据中缺少 'Perm_P' 列，请先运行带有置换检验的解码。")
        return

    group_significant_results = []

    for task_name in sorted(df['Task'].dropna().unique()):
        task_df = df[df['Task'] == task_name].copy()
        if task_df.empty:
            continue

        for model_name in MODELS.keys():
            m_data = task_df[task_df['Model'] == model_name].copy()
            if m_data.empty:
                continue

            roi_pvals = m_data.groupby('ROI_Name')['Perm_P'].apply(list).reset_index()
            roi_uars = m_data.groupby('ROI_Name')['UAR'].mean().reset_index()
            roi_info = roi_pvals.merge(roi_uars, on='ROI_Name')

            group_list = []
            for _, row in roi_info.iterrows():
                pvals = [p for p in row['Perm_P'] if not np.isnan(p)]
                if len(pvals) < 2:
                    combined_p = np.nan
                else:
                    _, combined_p = combine_pvalues(pvals, method='fisher')
                group_list.append({
                    'Task': task_name,
                    'Model': model_name,
                    'ROI_Name': row['ROI_Name'],
                    'Mean_UAR': row['UAR'],
                    'Combined_P': combined_p
                })

            group_df = pd.DataFrame(group_list)
            valid_mask = ~group_df['Combined_P'].isna()
            if valid_mask.any():
                pvals_to_correct = group_df.loc[valid_mask, 'Combined_P'].values
                _, p_fdr, _, _ = multipletests(pvals_to_correct, alpha=0.05, method='fdr_bh')
                group_df['P_FDR'] = np.nan
                group_df.loc[valid_mask, 'P_FDR'] = p_fdr
                group_df['Significant'] = group_df['P_FDR'] < 0.05
            else:
                group_df['P_FDR'] = np.nan
                group_df['Significant'] = False

            # 输出显著 ROI
            sig = group_df[group_df['Significant']]
            print(f"\n{task_name} | {model_name} | 显著ROI数 (FDR<0.05): {len(sig)}")
            if len(sig) > 0:
                for _, row in sig.iterrows():
                    print(f"  {row['ROI_Name']:30s}: Mean UAR={row['Mean_UAR']:.4f}, p_FDR={row['P_FDR']:.4f}")

            group_significant_results.append(group_df)

    # 保存汇总
    if group_significant_results:
        all_group_df = pd.concat(group_significant_results, ignore_index=True)
        group_path = os.path.join(OUTPUT_DIR, f'GROUP_LEVEL_FISHER_FDR.csv')
        all_group_df.to_csv(group_path, index=False, encoding='utf-8-sig')
        print(f"\nFisher合并检验完整结果已保存至: {group_path}")


# ==================== 评估指标计算 ====================
def calculate_metrics(y_true, y_pred, n_classes):
    """计算 7 种评估指标 (新增 AUC)"""
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    class_weights = np.sum(cm, axis=1) / np.sum(cm)
    class_acc = np.diag(cm) / (np.sum(cm, axis=1) + 1e-10)
    war = np.sum(class_weights * class_acc)
    uar = np.mean(class_acc)

    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    return {
        'ACC': acc, 'F1': f1_macro, 'WAR': war, 'UAR': uar,
        'WF1': f1_weighted, 'UF1': f1_macro
    }


# ==================== 加载数据 ====================
def load_subject_data(subject_id, fMRI_dir, labels_file, atlas_path):
    """加载单个被试的 fMRI 数据和标签 (保持不变)"""
    print(f"\n{'=' * 60}")
    print(f"加载被试 {subject_id} 数据")
    print(f"{'=' * 60}")

    fMRI_path = os.path.join(fMRI_dir, f'{subject_id}_emotion_trials.nii.gz')
    if not os.path.exists(fMRI_path):
        print(f"错误：fMRI 文件不存在 {fMRI_path}");
        return None

    fMRI_img = nib.load(fMRI_path)
    fMRI_data = fMRI_img.get_fdata(dtype=np.float32)

    labels_path = os.path.join(labels_file, f'{subject_id}_output.xlsx')
    if not os.path.exists(labels_path):
        print(f"错误：标签文件不存在 {labels_path}");
        return None

    labels_df = pd.read_excel(labels_path)
    required_cols = ['orderid', 'predicted_emotion', 'cocoid']  # ← 添加 'cocoid'
    for col in required_cols:
        if col not in labels_df.columns:
            print(f"错误：标签文件中不存在列 {col}")
            return None

    subject_trials = labels_df.copy()
    if len(subject_trials) == 0:
        return None

    orderids = subject_trials['orderid'].astype(int).values
    sort_idx = np.argsort(orderids)
    orderids = orderids[sort_idx]

    emotions_6class_raw = subject_trials['predicted_emotion'].values[sort_idx]
    cocoids = subject_trials['cocoid'].astype(int).values[sort_idx]  # ← 提取图片ID
    emotions_6class = np.array([EMOTION_6CLASS.get(str(e).lower().strip(), -1) for e in emotions_6class_raw])
    if -1 in emotions_6class:
        print(f"警告：发现未知 6 分类情绪标签，将替换为 neutral (4)")
        emotions_6class[emotions_6class == -1] = 4

    emotions_3class = np.array([EMOTION_3CLASS.get(str(e).lower().strip(), -1) for e in emotions_6class_raw])
    # if -1 in emotions_3class:
    #     print(f"警告：发现未知 3 分类情绪标签，将替换为 neutral (2)")
    #     emotions_3class[emotions_3class == -1] = 2

    if len(orderids) != fMRI_data.shape[3]:
        min_len = min(len(orderids), fMRI_data.shape[3])
        orderids = orderids[:min_len]
        emotions_6class = emotions_6class[:min_len]
        emotions_3class = emotions_3class[:min_len]
        cocoids = cocoids[:min_len]
        fMRI_data = fMRI_data[:, :, :, :min_len]

    atlas_img = nib.load(atlas_path)
    atlas_data = atlas_img.get_fdata()

    from nilearn.image import resample_to_img
    if atlas_data.shape != fMRI_data.shape[:3]:
        print(f"重采样图谱从 {atlas_data.shape} 到 {fMRI_data.shape[:3]}")
        atlas_img = resample_to_img(atlas_img, fMRI_img, interpolation='nearest')
        atlas_data = atlas_img.get_fdata()

    return {
        'fMRI_data': fMRI_data,
        'fMRI_img': fMRI_img,
        'atlas_data': atlas_data,
        'atlas_img': atlas_img,
        'orderids': orderids,
        'cocoids': cocoids,
        'emotions_3class': emotions_3class,
        'emotions_6class': emotions_6class,
        'n_trials': len(orderids)
    }


# ==================== 提取 ROI 特征 ====================
def extract_roi_features(fMRI_data, atlas_data, roi_id, mask_mode='extract'):
    """提取单个 ROI 内的体素信号"""
    roi_mask = (atlas_data == roi_id)
    if np.sum(roi_mask) == 0: return None

    n_trials = fMRI_data.shape[3]
    if mask_mode == 'extract':
        n_voxels = np.sum(roi_mask)
        roi_features = fMRI_data[roi_mask, :].reshape(n_voxels, n_trials).T
    else:
        masked_data = np.zeros_like(fMRI_data)
        masked_data[roi_mask, :] = fMRI_data[roi_mask, :]
        n_voxels_total = fMRI_data.shape[0] * fMRI_data.shape[1] * fMRI_data.shape[2]
        roi_features = masked_data.reshape(n_voxels_total, n_trials).T
    return roi_features

def _uar_for_permuted_labels(model, X_test_sel, y_test_perm):
    """计算打乱标签下的平衡准确率 (UAR)"""
    y_pred = model.predict(X_test_sel)
    return balanced_accuracy_score(y_test_perm, y_pred)


def train_and_evaluate_with_permutation(X, y, model, n_splits=10, n_perms=3000, random_state=42):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = y.astype(int)
    n_classes = len(np.unique(y))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    all_y_true = []
    all_y_pred = []
    cv_scores_real = []
    fold_y_tests = []
    fold_y_preds = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        selector = SelectKBest(score_func=f_classif, k=800)
        selector.fit(X_train_scaled, y_train)
        X_train_sel = selector.transform(X_train_scaled)
        X_test_sel = selector.transform(X_test_scaled)

        if X_train_sel.shape[1] == 0:
            print(f"  Fold {fold}: 无特征，跳过")
            continue

        model_fold = clone(model)
        model_fold.fit(X_train_sel, y_train)

        # 真实标签预测与准确率
        y_pred_real = model_fold.predict(X_test_sel)
        fold_uar_real = balanced_accuracy_score(y_test, y_pred_real)
        cv_scores_real.append(fold_uar_real)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred_real)

        fold_y_tests.append(np.array(y_test))
        fold_y_preds.append(np.array(y_pred_real))

        del X_train, X_test, X_train_scaled, X_test_scaled, X_train_sel, X_test_sel
        gc.collect()

    if len(all_y_true) == 0:
        metrics = {k: np.nan for k in ['ACC', 'F1', 'WAR', 'UAR', 'WF1', 'UF1', 'CV_UAR_Mean', 'CV_UAR_Std']}
        return metrics, np.array([]), np.array([]), np.array([])

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    # 整体真实指标
    metrics = calculate_metrics(all_y_true, all_y_pred, n_classes)

    metrics['CV_UAR_Mean'] = np.mean(cv_scores_real)
    metrics['CV_UAR_Std'] = np.std(cv_scores_real)

    rng = np.random.RandomState(random_state)
    perm_uars = np.zeros(n_perms, dtype=np.float64)

    for i_perm in range(n_perms):
        perm_y_true_all = []
        perm_y_pred_all = []

        for y_test_fold, y_pred_fold in zip(fold_y_tests, fold_y_preds):
            y_test_perm = rng.permutation(y_test_fold)

            perm_y_true_all.extend(y_test_perm)
            perm_y_pred_all.extend(y_pred_fold)

        perm_y_true_all = np.array(perm_y_true_all)
        perm_y_pred_all = np.array(perm_y_pred_all)

        perm_uars[i_perm] = balanced_accuracy_score(perm_y_true_all, perm_y_pred_all)

    return metrics, all_y_true, all_y_pred, perm_uars


# ==================== 保存单个模型结果 ====================
def save_model_results(results_list, subject_id, model_name, task_type, output_dir):
    if len(results_list) == 0:
        print(f"警告：{task_type} 任务无结果可保存");
        return None
    df = pd.DataFrame(results_list)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{subject_id}_{model_name}_{task_type}_results_{timestamp}.csv'
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"{task_type} 结果已保存：{output_path}")
    return df


# ==================== 主解码流程 ====================
def run_decoding(subject_id, data, output_dir):
    print(f"\n{'=' * 80}")
    print(f"开始被试 {subject_id} 的情绪解码分析 (Feature Selection: SelectFpr)")
    print(f"{'=' * 80}")

    labels_df = pd.read_csv(HCPex_LABELS)
    roi_ids = labels_df[labels_df['ID'] > 0]['ID'].astype(int).tolist()
    roi_names = labels_df[labels_df['ID'] > 0].set_index('ID')['Name'].to_dict()
    print(f"\n共 {len(roi_ids)} 个 ROI")

    fMRI_data, atlas_data = data['fMRI_data'], data['atlas_data']
    print(f"3 分类标签分布: {dict(pd.Series(data['emotions_3class']).value_counts().sort_index())}")
    print(f"6 分类标签分布: {dict(pd.Series(data['emotions_6class']).value_counts().sort_index())}")

    subject_all_results = []

    for model_name, model_config in tqdm(MODELS.items(), desc="模型"):
        model_results_by_task = {cfg['task_key']: [] for cfg in TASK_CONFIGS}

        for roi_id in tqdm(roi_ids, desc="ROI", leave=False):
            roi_name = roi_names.get(roi_id, f"ROI_{roi_id}")
            X = extract_roi_features(fMRI_data, atlas_data, roi_id)
            if X is None or X.shape[0] < 10: continue
            X_orig = X

            for task_cfg in TASK_CONFIGS:
                task_name = task_cfg['task_name']
                task_key = task_cfg['task_key']
                y_task_orig = data[task_cfg['label_key']]
                chance_level = task_cfg['chance_level']

                valid_mask = y_task_orig >= 0
                X_task = X_orig[valid_mask]
                y_task = y_task_orig[valid_mask]

                # 计算全局（所有试次）的 ANOVA F 值，选出前 k=800 个体素
                selector_global = SelectKBest(score_func=f_classif, k=800)
                selector_global.fit(X_task, y_task)
                global_support = selector_global.get_support(indices=True)  # 被选体素的列索引

                save_dir = os.path.join(output_dir, 'selected_voxel_indices')
                os.makedirs(save_dir, exist_ok=True)
                np.save(os.path.join(save_dir, f'{subject_id}_{model_name}_{task_key}_roi{roi_id}_indices.npy'), global_support)

                metrics_task, _, _, perm_uars = train_and_evaluate_with_permutation(
                    X_task, y_task, clone(model_config['model']),
                    n_splits=10, n_perms=3000,
                )
                true_uar = metrics_task['UAR']
                perm_uar_mean = np.mean(perm_uars)
                p_value = (np.sum(perm_uars >= true_uar) + 1) / (len(perm_uars) + 1)

                print(f"[{model_name}] {task_name} | ROI {roi_id} ({roi_name}): "
                      f"True UAR={true_uar:.4f}, Perm UAR mean={perm_uar_mean:.4f} "
                      f"(expected {chance_level:.4f}), p={p_value:.4f}")

                res_dict = {
                    'Subject': subject_id,
                    'Model': model_name,
                    'ROI_ID': roi_id,
                    'ROI_Name': roi_name,
                    'Task': task_name,
                    'ACC': metrics_task['ACC'],
                    'F1': metrics_task['F1'],
                    'WAR': metrics_task['WAR'],
                    'UAR': metrics_task['UAR'],
                    'WF1': metrics_task['WF1'],
                    'UF1': metrics_task['UF1'],
                    'CV_UAR_Mean': metrics_task['CV_UAR_Mean'],
                    'CV_UAR_Std': metrics_task['CV_UAR_Std'],
                    'N_Voxels_Original': X.shape[1],
                    'N_Voxels_Selected': 'Dynamic_per_Fold',
                    'N_Trials': X_task.shape[0],
                    'Chance_Level': chance_level,
                    'Perm_P': p_value,
                    'Perm_UAR_Mean': perm_uar_mean
                }

                model_results_by_task[task_key].append(res_dict)
                ALL_PERM_DATA.append({
                    'subject': subject_id,
                    'task': task_name,
                    'model': model_name,
                    'roi_id': roi_id,
                    'roi_name': roi_name,
                    'perm_uars': perm_uars,
                    'true_uar': true_uar,
                    'perm_uar_mean': perm_uar_mean,
                    'p_value': p_value,
                    'metrics': {k: metrics_task[k] for k in [
                        'ACC', 'F1', 'WAR', 'UAR', 'WF1', 'UF1',
                        'CV_UAR_Mean', 'CV_UAR_Std'
                    ]}
                })

            del X
            gc.collect()

        # 保存模型结果
        for task_cfg in TASK_CONFIGS:
            task_name = task_cfg['task_name']
            task_key = task_cfg['task_key']
            model_results = model_results_by_task[task_key]

            df_task = save_model_results(model_results, subject_id, model_name, task_key, output_dir)
            if df_task is not None:
                subject_all_results.append(df_task)

            ALL_RESULTS.extend(model_results)
            if task_key == '3class':
                ALL_RESULTS_3CLASS.extend(model_results)
            elif task_key == '6class':
                ALL_RESULTS_6CLASS.extend(model_results)

            if model_results:
                best = pd.DataFrame(model_results).loc[pd.DataFrame(model_results)['UAR'].idxmax()]
                print(f"{model_name} - {task_name} 最佳 ROI: {best['ROI_Name']} (UAR={best['UAR']:.4f})")

    return subject_all_results


def save_all_results_to_npz(output_path):
    """将所有结果（组指标、置换零分布等）保存为单个 NPZ 文件"""
    df = pd.DataFrame(ALL_RESULTS)
    dtypes = [(col, df[col].dtype if df[col].dtype != object else 'U256') for col in df.columns]
    group_arr = np.zeros(len(df), dtype=dtypes)
    for col in df.columns:
        group_arr[col] = df[col].values.astype(dtypes[df.columns.get_loc(col)][1])

    npz_dict = {'group_results': group_arr}
    for item in ALL_PERM_DATA:
        task_key = str(item['task']).lower()
        key = f"{item['subject']}_{item['model']}_{task_key}_roi{item['roi_id']}_perm"
        npz_dict[key] = item['perm_uars']
        meta = np.array(
            [(item['subject'], item['task'], item['model'], item['roi_id'], item['roi_name'],
              item['true_uar'], item['perm_uar_mean'], item['p_value'],
              item['metrics']['ACC'], item['metrics']['F1'], item['metrics']['WAR'],
              item['metrics']['UAR'], item['metrics']['WF1'], item['metrics']['UF1'],
              item['metrics']['CV_UAR_Mean'], item['metrics']['CV_UAR_Std'])],
            dtype=[
                ('subject', 'U10'), ('task', 'U10'), ('model', 'U10'), ('roi_id', int), ('roi_name', 'U80'),
                ('true_uar', float), ('perm_uar_mean', float), ('p_value', float),
                ('ACC', float), ('F1', float), ('WAR', float), ('UAR', float),
                ('WF1', float), ('UF1', float), ('CV_UAR_Mean', float), ('CV_UAR_Std', float)
            ]
        )
        npz_dict[f"{item['subject']}_{item['model']}_{task_key}_roi{item['roi_id']}_meta"] = meta

    np.savez_compressed(output_path, **npz_dict)
    print(f"所有结果已保存至 NPZ: {output_path}")

# ==================== 主程序 ====================
if __name__ == "__main__":
    # test_data_leak_with_synthetic_data()
    print(f"\n{'=' * 80}")
    print("fMRI 情绪解码分析 (SelectFpr Feature Selection)")
    print(f"开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}")

    subjects = ['sub01', 'sub02', 'sub03', 'sub04', 'sub05', 'sub06', 'sub07', 'sub08']
    for subject_id in subjects:
        print(f"\n{'#' * 80}\n# 处理被试：{subject_id}\n{'#' * 80}")
        data = load_subject_data(subject_id, FMRI_DIR, LABELS_FILE, HCPex_22_ATLAS)

        if data is None: continue
        run_decoding(subject_id, data, OUTPUT_DIR)
        print(f"\n[内存清理] 释放被试 {subject_id} 内存...")
        del data;
        gc.collect()

    # 保存组水平汇总
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if ALL_RESULTS:
        all_df = pd.DataFrame(ALL_RESULTS)
        all_df.to_csv(os.path.join(OUTPUT_DIR, f'GROUP_ALL_3class_6class_{timestamp}.csv'),
                      index=False, encoding='utf-8-sig')

        if ALL_RESULTS_3CLASS:
            pd.DataFrame(ALL_RESULTS_3CLASS).to_csv(
                os.path.join(OUTPUT_DIR, f'GROUP_ALL_3class.csv'),
                index=False, encoding='utf-8-sig'
            )
        if ALL_RESULTS_6CLASS:
            pd.DataFrame(ALL_RESULTS_6CLASS).to_csv(
                os.path.join(OUTPUT_DIR, f'GROUP_ALL_6class.csv'),
                index=False, encoding='utf-8-sig'
            )

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    npz_path = os.path.join(OUTPUT_DIR, f'ALL_DECODING_RESULTS.npz')
    save_all_results_to_npz(npz_path)

    print_final_summary()

    print(f"\n{'=' * 80}\n分析完成！结束时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'=' * 80}")