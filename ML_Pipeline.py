
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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectFdr, f_classif, SelectFpr  # [修改] 引入特征选择
from sklearn.base import clone
from datetime import datetime
from tqdm import tqdm
from joblib import Parallel, delayed

# 假设 utlies.py 中有 Logger 定义
try:
    from utlies import Logger
except ImportError:
    class Logger:
        def __init__(self, *args, **kwargs): pass

        def write(self, *args): pass

        def flush(self): pass

warnings.filterwarnings('ignore')
sys.stdout = Logger("ML_Train_SelectFpr.log")

# ==================== 配置路径 ====================
FMRI_DIR = r"..\data\extract_fmri"
LABELS_FILE = r"..\data\fmri\emotion"
HCPex_22_ATLAS = r"..\data\HCPex\HCPex_v1.0\HCPex_22Systems.nii"
HCPex_LABELS = r"..\data\HCPex\HCPex_v1.0\HCPex_22Systems_Labels.csv"
OUTPUT_DIR = r"..\data\decoding_selection"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 情绪标签映射 ====================
EMOTION_6CLASS = {'happy': 0, 'sad': 1, 'angry': 2, 'fear': 3, 'neutral': 4, 'surprise': 5}
EMOTION_3CLASS = {'happy': 0, 'surprise': 0, 'sad': 1, 'angry': 1, 'fear': 1, 'neutral': 2}
EMOTION_3CLASS_NAMES = {0: 'Positive', 1: 'Negative', 2: 'Neutral'}
EMOTION_6CLASS_NAMES = {0: 'Happy', 1: 'Sad', 2: 'Angry', 3: 'Fear', 4: 'Neutral', 5: 'Surprise'}

MODELS = {
    'SVM': {
        'model': SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=42, probability=True),
        'param_name': 'SVC'
    },
    'DT': {
        'model': DecisionTreeClassifier(max_depth=10, min_samples_split=5, class_weight='balanced', random_state=42),
        'param_name': 'DecisionTreeClassifier'
    },
    'RF': {
        'model': RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5,
                                        class_weight='balanced', random_state=42),
        'param_name': 'RandomForestClassifier'
    },
    'MLP': {
        'model': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, early_stopping=True,
                               random_state=42, verbose=0, validation_fraction=0.1),
        'param_name': 'MLPClassifier'
    }
}

# ==================== 全局结果存储 ====================
ALL_RESULTS_3CLASS = []
ALL_RESULTS_6CLASS = []


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

    # [修改] 计算 AUC
    # try:
    #     if n_classes == 2:
    #         auc = roc_auc_score(y_true, y_prob[:, 1])
    #     else:
    #         auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    # except Exception as e:
    #     print(f"AUC 计算失败 ({e})，设为 NaN")
    #     auc = np.nan

    return {
        'ACC': acc, 'F1': f1_macro, 'WAR': war, 'UAR': uar,
        'WF1': f1_weighted, 'UF1': f1_macro
    }


def run_strict_permutation(X, y_true, model, metric_key='ACC',
                           n_perms=1000, n_splits=5, random_state=42):
    """
    严格置换检验：固定X，仅置换y，完整重跑CV管线（含SelectFpr）
    返回：真实值、零分布均值、零分布标准差、p值、置换分布数组
    """
    # 1. 真实标签性能
    true_res, _, _ = train_and_evaluate(X, y_true, clone(model), n_splits=n_splits)
    true_val = true_res[metric_key]

    # 2. 置换检验核心循环（每次独立打乱y，完整走CV+SelectFpr+模型）
    def _permute_once(seed):
        rng = np.random.RandomState(seed)
        y_perm = rng.permutation(y_true)  # 打乱标签，保持类别边缘分布不变
        perm_res, _, _ = train_and_evaluate(X, y_perm, clone(model), n_splits=n_splits)
        return perm_res[metric_key]

    # 并行执行（避免全局随机种子冲突）
    null_vals = Parallel(n_jobs=-1, backend='loky')(
        delayed(_permute_once)(random_state + i) for i in range(n_perms)
    )
    null_vals = np.array(null_vals)

    # 3. 计算精确 p 值 (含连续性校正，符合统计学标准)
    p_val = (np.sum(null_vals >= true_val) + 1) / (n_perms + 1)

    return {
        'True_Value': true_val,
        'Null_Mean': np.mean(null_vals),
        'Null_Std': np.std(null_vals),
        'P_Value': p_val,
        'Null_Distribution': null_vals,  # 便于后续绘图/调试
        'N_Permutations': n_perms
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
    fMRI_data = fMRI_img.get_fdata()

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
        print(f"警告：发现未知情绪标签，将替换为中性 (4)")
        emotions_6class[emotions_6class == -1] = 4
    emotions_3class = np.array([EMOTION_3CLASS.get(str(e).lower().strip(), 2) for e in emotions_6class_raw])

    if len(orderids) != fMRI_data.shape[3]:
        min_len = min(len(orderids), fMRI_data.shape[3])
        orderids, emotions_6class, emotions_3class = orderids[:min_len], emotions_6class[:min_len], emotions_3class[
            :min_len]
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
        'emotions_6class': emotions_6class,
        'emotions_3class': np.array(emotions_3class),
        'n_trials': len(orderids)
    }


# ==================== 提取 ROI 特征 ====================
def extract_roi_features(fMRI_data, atlas_data, roi_id, mask_mode='extract'):
    """提取单个 ROI 内的体素信号 (保持不变)"""
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


# ==================== 按图片ID平均试次特征 ====================
def average_features_by_cocoid(X, cocoids, y):
    unique_coco = np.unique(cocoids)
    X_avg = np.zeros((len(unique_coco), X.shape[1]))
    y_avg = np.zeros(len(unique_coco), dtype=y.dtype)

    for i, coco in enumerate(unique_coco):
        idx = np.where(cocoids == coco)[0]
        X_avg[i, :] = np.mean(X[idx, :], axis=0)
        # 同一图片的标签必然相同，取第一个即可
        y_avg[i] = y[idx[0]]

    return X_avg, y_avg

# ==================== 交叉验证训练 ====================
def train_and_evaluate(X, y, model, n_splits=5, task_name='', subject_id='', model_name='', roi_id=''):
    """使用交叉验证训练和评估模型 (已修改：加入 SelectFpr 特征选择 & AUC 概率收集)"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_y_true, all_y_pred, cv_scores = [], [], []

    # NaN/Inf 检测与清理
    nan_count = np.sum(np.isnan(X))
    inf_count = np.sum(np.isinf(X))
    if nan_count > 0: print(
        f"警告：被试{subject_id} | 模型{model_name} | ROI{roi_id} | 数据中存在 {nan_count} 个 NaN 值")
    if inf_count > 0: print(
        f"警告：被试{subject_id} | 模型{model_name} | ROI{roi_id} | 数据中存在 {inf_count} 个 Inf 值")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = y.astype(int)
    n_classes = len(np.unique(y))

    print(f"  进行 {n_splits} 折交叉验证 (内含 SelectFpr 特征选择)...")

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        # 1. 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 2. [修改] 特征选择 (SelectFpr) - 仅在训练集上 fit，防止数据泄露
        selector = SelectFpr(score_func=f_classif, alpha=0.05)
        try:
            selector.fit(X_train_scaled, y_train)
            X_train_sel = selector.transform(X_train_scaled)
            X_test_sel = selector.transform(X_test_scaled)
        except Exception as e:
            print(f"特征选择失败 ({e})，使用原始特征")
            X_train_sel, X_test_sel = X_train_scaled, X_test_scaled

        if X_train_sel.shape[1] == 0:
            print(f"特征选择后无特征剩余，跳过此折")
            continue

        # 3. 训练与预测
        model_fold = clone(model)
        model_fold.fit(X_train_sel, y_train)
        y_pred = model_fold.predict(X_test_sel)

        # 4. 获取预测概率 (用于 AUC)
        # try:
        #     y_prob = model_fold.predict_proba(X_test_sel)
        # except AttributeError:
        #     try:
        #         y_prob = model_fold.decision_function(X_test_sel)
        #         if y_prob.ndim == 1: y_prob = np.vstack([-y_prob, y_prob]).T
        #     except Exception:
        #         y_prob = np.zeros((len(y_test), n_classes))

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        # all_y_prob.append(y_prob)

        fold_acc = accuracy_score(y_test, y_pred)
        cv_scores.append(fold_acc)
        print(f"  Fold {fold + 1}: ACC = {fold_acc:.4f} | 保留特征数: {X_train_sel.shape[1]}")

        del X_train, X_test, X_train_scaled, X_test_scaled, X_train_sel, X_test_sel
        gc.collect()

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    # all_y_prob = np.vstack(all_y_prob) if len(all_y_prob) > 0 else np.array([])
    # all_y_prob = np.array(all_y_prob)

    if len(all_y_true) == 0:
        return {k: np.nan for k in
                ['ACC', 'F1', 'WAR', 'UAR', 'WF1', 'UF1', 'CV_Mean', 'CV_Std']}, all_y_true, all_y_pred

    metrics = calculate_metrics(all_y_true, all_y_pred, n_classes)
    metrics['CV_Mean'] = np.mean(cv_scores)
    metrics['CV_Std'] = np.std(cv_scores)
    return metrics, all_y_true, all_y_pred


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
    y_3class, y_6class = data['emotions_3class'], data['emotions_6class']

    print(f"\n3 分类标签分布: {dict(pd.Series(y_3class).value_counts().sort_index())}")
    print(f"6 分类标签分布: {dict(pd.Series(y_6class).value_counts().sort_index())}")

    subject_all_results_3class, subject_all_results_6class = [], []

    for model_name, model_config in tqdm(MODELS.items(), desc="模型"):
        model_results_3class, model_results_6class = [], []
        cocoids = data['cocoids']

        for roi_id in tqdm(roi_ids, desc="ROI", leave=False):
            roi_name = roi_names.get(roi_id, f"ROI_{roi_id}")
            X = extract_roi_features(fMRI_data, atlas_data, roi_id)
            if X is None or X.shape[0] < 10: continue
            X_avg_3, y_3class_avg = average_features_by_cocoid(X, cocoids, y_3class)
            X_avg_6, y_6class_avg = average_features_by_cocoid(X, cocoids, y_6class)

            # 3 分类（使用平均后的数据）
            unique_3 = np.unique(y_3class_avg)
            if len(unique_3) < 3:
                metrics_3class = {k: np.nan for k in
                                  ['ACC', 'F1', 'WAR', 'UAR', 'WF1', 'UF1', 'CV_Mean', 'CV_Std']}
            else:
                metrics_3class, _, _ = train_and_evaluate(
                    X_avg_3, y_3class_avg, clone(model_config['model']),
                    n_splits=5, subject_id=subject_id, model_name=model_name, roi_id=roi_id
                )

            # 6 分类（使用平均后的数据）
            unique_6 = np.unique(y_6class_avg)
            if len(unique_6) < 6:
                metrics_6class = {k: np.nan for k in
                                  ['ACC', 'F1', 'WAR', 'UAR', 'WF1', 'UF1', 'CV_Mean', 'CV_Std']}
            else:
                metrics_6class, _, _ = train_and_evaluate(
                    X_avg_6, y_6class_avg, clone(model_config['model']),
                    n_splits=5, subject_id=subject_id, model_name=model_name, roi_id=roi_id
                )

            # [修改] 结果字典增加 AUC
            for task, y, metrics in [('3Class', y_3class, metrics_3class), ('6Class', y_6class, metrics_6class)]:
                # 根据任务选择对应的平均后样本数
                if task == '3Class':
                    n_trials_avg = X_avg_3.shape[0]
                else:
                    n_trials_avg = X_avg_6.shape[0]

                res_dict = {
                    'Subject': subject_id, 'Model': model_name, 'ROI_ID': roi_id, 'ROI_Name': roi_name,
                    'Task': task, 'ACC': metrics['ACC'], 'F1': metrics['F1'], 'WAR': metrics['WAR'],
                    'UAR': metrics['UAR'], 'WF1': metrics['WF1'], 'UF1': metrics['UF1'],
                    'CV_Mean': metrics['CV_Mean'], 'CV_Std': metrics['CV_Std'],
                    'N_Voxels_Original': X.shape[1], 'N_Voxels_Selected': 'Dynamic_per_Fold',
                    'N_Trials_Original': X.shape[0],  # 原始试次数（可选）
                    'N_Trials': n_trials_avg  # 平均后实际样本数
                }
                if task == '3Class':
                    model_results_3class.append(res_dict)
                else:
                    model_results_6class.append(res_dict)

            del X;
            gc.collect()

        # 保存模型结果
        df_3class = save_model_results(model_results_3class, subject_id, model_name, '3class', output_dir)
        df_6class = save_model_results(model_results_6class, subject_id, model_name, '6class', output_dir)
        if df_3class is not None: subject_all_results_3class.append(df_3class)
        if df_6class is not None: subject_all_results_6class.append(df_6class)
        ALL_RESULTS_3CLASS.extend(model_results_3class)
        ALL_RESULTS_6CLASS.extend(model_results_6class)

        # 打印最佳 ROI
        if model_results_3class:
            best = pd.DataFrame(model_results_3class).loc[pd.DataFrame(model_results_3class)['ACC'].idxmax()]
            print(f"{model_name} - 3Class 最佳 ROI: {best['ROI_Name']} (ACC={best['ACC']:.4f})")
        if model_results_6class:
            best = pd.DataFrame(model_results_6class).loc[pd.DataFrame(model_results_6class)['ACC'].idxmax()]
            print(f"{model_name} - 6Class 最佳 ROI: {best['ROI_Name']} (ACC={best['ACC']:.4f})")

    return subject_all_results_3class, subject_all_results_6class


# ==================== 打印汇总统计 ====================
def print_final_summary():
    print(f"\n{'=' * 80}")
    print("最终解码结果汇总")
    print(f"{'=' * 80}")
    metrics_to_print = ['ACC', 'F1', 'WAR', 'UAR', 'WF1', 'UF1']

    for task_name, results_list in [("3 分类", ALL_RESULTS_3CLASS), ("6 分类", ALL_RESULTS_6CLASS)]:
        if not results_list: continue
        df = pd.DataFrame(results_list)
        print(f"\n{'=' * 60}\n{task_name} 任务汇总\n{'=' * 60}")
        for model in MODELS.keys():
            m_data = df[df['Model'] == model]
            if len(m_data) > 0:
                best = m_data.loc[m_data['ACC'].idxmax()]
                print(f"\n{model}: 最佳 ROI={best['ROI_Name']} (ACC={best['ACC']:.4f})")
                print(f"  平均 ACC: {m_data['ACC'].mean():.4f} ± {m_data['ACC'].std():.4f}")
        print(f"\n所有指标均值:")
        for m in metrics_to_print:
            if m in df.columns:
                print(f"  {m}: {df[m].mean():.4f} ± {df[m].std():.4f}")


# ==================== 主程序 ====================
if __name__ == "__main__":
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
    if ALL_RESULTS_3CLASS:
        pd.DataFrame(ALL_RESULTS_3CLASS).to_csv(os.path.join(OUTPUT_DIR, f'GROUP_ALL_3class_{timestamp}.csv'),
                                                index=False, encoding='utf-8-sig')
    if ALL_RESULTS_6CLASS:
        pd.DataFrame(ALL_RESULTS_6CLASS).to_csv(os.path.join(OUTPUT_DIR, f'GROUP_ALL_6class_{timestamp}.csv'),
                                                index=False, encoding='utf-8-sig')

    print_final_summary()
    print(f"\n{'=' * 80}\n分析完成！结束时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'=' * 80}")