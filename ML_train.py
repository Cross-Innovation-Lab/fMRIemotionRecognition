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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime
from tqdm import tqdm
from sklearn.base import clone
from utlies import *
warnings.filterwarnings('ignore')
sys.stdout = Logger("ML_Train_no_select_.log")
# ==================== 配置路径 ====================
FMRI_DIR = r"..\data\extract_fmri"
LABELS_FILE = r"..\data\fmri\emotion"
HCPex_22_ATLAS = r"..\HCPex\HCPex_v1.0\HCPex_22Systems.nii"
HCPex_LABELS = r"..\data\HCPex\HCPex_v1.0\HCPex_22Systems_Labels.csv"
OUTPUT_DIR = r"..\data\decoding"

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 情绪标签映射 ====================
EMOTION_6CLASS = {
    'happy': 0, 'sad': 1, 'angry': 2, 'fear': 3, 'neutral': 4, 'surprise': 5
}

EMOTION_3CLASS = {
    'happy': 0, 'surprise': 0, 'sad': 1, 'angry': 1, 'fear': 1, 'neutral': 2
}

EMOTION_3CLASS_NAMES = {0: 'Positive', 1: 'Negative', 2: 'Neutral'}
EMOTION_6CLASS_NAMES = {0: 'Happy', 1: 'Sad', 2: 'Angry', 3: 'Fear', 4: 'Neutral', 5: 'Surprise'}

# ==================== 机器学习模型配置 ====================
MODELS = {
    'SVM': {
        'model': SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=42),
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
    """计算 6 种评估指标"""
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    class_weights = np.sum(cm, axis=1) / np.sum(cm)
    class_acc = np.diag(cm) / (np.sum(cm, axis=1) + 1e-10)
    war = np.sum(class_weights * class_acc)
    uar = np.mean(class_acc)

    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    uf1 = f1_macro

    return {
        'ACC': acc, 'F1': f1_macro, 'WAR': war, 'UAR': uar,
        'WF1': f1_weighted, 'UF1': uf1
    }


# ==================== 加载数据 ====================
def load_subject_data(subject_id, fMRI_dir, labels_file, atlas_path):
    """加载单个被试的 fMRI 数据和标签"""
    print(f"\n{'=' * 60}")
    print(f"加载被试 {subject_id} 数据")
    print(f"{'=' * 60}")

    # 1. 加载 fMRI 数据
    fMRI_path = os.path.join(fMRI_dir, f'{subject_id}_emotion_trials.nii.gz')
    if not os.path.exists(fMRI_path):
        print(f"错误：fMRI 文件不存在 {fMRI_path}")
        return None

    fMRI_img = nib.load(fMRI_path)
    fMRI_data = fMRI_img.get_fdata()
    print(f"fMRI 数据形状：{fMRI_data.shape}")

    # 2. 加载标签
    labels_path = os.path.join(labels_file, f'{subject_id}_output.xlsx')
    if not os.path.exists(labels_path):
        print(f"错误：标签文件不存在 {labels_path}")
        return None

    labels_df = pd.read_excel(labels_path)
    required_cols = ['orderid', 'predicted_emotion']
    for col in required_cols:
        if col not in labels_df.columns:
            print(f"错误：标签文件中不存在列 {col}")
            return None

    subject_trials = labels_df.copy()
    if len(subject_trials) == 0:
        print(f"错误：被试 {subject_id} 没有有效的 trial")
        return None

    # 3. 获取 orderid 并排序
    orderids = subject_trials['orderid'].astype(int).values
    sort_idx = np.argsort(orderids)
    orderids = orderids[sort_idx]

    # 4. 获取情绪标签（转换为整数）
    emotions_6class_raw = subject_trials['predicted_emotion'].values[sort_idx]
    emotions_6class = np.array([EMOTION_6CLASS.get(str(e).lower().strip(), -1) for e in emotions_6class_raw])

    if -1 in emotions_6class:
        print(f"警告：发现未知情绪标签，将替换为中性 (4)")
        emotions_6class[emotions_6class == -1] = 4

    emotions_3class = np.array([EMOTION_3CLASS.get(str(e).lower().strip(), 2) for e in emotions_6class_raw])

    print(f"有效 trial 数：{len(orderids)}")
    print(f"fMRI 第 4 维大小：{fMRI_data.shape[3]}")

    # 5. 检查维度匹配
    if len(orderids) != fMRI_data.shape[3]:
        print(f"警告：trial 数 ({len(orderids)}) 与 fMRI 第 4 维 ({fMRI_data.shape[3]}) 不匹配！")
        min_len = min(len(orderids), fMRI_data.shape[3])
        orderids = orderids[:min_len]
        emotions_6class = emotions_6class[:min_len]
        emotions_3class = emotions_3class[:min_len]
        fMRI_data = fMRI_data[:, :, :, :min_len]

    # 6. 加载图谱
    atlas_img = nib.load(atlas_path)
    atlas_data = atlas_img.get_fdata()

    # 7. 重采样图谱
    from nilearn.image import resample_to_img
    if atlas_data.shape != fMRI_data.shape[:3]:
        print(f"重采样图谱从 {atlas_data.shape} 到 {fMRI_data.shape[:3]}")
        atlas_img = resample_to_img(atlas_img, fMRI_img, interpolation='nearest')
        atlas_data = atlas_img.get_fdata()

    return {
        'fMRI_data': fMRI_data, 'fMRI_img': fMRI_img,
        'atlas_data': atlas_data, 'atlas_img': atlas_img,
        'orderids': orderids, 'emotions_6class': emotions_6class,
        'emotions_3class': np.array(emotions_3class), 'n_trials': len(orderids)
    }


# ==================== 提取 ROI 特征 ====================
def extract_roi_features(fMRI_data, atlas_data, roi_id, mask_mode='extract'):
    """提取单个 ROI 内的体素信号"""
    roi_mask = (atlas_data == roi_id)

    if np.sum(roi_mask) == 0:
        print(f"警告：ROI {roi_id} 没有体素")
        return None

    n_trials = fMRI_data.shape[3]

    if mask_mode == 'extract':
        n_voxels = np.sum(roi_mask)
        roi_features = fMRI_data[roi_mask, :].reshape(n_voxels, n_trials).T
        print(f"ROI {roi_id}: {n_voxels} 个体素，特征形状 {roi_features.shape}")
    else:
        masked_data = np.zeros_like(fMRI_data)
        masked_data[roi_mask, :] = fMRI_data[roi_mask, :]
        n_voxels_total = fMRI_data.shape[0] * fMRI_data.shape[1] * fMRI_data.shape[2]
        roi_features = masked_data.reshape(n_voxels_total, n_trials).T
        n_voxels_roi = np.sum(roi_mask)
        print(f"ROI {roi_id}: {n_voxels_roi} 个有效体素，特征形状 {roi_features.shape}")

    return roi_features


# ==================== 交叉验证训练 ====================
def train_and_evaluate(X, y, model, n_splits=5, task_name='', subject_id='', model_name='', roi_id=''):
    """使用交叉验证训练和评估模型"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_y_true = []
    all_y_pred = []
    cv_scores = []

    # 【修改】检测 NaN 并给出警告
    nan_count = np.sum(np.isnan(X))
    inf_count = np.sum(np.isinf(X))
    if nan_count > 0:
        print(f"警告：被试{subject_id} | 模型{model_name} | ROI{roi_id} | 数据中存在 {nan_count} 个 NaN 值")
    if inf_count > 0:
        print(f"警告：被试{subject_id} | 模型{model_name} | ROI{roi_id} | 数据中存在 {inf_count} 个 Inf 值")

    # 清理 NaN/Inf 为 0（避免训练失败）
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  进行 {n_splits} 折交叉验证...")
    y = y.astype(int)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 再次检查
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        fold_acc = accuracy_score(y_test, y_pred)
        cv_scores.append(fold_acc)
        print(f"Fold {fold + 1}: ACC = {fold_acc:.4f}")

        del X_train, X_test, X_train_scaled, X_test_scaled
        gc.collect()

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    metrics = calculate_metrics(all_y_true, all_y_pred, len(np.unique(y)))
    metrics['CV_Mean'] = np.mean(cv_scores)
    metrics['CV_Std'] = np.std(cv_scores)

    return metrics, all_y_true, all_y_pred


# ==================== 保存单个模型结果 ====================
def save_model_results(results_list, subject_id, model_name, task_type, output_dir):
    """保存单个模型的结果到 CSV"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if len(results_list) == 0:
        print(f"警告：{task_type} 任务无结果可保存")
        return None

    df = pd.DataFrame(results_list)
    filename = f'{subject_id}_{model_name}_{task_type}_results_{timestamp}.csv'
    output_path = os.path.join(output_dir, filename)

    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"{task_type} 结果已保存：{output_path}")

    return df


# ==================== 主解码流程 ====================
def run_decoding(subject_id, data, output_dir):
    """执行完整的解码流程"""
    print(f"\n{'=' * 80}")
    print(f"开始被试 {subject_id} 的情绪解码分析")
    print(f"{'=' * 80}")

    # 加载 ROI 标签
    labels_df = pd.read_csv(HCPex_LABELS)
    roi_ids = labels_df[labels_df['ID'] > 0]['ID'].astype(int).tolist()
    roi_names = labels_df[labels_df['ID'] > 0].set_index('ID')['Name'].to_dict()
    print(f"\n共 {len(roi_ids)} 个 ROI")

    # 获取数据
    fMRI_data = data['fMRI_data']
    atlas_data = data['atlas_data']
    y_3class = data['emotions_3class']
    y_6class = data['emotions_6class']

    # 打印标签分布
    print(f"\n{'=' * 60}")
    print("3 分类标签分布:")
    for label, count in pd.Series(y_3class).value_counts().sort_index().items():
        print(f"  {EMOTION_3CLASS_NAMES[label]}: {count} ({count / len(y_3class) * 100:.1f}%)")

    print(f"\n6 分类标签分布:")
    for label, count in pd.Series(y_6class).value_counts().sort_index().items():
        print(f"  {EMOTION_6CLASS_NAMES.get(label, label)}: {count} ({count / len(y_6class) * 100:.1f}%)")

    # 存储当前被试的所有结果
    subject_all_results_3class = []
    subject_all_results_6class = []

    # ==================== 循环：模型 × ROI × 任务 ====================
    for model_name, model_config in tqdm(MODELS.items(), desc="模型"):
        print(f"\n{'=' * 80}")
        print(f"模型：{model_name}")
        print(f"{'=' * 80}")

        # 当前模型的结果
        model_results_3class = []
        model_results_6class = []

        for roi_id in tqdm(roi_ids, desc="ROI", leave=False):
            roi_name = roi_names.get(roi_id, f"ROI_{roi_id}")

            # 提取 ROI 特征
            X = extract_roi_features(fMRI_data, atlas_data, roi_id)

            if X is None or X.shape[0] < 10:
                print(f"跳过 ROI {roi_id} (体素数不足)")
                continue

            # ----- 3 分类任务 -----
            unique_classes_3 = np.unique(y_3class)
            if len(unique_classes_3) < 3:
                print(f"警告：3 分类任务中类别数不足 ({len(unique_classes_3)}), 跳过")
                metrics_3class = {k: np.nan for k in ['ACC', 'F1', 'WAR', 'UAR', 'WF1', 'UF1', 'CV_Mean', 'CV_Std']}
            else:
                print(f"\n  ROI {roi_id} ({roi_name}) - 3 分类任务")
                model_3class = clone(model_config['model'])
                metrics_3class, _, _ = train_and_evaluate(
                    X, y_3class, model_3class, n_splits=5, task_name='3class',
                    subject_id=subject_id, model_name=model_name, roi_id=roi_id
                )

            # ----- 6 分类任务 -----
            unique_classes_6 = np.unique(y_6class)
            if len(unique_classes_6) < 6:
                print(f"警告：6 分类任务中类别数不足 ({len(unique_classes_6)}), 跳过")
                metrics_6class = {k: np.nan for k in ['ACC', 'F1', 'WAR', 'UAR', 'WF1', 'UF1', 'CV_Mean', 'CV_Std']}
            else:
                print(f"\n  ROI {roi_id} ({roi_name}) - 6 分类任务")
                model_6class = clone(model_config['model'])
                metrics_6class, _, _ = train_and_evaluate(
                    X, y_6class, model_6class, n_splits=5, task_name='6class',
                    subject_id=subject_id, model_name=model_name, roi_id=roi_id
                )

            # 保存结果到列表
            model_results_3class.append({
                'Subject': subject_id, 'Model': model_name, 'ROI_ID': roi_id,
                'ROI_Name': roi_name, 'Task': '3Class',
                'ACC': metrics_3class['ACC'], 'F1': metrics_3class['F1'],
                'WAR': metrics_3class['WAR'], 'UAR': metrics_3class['UAR'],
                'WF1': metrics_3class['WF1'], 'UF1': metrics_3class['UF1'],
                'CV_Mean': metrics_3class['CV_Mean'], 'CV_Std': metrics_3class['CV_Std'],
                'N_Voxels': X.shape[1], 'N_Trials': X.shape[0]
            })

            model_results_6class.append({
                'Subject': subject_id, 'Model': model_name, 'ROI_ID': roi_id,
                'ROI_Name': roi_name, 'Task': '6Class',
                'ACC': metrics_6class['ACC'], 'F1': metrics_6class['F1'],
                'WAR': metrics_6class['WAR'], 'UAR': metrics_6class['UAR'],
                'WF1': metrics_6class['WF1'], 'UF1': metrics_6class['UF1'],
                'CV_Mean': metrics_6class['CV_Mean'], 'CV_Std': metrics_6class['CV_Std'],
                'N_Voxels': X.shape[1], 'N_Trials': X.shape[0]
            })

            del X
            gc.collect()

        # 【修改】每个模型训练完后立即保存结果
        print(f"\n{'=' * 80}")
        print(f"保存 {model_name} 模型结果...")
        print(f"{'=' * 80}")

        df_3class = save_model_results(model_results_3class, subject_id, model_name, '3class', output_dir)
        df_6class = save_model_results(model_results_6class, subject_id, model_name, '6class', output_dir)

        # 添加到被试总结果
        if df_3class is not None:
            subject_all_results_3class.append(df_3class)
        if df_6class is not None:
            subject_all_results_6class.append(df_6class)

        # 添加到全局结果
        ALL_RESULTS_3CLASS.extend(model_results_3class)
        ALL_RESULTS_6CLASS.extend(model_results_6class)

        # 打印当前模型汇总
        if len(model_results_3class) > 0:
            df_temp = pd.DataFrame(model_results_3class)
            best_row = df_temp.loc[df_temp['ACC'].idxmax()]
            print(f"{model_name} - 3 分类最佳 ROI: {best_row['ROI_Name']} (ACC = {best_row['ACC']:.4f})")

        if len(model_results_6class) > 0:
            df_temp = pd.DataFrame(model_results_6class)
            best_row = df_temp.loc[df_temp['ACC'].idxmax()]
            print(f"{model_name} - 6 分类最佳 ROI: {best_row['ROI_Name']} (ACC = {best_row['ACC']:.4f})")

    # 保存被试汇总结果
    print(f"\n{'=' * 80}")
    print(f"保存被试 {subject_id} 汇总结果...")
    print(f"{'=' * 80}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if len(subject_all_results_3class) > 0:
        subject_3class_df = pd.concat(subject_all_results_3class, ignore_index=True)
        output_3class = os.path.join(output_dir, f'{subject_id}_all_3class_{timestamp}.csv')
        subject_3class_df.to_csv(output_3class, index=False, encoding='utf-8-sig')
        print(f"被试 3 分类汇总：{output_3class}")

    if len(subject_all_results_6class) > 0:
        subject_6class_df = pd.concat(subject_all_results_6class, ignore_index=True)
        output_6class = os.path.join(output_dir, f'{subject_id}_all_6class_{timestamp}.csv')
        subject_6class_df.to_csv(output_6class, index=False, encoding='utf-8-sig')
        print(f"被试 6 分类汇总：{output_6class}")

    return subject_all_results_3class, subject_all_results_6class


# ==================== 打印汇总统计 ====================
def print_final_summary():
    """打印最终汇总统计"""
    print(f"\n{'=' * 80}")
    print("最终解码结果汇总")
    print(f"{'=' * 80}")

    if len(ALL_RESULTS_3CLASS) > 0:
        df_3class = pd.DataFrame(ALL_RESULTS_3CLASS)

        print(f"\n{'=' * 60}")
        print("3 分类任务汇总")
        print(f"{'=' * 60}")

        for model in MODELS.keys():
            model_data = df_3class[df_3class['Model'] == model]
            if len(model_data) > 0:
                best_row = model_data.loc[model_data['ACC'].idxmax()]
                print(f"\n{model}:")
                print(f"  最佳 ROI: {best_row['ROI_Name']} (ACC = {best_row['ACC']:.4f})")
                print(f"  平均 ACC: {model_data['ACC'].mean():.4f} ± {model_data['ACC'].std():.4f}")

        print(f"\n{'=' * 60}")
        print("所有指标汇总 (3 分类)")
        print(f"{'=' * 60}")
        metrics = ['ACC', 'F1', 'WAR', 'UAR', 'WF1', 'UF1']
        for metric in metrics:
            if metric in df_3class.columns:
                mean_val = df_3class[metric].mean()
                std_val = df_3class[metric].std()
                print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")

    if len(ALL_RESULTS_6CLASS) > 0:
        df_6class = pd.DataFrame(ALL_RESULTS_6CLASS)

        print(f"\n{'=' * 60}")
        print("6 分类任务汇总")
        print(f"{'=' * 60}")

        for model in MODELS.keys():
            model_data = df_6class[df_6class['Model'] == model]
            if len(model_data) > 0:
                best_row = model_data.loc[model_data['ACC'].idxmax()]
                print(f"\n{model}:")
                print(f"  最佳 ROI: {best_row['ROI_Name']} (ACC = {best_row['ACC']:.4f})")
                print(f"  平均 ACC: {model_data['ACC'].mean():.4f} ± {model_data['ACC'].std():.4f}")


# ==================== 主程序 ====================
if __name__ == "__main__":
    print(f"\n{'=' * 80}")
    print("fMRI 情绪解码分析")
    print(f"开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}")

    # 配置被试列表
    subjects = ['sub06']

    for subject_id in subjects:
        print(f"\n{'#' * 80}")
        print(f"# 处理被试：{subject_id}")
        print(f"{'#' * 80}")

        # 加载数据
        data = load_subject_data(subject_id, FMRI_DIR, LABELS_FILE, HCPex_22_ATLAS)

        if data is None:
            print(f"跳过被试 {subject_id}")
            continue

        # 执行解码（每个模型完成后自动保存）
        run_decoding(subject_id, data, OUTPUT_DIR)

        print(f"\n[内存清理] 正在释放被试 {subject_id} 的内存...")
        del data  # 删除包含 fMRI_data 和 atlas_data 的字典
        gc.collect()  # 强制垃圾回收

    # ==================== 保存所有被试的汇总 ====================
    print(f"\n{'=' * 80}")
    print("保存组水平汇总结果...")
    print(f"{'=' * 80}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if len(ALL_RESULTS_3CLASS) > 0:
        all_3class_df = pd.DataFrame(ALL_RESULTS_3CLASS)
        output_path = os.path.join(OUTPUT_DIR, f'GROUP_ALL_3class_results_{timestamp}.csv')
        all_3class_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"组水平 3 分类汇总：{output_path}")

    if len(ALL_RESULTS_6CLASS) > 0:
        all_6class_df = pd.DataFrame(ALL_RESULTS_6CLASS)
        output_path = os.path.join(OUTPUT_DIR, f'GROUP_ALL_6class_results_{timestamp}.csv')
        all_6class_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"组水平 6 分类汇总：{output_path}")

    # 打印最终汇总
    print_final_summary()

    print(f"\n{'=' * 80}")
    print(f"分析完成！结束时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}")