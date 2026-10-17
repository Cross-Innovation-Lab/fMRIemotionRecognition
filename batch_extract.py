import gc
import os
import glob
import numpy as np
import pandas as pd
from nilearn import image
import nibabel as nib
from tqdm import tqdm
from datetime import datetime


def count_session_files(nii_dir):
    """
    自动检测被试文件夹中有多少个betas_session文件

    参数:
    -----------
    nii_dir : str
        nii 文件所在目录

    返回:
    -----------
    session_count : int
        session 文件数量
    session_files : list
        所有session文件路径列表
    """
    # 查找所有 betas_session*.nii.gz 文件
    pattern = os.path.join(nii_dir, 'betas_session*.nii.gz')
    session_files = sorted(glob.glob(pattern))

    # 提取session编号并排序
    session_nums = []
    for f in session_files:
        filename = os.path.basename(f)
        # 提取 session 编号，如 betas_session01.nii.gz → 1
        try:
            session_num = int(filename.split('session')[1].split('.')[0])
            session_nums.append(session_num)
        except:
            print(f"警告：无法解析文件名 {filename}")

    session_nums = sorted(session_nums)

    return len(session_files), session_files, session_nums


def extract_emotion_trials(excel_path, nii_dir, output_path, subject_id):
    """
    从 NSD 数据集中抽取带表情图像的 trial

    参数:
    -----------
    excel_path : str
        Excel 文件路径（包含 orderid 列）
    nii_dir : str
        nii 文件所在目录
    output_path : str
        输出 nii 文件路径
    subject_id : str
        被试ID（用于打印信息）
    """

    print(f"\n{'=' * 60}")
    print(f"开始处理 {subject_id}")
    print(f"{'=' * 60}")

    # 0. 检测session文件数量
    print(f"\n[步骤0] 检测 {subject_id} 的 session 文件...")
    session_count, session_files, session_nums = count_session_files(nii_dir)
    print(f"  ✓ 找到 {session_count} 个 session 文件")
    print(f"  ✓ Session 编号: {session_nums}")

    if session_count == 0:
        print(f"  ✗ 错误：{subject_id} 没有找到任何 session 文件！")
        return None

    # 1. 读取 Excel 文件，获取所有表情图像的 orderid
    print(f"\n[步骤1] 读取 Excel 文件...")
    df = pd.read_excel(excel_path)

    # 获取所有有效的 orderid（排除 NaN 和重复值）
    orderids_all = df['orderid'].dropna().unique().astype(int)
    orderids_all = np.sort(orderids_all)

    valid_orderids = []
    for oid in orderids_all:
        sess = oid // 750 + 1
        if sess in session_nums:
            valid_orderids.append(oid)
    orderids = np.array(valid_orderids)

    print(f"  ✓ 共找到 {len(orderids)} 个表情图像 trial")
    print(f"  ✓ orderid 范围：{orderids.min()} - {orderids.max()}")

    # 2. 初始化存储数组
    # 先加载一个 nii 文件获取空间维度信息
    sample_nii_path = session_files[0]
    sample_img = image.load_img(sample_nii_path)
    spatial_shape = sample_img.shape[:3]  # (182, 218, 182)

    print(f"\n[步骤2] 空间维度：{spatial_shape}")

    # 创建 4D 数组存储所有抽取的 trial
    num_trials = len(orderids)
    data_4d = np.zeros((*spatial_shape, num_trials), dtype=np.float32)

    current_idx = 0
    # 3. 按 session 分组处理，避免内存累积
    print(f"\n[步骤3] 按 session 分组处理...")

    # 按 session 分组 orderids
    session_groups = {}
    for orderid in orderids:
        session_num = orderid // 750 + 1
        if session_num not in session_groups:
            session_groups[session_num] = []
        session_groups[session_num].append(orderid)

    print(f"  ✓ 共涉及 {len(session_groups)} 个 session")

    # 检查是否有超出实际session范围的orderid
    max_session = max(session_nums)
    invalid_sessions = [s for s in session_groups.keys() if s > max_session]
    if invalid_sessions:
        print(f"  ⚠ 警告：发现超出实际session范围的orderid (session {invalid_sessions})")
        print(f"  ⚠ 这些trial将被跳过")

    for session_num in tqdm(sorted(session_groups.keys()), desc=f"处理 {subject_id} session"):
        # 检查session文件是否存在
        if session_num not in session_nums:
            print(f"  ⚠ 跳过 session{session_num:02d} (文件不存在)")
            # 跳过这些trial
            current_idx += len(session_groups[session_num])
            continue

        # 构建 nii 文件路径
        nii_filename = f'betas_session{session_num:02d}.nii.gz'
        nii_filepath = os.path.join(nii_dir, nii_filename)

        # 检查文件是否存在
        if not os.path.exists(nii_filepath):
            print(f"  ⚠ 警告：文件不存在 {nii_filepath}")
            current_idx += len(session_groups[session_num])
            continue

        # 加载当前 session 数据
        session_img = image.load_img(nii_filepath)
        session_data = session_img.get_fdata()  # 形状：(182, 218, 182, 750)

        # 处理该 session 的所有 trial
        for orderid in session_groups[session_num]:
            # 计算在该 session 中的位置（0-749）
            trial_idx_in_session = orderid % 750

            # 抽取该 trial 的数据
            trial_data = session_data[:, :, :, trial_idx_in_session]

            # 存储到 4D 数组
            data_4d[:, :, :, current_idx] = trial_data
            current_idx += 1

        # 立即释放当前 session 数据，避免内存累积
        del session_data
        del session_img
        gc.collect()  # 强制垃圾回收

    # 4. 创建新的 nii 文件
    print(f"\n[步骤4] 创建输出 nii 文件...")

    # 使用第一个 session 的 affine 矩阵
    new_img = nib.Nifti1Image(data_4d, sample_img.affine)

    # 设置一些元数据
    new_img.header.set_xyzt_units('mm')

    # 保存文件
    nib.save(new_img, output_path)

    print(f"\n{'=' * 60}")
    print(f"✅ {subject_id} 处理完成！")
    print(f"{'=' * 60}")
    print(f"  输出文件：{output_path}")
    print(f"  输出形状：{data_4d.shape}")
    print(f"  总 trial 数：{num_trials}")
    print(f"  Session 数量：{session_count}")
    print(f"{'=' * 60}\n")

    # 释放内存
    del data_4d
    del new_img
    gc.collect()

    return {
        'subject_id': subject_id,
        'session_count': session_count,
        'num_trials': num_trials,
        'output_path': output_path,
        'output_shape': data_4d.shape
    }


def verify_extraction(output_path, orderids, nii_dir, subject_id):
    """
    验证抽取结果是否正确
    """
    print(f"\n{'=' * 60}")
    print(f"验证 {subject_id} 抽取结果")
    print(f"{'=' * 60}")

    # 加载输出文件
    output_img = image.load_img(output_path)
    output_data = output_img.get_fdata()

    # 随机检查几个 trial
    check_indices = np.random.choice(len(orderids), min(5, len(orderids)), replace=False)

    all_match = True
    for idx in check_indices:
        orderid = orderids[idx]
        session_num = orderid // 750 + 1
        trial_idx = orderid % 750

        # 从原始文件读取
        nii_path = os.path.join(nii_dir, f'betas_session{session_num:02d}.nii.gz')
        if not os.path.exists(nii_path):
            print(f"  Trial {orderid}: Session 文件不存在，跳过验证")
            continue

        original_img = image.load_img(nii_path)
        original_data = original_img.get_fdata()
        original_trial = original_data[:, :, :, trial_idx]

        # 从输出文件读取
        extracted_trial = output_data[:, :, :, idx]

        # 比较
        is_match = np.allclose(original_trial, extracted_trial)
        mean_diff = np.mean(np.abs(original_trial - extracted_trial))

        if not is_match:
            all_match = False

        print(f"  Trial {orderid}: Session{session_num:02d}[{trial_idx}] - "
              f"匹配：{is_match}, 平均差异：{mean_diff:.2e}")

    if all_match:
        print(f"\n  ✅ {subject_id} 验证通过！")
    else:
        print(f"\n  ⚠ {subject_id} 验证发现不匹配！")

    print(f"{'=' * 60}\n")

    # 释放内存
    del output_data
    del output_img
    gc.collect()

    return all_match


def get_session_statistics(excel_path, nii_dir, subject_id):
    """
    统计每个 session 需要抽取多少个 trial
    """
    df = pd.read_excel(excel_path)
    orderids = df['orderid'].dropna().unique().astype(int)

    # 检测实际session数量
    session_count, _, session_nums = count_session_files(nii_dir)

    # 计算每个 session 的 trial 数量
    session_counts = {}
    for orderid in orderids:
        session_num = orderid // 750 + 1
        session_counts[session_num] = session_counts.get(session_num, 0) + 1

    print(f"\n{'=' * 60}")
    print(f"{subject_id} - 各 Session Trial 数量统计")
    print(f"{'=' * 60}")
    print(f"  实际 session 文件数量：{session_count}")
    print(f"  实际 session 编号：{session_nums}")
    print(f"  表情 trial 总数：{len(orderids)}")
    print(f"\n  各 session 分布:")

    for session in sorted(session_counts.keys()):
        count = session_counts[session]
        status = "✓" if session in session_nums else "✗ (文件缺失)"
        print(f"    Session {session:02d}: {count:4d} 个 trial {status}")

    print(f"{'=' * 60}\n")

    return session_counts


# ==================== 主程序 ====================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("NSD 数据集 - 8被试表情trial批量抽取")
    print(f"开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 配置基础路径
    BASE_DATA_DIR = r'D:\临时使用文件夹\NSD_Project\data\fmri\emotion'
    BASE_NII_DIR = r'D:\临时使用文件夹'
    OUTPUT_DIR = r'D:\临时使用文件夹\NSD_Project'

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 被试列表
    subjects = ['sub01', 'sub02', 'sub03', 'sub04', 'sub05', 'sub06', 'sub07', 'sub08']

    # 存储所有被试的处理结果
    results = []

    # 打印总体信息
    print(f"\n{'=' * 60}")
    print("被试列表及文件检测")
    print(f"{'=' * 60}")

    # 先检测所有被试的session文件数量
    subject_session_info = {}
    for sub in subjects:
        sub_nii_dir = os.path.join(BASE_NII_DIR, sub, 'betas_fithrf')
        if os.path.exists(sub_nii_dir):
            session_count, _, session_nums = count_session_files(sub_nii_dir)
            subject_session_info[sub] = {
                'session_count': session_count,
                'session_nums': session_nums,
                'exists': True
            }
            print(f"  {sub}: {session_count} 个 session 文件 (编号: {session_nums})")
        else:
            subject_session_info[sub] = {
                'session_count': 0,
                'session_nums': [],
                'exists': False
            }
            print(f"  {sub}: ✗ 目录不存在")

    print(f"{'=' * 60}\n")

    # 批量处理每个被试
    for i, sub in enumerate(subjects, 1):
        print(f"\n{'#' * 60}")
        print(f"# 被试 {i}/8: {sub}")
        print(f"{'#' * 60}\n")

        # 检查被试目录是否存在
        if not subject_session_info[sub]['exists']:
            print(f"  ⚠ 跳过 {sub} (目录不存在)")
            results.append({
                'subject_id': sub,
                'status': 'skipped',
                'reason': 'directory_not_found'
            })
            continue

        # 配置当前被试的路径
        subject_id = sub  # 如 'sub01'
        subject_num = sub.replace('sub', '')  # 如 '01'

        excel_path = os.path.join(BASE_DATA_DIR, f'{sub}_output.xlsx')
        nii_dir = os.path.join(BASE_NII_DIR, subject_id, 'betas_fithrf')
        output_path = os.path.join(OUTPUT_DIR, f'{subject_id}_emotion_trials.nii.gz')

        # 检查 Excel 文件是否存在
        if not os.path.exists(excel_path):
            print(f"  ⚠ 跳过 {sub} (Excel 文件不存在：{excel_path})")
            results.append({
                'subject_id': sub,
                'status': 'skipped',
                'reason': 'excel_not_found'
            })
            continue

        # 检查 nii 目录是否存在
        if not os.path.exists(nii_dir):
            print(f"  ⚠ 跳过 {sub} (NII 目录不存在：{nii_dir})")
            results.append({
                'subject_id': sub,
                'status': 'skipped',
                'reason': 'nii_dir_not_found'
            })
            continue

        try:
            # 1. 查看统计信息
            get_session_statistics(excel_path, nii_dir, subject_id)

            # 2. 执行抽取
            result = extract_emotion_trials(
                excel_path=excel_path,
                nii_dir=nii_dir,
                output_path=output_path,
                subject_id=subject_id
            )

            if result:
                # 3. 验证结果
                df = pd.read_excel(excel_path)
                orderids = sorted(df['orderid'].dropna().unique().astype(int))
                verify_result = verify_extraction(output_path, orderids, nii_dir, subject_id)

                result['verify_passed'] = verify_result
                result['status'] = 'completed'
                results.append(result)
            else:
                results.append({
                    'subject_id': sub,
                    'status': 'failed',
                    'reason': 'extraction_failed'
                })

            # 强制垃圾回收
            gc.collect()

        except Exception as e:
            print(f"\n  ✗ {sub} 处理出错：{str(e)}")
            results.append({
                'subject_id': sub,
                'status': 'error',
                'reason': str(e)
            })
            gc.collect()
            continue

    # ==================== 汇总报告 ====================
    print("\n" + "=" * 60)
    print("批量处理汇总报告")
    print(f"结束时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    print(f"\n总被试数：{len(subjects)}")
    print(f"成功完成：{sum(1 for r in results if r['status'] == 'completed')}")
    print(f"跳过：{sum(1 for r in results if r['status'] == 'skipped')}")
    print(f"失败：{sum(1 for r in results if r['status'] == 'failed' or r['status'] == 'error')}")

    print(f"\n{'=' * 60}")
    print("各被试详细信息")
    print(f"{'=' * 60}")
    print(f"{'被试':<10} {'状态':<12} {'Session数':<12} {'Trial数':<12} {'输出形状':<25}")
    print(f"{'-' * 60}")

    for r in results:
        subject_id = r['subject_id']
        status = r['status']

        if status == 'completed':
            session_count = r.get('session_count', 'N/A')
            num_trials = r.get('num_trials', 'N/A')
            output_shape = str(r.get('output_shape', 'N/A'))
            verify = "✓" if r.get('verify_passed', False) else "✗"
            print(
                f"{subject_id:<10} {status + ' ' + verify:<12} {session_count:<12} {num_trials:<12} {output_shape:<25}")
        else:
            reason = r.get('reason', 'unknown')
            print(f"{subject_id:<10} {status:<12} {reason:<37}")

    print(f"{'=' * 60}\n")

    # 保存汇总报告
    report_path = os.path.join(OUTPUT_DIR, 'batch_processing_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("NSD 数据集 - 8被试表情trial批量抽取报告\n")
        f.write(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        f.write("各被试详细信息:\n")
        f.write(f"{'被试':<10} {'状态':<12} {'Session数':<12} {'Trial数':<12} {'输出形状':<25}\n")
        f.write("-" * 60 + "\n")

        for r in results:
            subject_id = r['subject_id']
            status = r['status']

            if status == 'completed':
                session_count = r.get('session_count', 'N/A')
                num_trials = r.get('num_trials', 'N/A')
                output_shape = str(r.get('output_shape', 'N/A'))
                verify = "✓" if r.get('verify_passed', False) else "✗"
                f.write(
                    f"{subject_id:<10} {status + ' ' + verify:<12} {session_count:<12} {num_trials:<12} {output_shape:<25}\n")
            else:
                reason = r.get('reason', 'unknown')
                f.write(f"{subject_id:<10} {status:<12} {reason:<37}\n")

    print(f"✓ 汇总报告已保存：{report_path}\n")