# run_encoding_and_save.py
import os
import numpy as np
import pandas as pd
import torch
import clip
from tqdm import tqdm
import pickle

# 导入您原有的配置和函数
from your_main_script import cfg, stage2_encoding_eval_save_voxelwise


def run_all_subjects_and_save():
    """
    对所有被试和情绪条件运行编码模型，并保存voxel-level的R²结果
    """
    os.makedirs(cfg['output_dir'], exist_ok=True)

    # 加载CLIP模型
    backbone = cfg['backbone']
    clip_model, preprocess = clip.load(backbone, device=cfg['device'], jit=False)
    clip_model = clip_model.float()
    print(f"Using CLIP model: {backbone}")

    all_results = []

    for emo in cfg['emotions']:
        print(f"\n{'=' * 60}")
        print(f"Processing emotion: {emo}")
        print(f"{'=' * 60}")

        for subj in cfg['subjects']:
            print(f"\n--- Subject: {subj} ---")

            # 运行评估并获取voxel-level结果
            result = stage2_encoding_eval_save_voxelwise(
                clip_model, preprocess, cfg,
                subject_id=subj,
                emotion=emo,
                save_results=True
            )

            if result is not None:
                summary_df, voxel_r2, voxel_pvalues = result

                # 保存voxel-level的R²和p值
                save_path = os.path.join(
                    cfg['output_dir'],
                    f"{backbone.replace('/', '_')}_{subj}_{emo}_voxel_r2.pkl"
                )
                with open(save_path, 'wb') as f:
                    pickle.dump({
                        'r2': voxel_r2,
                        'pvalues': voxel_pvalues,
                        'subject': subj,
                        'emotion': emo,
                        'model': backbone
                    }, f)
                print(f"Saved voxel-level results to {save_path}")

                all_results.append(summary_df)

    # 保存汇总统计
    if all_results:
        summary_df = pd.concat(all_results, ignore_index=True)
        summary_path = os.path.join(cfg['output_dir'],
                                    f"{backbone.replace('/', '_')}_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    run_all_subjects_and_save()