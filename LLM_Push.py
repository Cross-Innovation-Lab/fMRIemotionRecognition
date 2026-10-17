#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
batch_generate_captions_test.py
从 Excel 读取标签，调用 qwen-vl-plus 为前 N 张图片各生成一句包含情绪词的自然语言描述。
先跑 20 条看看效果！
"""

import os
import sys
import base64
import time
import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ==================== 配置 ====================
# 你的图片文件夹
IMAGE_DIR = r"..\data\fmri\emotion\coco(nsd)"

# 标签 Excel 文件路径
LABELS_XLSX = r"..\data\fmri\emotion\output.xlsx"

# 输出 CSV
OUTPUT_CSV = r".\image_text_pairs_test20.csv"

MODEL_NAME = "*****"

# ---------- 运行控制 ----------
MAX_WORKERS = 1
MAX_RETRIES = 3          # 失败重试次数
SLEEP_ON_RATE_LIMIT = 2  # 限流等待秒数
N_SAMPLE = None            # ★ 先测试 20 条；全量生成时改成 None

# ---------- 情绪关键词映射 ----------
EMOTION_KEYWORD_MAP = {
    "happy":    "happy",
    "sad":      "sad",
    "angry":    "angry",
    "fear":     "fearful",
    "neutral":  "neutral",
    "surprise": "surprised",
}

# ---------- System Prompt 模板 ----------
SYSTEM_PROMPT = """你是一位视觉情感描述专家。观察这张人脸图片，用一句简短的英文描述人物此刻正在表达的情绪。描述要自然生动，可以参照人物的性别、年龄、神态、面部特征（如眼神、嘴角、皱纹等）和整体氛围来体现情绪，但不得直接罗列特征。句子中必须恰好出现一次单词 "{emotion_word}"，让它自然地融入句中。每一张人脸都是独一无二的，因此你的描述也必须各不相同，避免任何重复、模板化或泛泛的表达。严禁输出任何解释、括号备注、字数统计、数学符号或多余文字。注意，只输出这一句英文本身，不加引号。长度严格控制在20个单词以内。"""
# ==================== 工具函数 ====================
def encode_image_to_base64(image_path: str) -> str:
    """本地图片 → data URI"""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png", ".webp": "webp"}
    mime = mime_map.get(ext, "jpeg")
    return f"data:image/{mime};base64,{b64}"


def generate_caption(row, client: OpenAI):
    """
    为一行记录生成描述。
    返回 (image_id, emotion, caption) 或 (image_id, emotion, None)
    """
    filename = str(int(row["filename"]))
    emotion = row["predicted_emotion"].strip().lower()
    emotion_word = EMOTION_KEYWORD_MAP.get(emotion, emotion)
    system_prompt = SYSTEM_PROMPT.format(emotion_word=emotion_word)

    # 拼接图片文件名（补齐 12 位）
    image_id = filename.zfill(12)  # 例如 "294" → "000000000294"
    image_name = f"{image_id}_face_0.jpg"
    image_path = os.path.join(IMAGE_DIR, image_name)

    if not os.path.exists(image_path):
        # 尝试其他扩展名
        for alt_ext in [".png", ".jpeg", ".webp"]:
            alt_path = os.path.join(IMAGE_DIR, f"{image_id}_face_0{alt_ext}")
            if os.path.exists(alt_path):
                image_path = alt_path
                break
        else:
            print(f"  ⚠ 图片不存在: {image_name}")
            return (image_id, emotion, None)

    # 编码
    try:
        b64_uri = encode_image_to_base64(image_path)
    except Exception as e:
        print(f"  ✗ 编码失败 {image_name}: {e}")
        return (image_id, emotion, None)

    # 调用 API（带重试）
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": b64_uri}},
                        {"type": "text", "text": "Describe this face in one sentence."}
                    ]}
                ],
                # max_tokens=80,
                # temperature=0.8,
                # top_p=0.95
            )
            caption = response.choices[0].message.content.strip()

            # 简单校验：必须包含情绪关键词
            if emotion_word.lower() not in caption.lower():
                print(f"  ⚠ 输出不含'{emotion_word}'，重试: {caption}")
                continue

            return (image_id, emotion, caption)

        except Exception as e:
            err_msg = str(e).lower()
            if any(kw in err_msg for kw in ["rate", "throttling", "429", "too many requests"]):
                print(f"  ⏳ 限流，等待 {SLEEP_ON_RATE_LIMIT}s...")
                time.sleep(SLEEP_ON_RATE_LIMIT)
            else:
                if attempt < MAX_RETRIES - 1:
                    print(f"  ↻ 重试 (attempt {attempt+1}): {str(e)[:80]}")
                    time.sleep(1)
                else:
                    print(f"  ✗ 失败: {str(e)[:80]}")

    return (image_id, emotion, None)


# ==================== 主函数 ====================
def main():
    # 1. 读取标签
    if not os.path.exists(LABELS_XLSX):
        print(f"错误：标签文件不存在: {LABELS_XLSX}")
        sys.exit(1)

    df = pd.read_excel(LABELS_XLSX, dtype={'image_id': str})
    print(f"共加载 {len(df)} 条标签记录")
    print("情绪分布:")
    print(df["predicted_emotion"].value_counts().to_string())

    # 2. 取前 N 条
    if N_SAMPLE is not None:
        df = df.head(N_SAMPLE)
        print(f"★ 测试模式：仅处理前 {len(df)} 条")

    # 3. 初始化客户端
    client = OpenAI(
        base_url="*****",
        api_key="*****",
    )

    # 4. 批量生成
    results = []
    failed = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_row = {
            executor.submit(generate_caption, row, client): row
            for _, row in df.iterrows()
        }

        with tqdm(total=len(future_to_row), desc="生成文本") as pbar:
            for future in as_completed(future_to_row):
                image_id, emotion, caption = future.result()
                if caption is not None:
                    results.append({
                        "image_id": image_id,
                        "emotion": emotion,
                        "caption": caption
                    })
                else:
                    failed.append(image_id)
                pbar.update(1)

    # 5. 保存结果
    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✅ 完成！成功: {len(results)}, 失败: {len(failed)}")
    print(f"结果已保存至: {OUTPUT_CSV}")

    # 6. 预览前 10 条
    print("\n" + "=" * 60)
    print("预览前 10 条:")
    print("=" * 60)
    for _, row in result_df.head(10).iterrows():
        print(f"[{row['emotion']:10s}] {row['image_id']}: {row['caption']}")

    # 7. 保存失败清单
    if failed:
        fail_path = OUTPUT_CSV.replace(".csv", "_failed.csv")
        pd.DataFrame({"image_id": failed}).to_csv(fail_path, index=False)
        print(f"失败清单已保存至: {fail_path}")


import pandas as pd


def find_duplicate_captions(csv_path: str):
    """
    读取已生成的 CSV，找出 caption 完全重复的行（保留第一个，返回其余需重新生成的记录）。

    返回:
        dup_records: list[dict]
            每个字典包含 'image_id', 'emotion', 'original_caption'，
            代表因重复而需要重新生成的条目。
    """
    if not os.path.exists(csv_path):
        print(f"[check] 文件不存在: {csv_path}")
        return []

    df = pd.read_csv(csv_path, dtype={'image_id': str})
    if 'caption' not in df.columns or 'image_id' not in df.columns:
        print("[check] CSV 缺少必要列 'caption' 或 'image_id'")
        return []

    # 找出所有重复的 caption
    dup_mask = df.duplicated(subset='caption', keep='first')  # keep='first' 保留第一条
    dup_df = df[dup_mask].copy()

    print(f"[check] 发现 {len(dup_df)} 条重复描述（共 {dup_mask.sum()} 行）")
    dup_records = dup_df[['image_id', 'emotion']].to_dict(orient='records')
    # 也可保留原始 caption 供参考
    for rec, (_, row) in zip(dup_records, dup_df.iterrows()):
        rec['original_caption'] = row['caption']

    return dup_records


def regenerate_failed_and_duplicates(
        output_csv=OUTPUT_CSV,
        labels_xlsx=LABELS_XLSX,
        image_dir=IMAGE_DIR,
        client=None,
        max_workers=MAX_WORKERS,
):
    """
    1. 读取成功CSV，检测其中的重复caption。
    2. 与标签Excel对比，找出尚未生成文本的图片（缺失项）。
    3. 调用 generate_caption 为重复项和缺失项重新生成。
    4. 合并原有的非重复成功记录与新生成记录，写回 output_csv。
    """
    # ---- 初始化 ----
    if client is None:
        client = OpenAI(base_url="****", api_key="ollama")

    # 1. 读取成功CSV
    if not os.path.exists(output_csv):
        print(f"[regenerate] 找不到 {output_csv}，视为无成功记录")
        success_df = pd.DataFrame(columns=['image_id', 'emotion', 'caption'])
    else:
        success_df = pd.read_csv(output_csv, dtype={'image_id': str})
        print(f"[regenerate] 原有成功记录: {len(success_df)} 条")

    # 2. 读取标签Excel，建立全量图片→情绪映射
    labels_df = pd.read_excel(labels_xlsx, dtype={'filename': str})  # 统一用str读，避免float/int
    # 补零为12位字符串作为 image_id
    labels_df['image_id'] = labels_df['filename'].apply(lambda x: str(int(float(x))).zfill(12))
    # 情绪列统一小写去空格
    labels_df['emotion'] = labels_df['predicted_emotion'].str.strip().str.lower()
    # 图片ID → 情绪映射
    all_images = dict(zip(labels_df['image_id'], labels_df['emotion']))
    print(f"[regenerate] 标签文件中共有 {len(all_images)} 张图片")

    # 3. 找出需要重新生成的条目（两种来源）
    to_regenerate = []  # list of dict: {'image_id': ..., 'emotion': ...}

    # 3.1 重复项（从成功记录中查找）
    dup_records = find_duplicate_captions(output_csv)
    to_regenerate.extend(dup_records)
    print(f"[regenerate] 重复描述需重新生成: {len(dup_records)} 条")

    # 3.2 缺失项（标签中存在但成功CSV中不存在的图片）
    existing_ids = set(success_df['image_id']) if not success_df.empty else set()
    missing_ids = set(all_images.keys()) - existing_ids
    # 只保留在标签中有情绪信息的图片（一般情况下都会有）
    missing_records = [{'image_id': img_id, 'emotion': all_images[img_id]} for img_id in missing_ids if img_id in all_images]
    print(f"[regenerate] 缺失（未生成）条目: {len(missing_records)} 条")
    to_regenerate.extend(missing_records)

    if not to_regenerate:
        print("[regenerate] 没有需要重新生成的条目，退出。")
        return

    # 去重 (可能在重复和缺失中有重叠)
    seen = set()
    unique_reg = []
    for item in to_regenerate:
        if item['image_id'] not in seen:
            seen.add(item['image_id'])
            unique_reg.append(item)
    to_regenerate = unique_reg
    print(f"[regenerate] 最终需重新生成: {len(to_regenerate)} 条")

    # 4. 重新生成
    new_results = []
    for item in tqdm(to_regenerate, desc="重新生成"):
        try:
            filename_int = int(item['image_id'])
        except ValueError:
            print(f"  ⚠ 无法解析 image_id: {item['image_id']}，跳过")
            continue
        row = pd.Series({
            'filename': filename_int,
            'predicted_emotion': item['emotion']
        })
        img_id, emotion, caption = generate_caption(row, client)
        if caption is not None:
            new_results.append({
                'image_id': img_id,
                'emotion': emotion,
                'caption': caption
            })
        else:
            print(f"  ✗ 重新生成失败: {item['image_id']}")

    # 5. 合并结果
    reg_ids_set = {item['image_id'] for item in to_regenerate}
    if not success_df.empty:
        preserved = success_df[~success_df['image_id'].isin(reg_ids_set)]
    else:
        preserved = pd.DataFrame(columns=['image_id', 'emotion', 'caption'])

    new_df = pd.DataFrame(new_results)
    merged = pd.concat([preserved, new_df], ignore_index=True)
    merged = merged.drop_duplicates(subset='image_id', keep='last')

    # 6. 保存
    merged.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"[regenerate] 完成！最终记录数: {len(merged)}，已保存至 {output_csv}")

    # 7. 更新失败清单（针对仍然生成失败的图片）
    final_success_ids = set(merged['image_id'])
    final_failed_ids = reg_ids_set - final_success_ids
    if final_failed_ids:
        fail_out = output_csv.replace('.csv', '_failed.csv')
        pd.DataFrame({'image_id': list(final_failed_ids)}).to_csv(fail_out, index=False)
        print(f"[regenerate] 仍有 {len(final_failed_ids)} 条失败，保存至 {fail_out}")
    else:
        # 如果之前有失败清单，且现在全部成功，删除之
        old_fail_csv = output_csv.replace('.csv', '_failed.csv')
        if os.path.exists(old_fail_csv):
            os.remove(old_fail_csv)
            print(f"[regenerate] 所有条目生成成功，已删除旧失败清单")


if __name__ == "__main__":
    # main()
    client = OpenAI(base_url="****", api_key="ollama")
    regenerate_failed_and_duplicates(
        output_csv=OUTPUT_CSV,
        labels_xlsx=LABELS_XLSX,
        image_dir=IMAGE_DIR,
        client=client,
        max_workers=MAX_WORKERS,
    )