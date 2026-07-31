"""
内存高效训练管线 —— 批量处理85万首诗歌
避免内存溢出，逐文件加载、增量训练
"""

import os
import sys
import json
import gc
from collections import Counter
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import split_sentences, classify_poem_type, extract_chinese
from src.rhyme_dict import RhymeDict
from src.analyzer import PoetryAnalyzer

def process_single_csv(filepath):
    """处理单个CSV文件，返回清洗后的诗歌列表（分批释放内存）"""
    import pandas as pd
    poems = []
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except Exception as e:
        print(f'  跳过 {os.path.basename(filepath)}: {e}')
        return poems

    content_col = None
    for col in ['内容', 'content']:
        if col in df.columns:
            content_col = col
            break
    if not content_col:
        return poems

    for _, row in df.iterrows():
        content = str(row[content_col]) if pd.notna(row[content_col]) else ''
        if not content or len(content) < 4:
            continue
        chinese_only = extract_chinese(content)
        if len(chinese_only) < 8:
            continue
        sentences = split_sentences(content)
        if len(sentences) < 2:
            continue
        poem_type = classify_poem_type(sentences)
        poems.append({
            'content': chinese_only,
            'sentences': sentences,
            'type': poem_type,
        })
    return poems


def train_all(raw_dir='raw_poetry', processed_dir='data/processed'):
    """
    内存高效的完整训练管线
    Step 1: 逐文件处理，保存清洗数据
    Step 2: 增量统计分析
    Step 3: 构建韵部
    Step 4: 构建韵部
    """
    # 解析为项目根目录下的绝对路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(raw_dir):
        raw_dir = os.path.join(project_root, raw_dir)
    if not os.path.isabs(processed_dir):
        processed_dir = os.path.join(project_root, processed_dir)

    os.makedirs(processed_dir, exist_ok=True)

    csv_files = sorted([f for f in os.listdir(raw_dir) if f.endswith('.csv')])
    print(f'找到 {len(csv_files)} 个CSV文件\n')

    # ============================================================
    # Step 1: 逐文件处理并保存
    # ============================================================
    print('=' * 60)
    print('[1/3] 逐文件加载、清洗、保存')
    print('=' * 60)

    all_poems_path = os.path.join(processed_dir, 'all_poems.jsonl')
    corpus_path = os.path.join(processed_dir, 'corpus_all.txt')

    total_poems = 0
    total_chars = 0
    type_counts = Counter()

    with open(all_poems_path, 'w', encoding='utf-8') as f_jsonl, \
         open(corpus_path, 'w', encoding='utf-8') as f_corpus:

        for csv_file in tqdm(csv_files, desc='处理CSV'):
            filepath = os.path.join(raw_dir, csv_file)
            poems = process_single_csv(filepath)

            for poem in poems:
                # 写入JSONL
                f_jsonl.write(json.dumps(poem, ensure_ascii=False) + '\n')
                # 写入语料
                f_corpus.write(poem['content'] + '\n')
                type_counts[poem['type']] += 1
                total_chars += len(poem['content'])

            total_poems += len(poems)
            # 释放内存
            del poems
            gc.collect()

    print(f'\n总诗歌数: {total_poems}')
    print(f'总字符数: {total_chars}')
    print(f'诗体分布 (Top 10):')
    for t, c in type_counts.most_common(10):
        print(f'  {t}: {c}')

    # ============================================================
    # Step 2: 统计分析（流式读取JSONL，内存高效）
    # ============================================================
    print('\n' + '=' * 60)
    print('[2/3] 统计分析（流式）')
    print('=' * 60)

    analyzer = PoetryAnalyzer()
    analyzer.analyze_jsonl(all_poems_path)
    analyzer.save()

    # ============================================================
    # Step 3: 构建韵部字典（流式读取JSONL，内存高效）
    # ============================================================
    print('\n' + '=' * 60)
    print('[3/3] 构建韵部字典（流式）')
    print('=' * 60)

    rhyme_dict = RhymeDict()
    rhyme_dict.build_from_jsonl(all_poems_path, total_poems)

    # ============================================================
    # 完成
    # ============================================================
    print('\n' + '=' * 60)
    print('  预处理完成!')
    print(f'  诗歌: {total_poems} 首')
    print(f'  词汇: {len(analyzer.char_freq)} 字')
    print(f'  韵部: {len(rhyme_dict.rhyme_groups)} 个')
    print('')
    print('  Transformer 模型训练请使用 cloud_train.py 上传 AutoDL 云端训练')
    print('=' * 60)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-dir', default='raw_poetry')
    parser.add_argument('--processed-dir', default='data/processed')
    args = parser.parse_args()

    train_all(args.raw_dir, args.processed_dir)
