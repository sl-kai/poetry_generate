"""
数据加载与预处理模块
负责读取原始CSV语料、清洗诗歌文本、按诗体分类、输出处理后数据
"""

import os
import re
import json

# 中文字符正则
CHINESE_CHAR_PATTERN = re.compile(r'[一-鿿]')
# 中文标点（用于分割句子）
CHINESE_PUNCT = set('，。！？；：、．，？！；：')

# 常见中文标点（全角+半角），用于清洗
PUNCTUATION_PATTERN = re.compile(
    r'[，。！？；：、．,\.!\?;:\s　 -‏ -  ﻿]'
)


def extract_chinese(text):
    """提取文本中的中文字符（去除标点、空格、数字等）"""
    return ''.join(CHINESE_CHAR_PATTERN.findall(str(text)))


def split_sentences(poem_text):
    """
    将一首诗按标点分割为句子列表
    只保留中文字符
    """
    # 先按常见标点分割
    parts = re.split(r'[，。！？；：、．\s]+', str(poem_text))
    # 每部分只保留中文字符
    sentences = [extract_chinese(p) for p in parts if extract_chinese(p)]
    return sentences


def classify_poem_type(sentences):
    """
    根据句子长度分布判断诗体类型
    返回: '五言绝句', '五言律诗', '七言绝句', '七言律诗', '四言', '杂言', '词', '其他'
    """
    if not sentences:
        return '其他'

    lengths = [len(s) for s in sentences]
    avg_len = sum(lengths) / len(lengths)
    n_sentences = len(sentences)

    # 词：句子长度变化大（方差大）
    if len(set(lengths)) > 3 and n_sentences > 2:
        length_var = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)
        if length_var > 4:
            return '词'

    # 检查是否所有句子长度相同
    if len(set(lengths)) == 1:
        l = lengths[0]
        if l == 5:
            if n_sentences == 4:
                return '五言绝句'
            elif n_sentences == 8:
                return '五言律诗'
            elif 4 <= n_sentences <= 16:
                return '五言诗'
        elif l == 7:
            if n_sentences == 4:
                return '七言绝句'
            elif n_sentences == 8:
                return '七言律诗'
            elif 4 <= n_sentences <= 16:
                return '七言诗'
        elif l == 4:
            return '四言诗'
        else:
            return f'{l}言诗'

    # 主要都是5言
    if all(l in [4, 5, 6] for l in lengths) and lengths.count(5) / len(lengths) > 0.7:
        if n_sentences == 4:
            return '五言绝句'
        elif n_sentences == 8:
            return '五言律诗'
        return '五言诗'

    # 主要都是7言
    if all(l in [6, 7, 8] for l in lengths) and lengths.count(7) / len(lengths) > 0.7:
        if n_sentences == 4:
            return '七言绝句'
        elif n_sentences == 8:
            return '七言律诗'
        return '七言诗'

    return '杂言'



def load_processed_poems(processed_dir='data/processed'):
    """加载已处理的诗歌数据（支持 JSON 和 JSONL 格式）"""
    # 优先尝试 JSONL（train_pipeline.py 的输出格式）
    jsonl_path = os.path.join(processed_dir, 'all_poems.jsonl')
    json_path = os.path.join(processed_dir, 'all_poems.json')

    if os.path.exists(jsonl_path):
        poems = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    poems.append(json.loads(line))
        return poems
    elif os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        print(f'未找到处理后数据，请先运行: python main.py train')
        return []


if __name__ == '__main__':
    # 测试核心函数
    print('=== data_loader 核心函数测试 ===')

    test_text = '春风又绿江南岸，明月何时照我还？'
    print(f'原始文本: {test_text}')

    chinese = extract_chinese(test_text)
    print(f'提取中文: {chinese}')

    sents = split_sentences(test_text)
    print(f'分句结果: {sents}')

    poem_type = classify_poem_type(sents)
    print(f'诗体判定: {poem_type}')

    # 测试加载已处理数据
    poems = load_processed_poems()
    if poems:
        print(f"\n已加载 {len(poems)} 首诗歌")
        p = poems[0]
        print("示例: [{}] {}...".format(p["type"], p["content"][:40]))
    else:
        print("未找到已处理数据，请先运行: python main.py train")
