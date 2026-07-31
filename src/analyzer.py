"""
统计分析模块
字频统计、字共现矩阵、关键词扩展
"""

import os
import json
from collections import defaultdict, Counter
from tqdm import tqdm


class PoetryAnalyzer:
    """诗歌统计分析器"""

    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(__file__), '..', 'data', 'processed'
        )
        os.makedirs(self.cache_dir, exist_ok=True)

        # 字频
        self.char_freq = Counter()
        # 句首字频
        self.line_start_freq = Counter()
        # 句末字频
        self.line_end_freq = Counter()
        # 相邻字共现: {char: {next_char: count}}
        self.cooccur = defaultdict(Counter)
        # 同句共现（语义关联）: {char: {co_char: count}}
        self.semantic_cooccur = defaultdict(Counter)

    def analyze(self, poems):
        """分析诗歌语料，构建统计模型"""
        print('正在分析诗歌语料...')

        for poem in tqdm(poems, desc='统计分析'):
            sentences = poem.get('sentences', [])

            for sent in sentences:
                if not sent:
                    continue

                # 字频统计
                for ch in sent:
                    if '一' <= ch <= '鿿':
                        self.char_freq[ch] += 1

                # 句首字
                if sent:
                    self.line_start_freq[sent[0]] += 1
                # 句末字
                if len(sent) > 1:
                    self.line_end_freq[sent[-1]] += 1

                # 相邻字共现 (bigram)
                for i in range(len(sent) - 1):
                    c1, c2 = sent[i], sent[i + 1]
                    if '一' <= c1 <= '鿿' and '一' <= c2 <= '鿿':
                        self.cooccur[c1][c2] += 1

                # 同句语义共现（相隔较远的字也计入，但权重递减）
                chars = [c for c in sent if '一' <= c <= '鿿']
                for i, c1 in enumerate(chars):
                    for j, c2 in enumerate(chars):
                        if i != j:
                            weight = 1.0 / (abs(i - j) + 1)
                            self.semantic_cooccur[c1][c2] += weight

        print(f'统计完成:')
        print(f'  不同汉字数: {len(self.char_freq)}')
        print(f'  字对(bigram)数: {sum(len(v) for v in self.cooccur.values())}')
        print(f'  句子数: {sum(self.line_start_freq.values())}')

    def analyze_jsonl(self, jsonl_path):
        """从 JSONL 文件流式分析诗歌语料（内存高效，逐行读取）"""
        print('正在分析诗歌语料...')

        line_count = 0
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc='统计分析'):
                poem = json.loads(line)
                sentences = poem.get('sentences', [])
                for sent in sentences:
                    if not sent:
                        continue
                    line_count += 1
                    for ch in sent:
                        if '一' <= ch <= '鿿':
                            self.char_freq[ch] += 1
                    if sent:
                        self.line_start_freq[sent[0]] += 1
                    if len(sent) > 1:
                        self.line_end_freq[sent[-1]] += 1
                    for i in range(len(sent) - 1):
                        c1, c2 = sent[i], sent[i + 1]
                        if '一' <= c1 <= '鿿' and '一' <= c2 <= '鿿':
                            self.cooccur[c1][c2] += 1
                    chars = [c for c in sent if '一' <= c <= '鿿']
                    for i, c1 in enumerate(chars):
                        for j, c2 in enumerate(chars):
                            if i != j:
                                weight = 1.0 / (abs(i - j) + 1)
                                self.semantic_cooccur[c1][c2] += weight

        print(f'统计完成:')
        print(f'  不同汉字数: {len(self.char_freq)}')
        print(f'  字对(bigram)数: {sum(len(v) for v in self.cooccur.values())}')
        print(f'  句子数: {line_count}')

    def get_char_frequency(self, char):
        """获取字的频率"""
        return self.char_freq.get(char, 0)

    def get_top_chars(self, n=100, category='all'):
        """获取频率最高的n个字"""
        if category == 'line_start':
            return self.line_start_freq.most_common(n)
        elif category == 'line_end':
            return self.line_end_freq.most_common(n)
        else:
            return self.char_freq.most_common(n)

    def get_most_likely_next(self, char, n=10):
        """
        获取给定字后面最可能出现的字
        基于bigram统计
        """
        if char not in self.cooccur:
            return []
        next_chars = self.cooccur[char]
        total = sum(next_chars.values())
        return [(c, cnt / total) for c, cnt in next_chars.most_common(n)]

    def get_related_keywords(self, keyword, n=20):
        """
        获取与关键词语义相关的字

        基于同句共现和相邻共现的综合
        """
        if keyword not in self.semantic_cooccur:
            return []

        related = Counter()
        # 语义相关
        for c, cnt in self.semantic_cooccur[keyword].most_common(50):
            related[c] += cnt * 2  # 语义权重高

        # 邻接相关
        if keyword in self.cooccur:
            for c, cnt in self.cooccur[keyword].most_common(30):
                related[c] += cnt

        return related.most_common(n)

    def expand_keywords(self, keywords, n_per_keyword=10):
        """
        扩展关键词集合

        给定一组关键词，找到与每个关键词语义相关的字，
        合并去重后返回扩展的关键词集合
        """
        expanded = set(keywords)
        for kw in keywords:
            related = self.get_related_keywords(kw, n=n_per_keyword)
            for c, _ in related:
                expanded.add(c)
        return list(expanded)

    def save(self):
        """保存统计结果"""
        data = {
            'char_freq': dict(self.char_freq.most_common(10000)),
            'line_start_freq': dict(self.line_start_freq.most_common(5000)),
            'line_end_freq': dict(self.line_end_freq.most_common(5000)),
            'cooccur': {k: dict(v.most_common(200)) for k, v in self.cooccur.items()
                        if sum(v.values()) >= 5},  # 过滤低频
            'semantic_cooccur': {k: dict(v.most_common(100)) for k, v in self.semantic_cooccur.items()
                                 if sum(v.values()) >= 3},
        }

        filepath = os.path.join(self.cache_dir, 'statistics.json')
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        print(f'统计数据已保存至: {filepath}')

    def load(self):
        """加载统计结果"""
        filepath = os.path.join(self.cache_dir, 'statistics.json')
        if not os.path.exists(filepath):
            print(f'未找到统计数据文件: {filepath}')
            return False

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.char_freq = Counter(data.get('char_freq', {}))
        self.line_start_freq = Counter(data.get('line_start_freq', {}))
        self.line_end_freq = Counter(data.get('line_end_freq', {}))

        self.cooccur = defaultdict(Counter)
        for k, v in data.get('cooccur', {}).items():
            self.cooccur[k] = Counter(v)

        self.semantic_cooccur = defaultdict(Counter)
        for k, v in data.get('semantic_cooccur', {}).items():
            self.semantic_cooccur[k] = Counter(v)

        print(f'统计数据已加载 ({len(self.char_freq)} 字)')
        return True


if __name__ == '__main__':
    from data_loader import load_processed_poems

    poems = load_processed_poems()
    if poems:
        analyzer = PoetryAnalyzer()
        analyzer.analyze(poems)
        analyzer.save()

        # 测试
        print('\n===== 高频字 Top 20 =====')
        for c, f in analyzer.get_top_chars(20):
            print(f'  {c}: {f}')

        print('\n===== "春"后面最可能出现的字 =====')
        for c, prob in analyzer.get_most_likely_next('春', 10):
            print(f'  春{c}: {prob:.4f}')

        print('\n===== "春"的相关词 =====')
        for c, score in analyzer.get_related_keywords('春', 10):
            print(f'  {c}: {score:.1f}')
