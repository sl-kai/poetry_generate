"""
韵部字典模块
基于语料库统计构建韵部，用于押韵判断
偶句末字押韵是古诗的基本特征之一
"""

import os
import json
from collections import defaultdict

from tqdm import tqdm

try:
    from pypinyin import pinyin, Style
    HAS_PYPINYIN = True
except ImportError:
    HAS_PYPINYIN = False


class RhymeDict:
    """韵律字典"""

    def __init__(self, cache_path=None):
        self._cache_path = cache_path or os.path.join(
            os.path.dirname(__file__), '..', 'data', 'processed', 'rhyme_cache.json'
        )
        # rhyme_groups: {韵母: [字列表]}
        self.rhyme_groups = defaultdict(list)
        # char_to_rhyme: {字: 韵母}
        self.char_to_rhyme = {}
        self._load_cache()

    def _load_cache(self):
        if os.path.exists(self._cache_path):
            try:
                with open(self._cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.rhyme_groups = defaultdict(list, data.get('rhyme_groups', {}))
                    self.char_to_rhyme = data.get('char_to_rhyme', {})
            except Exception:
                pass

    def save_cache(self):
        os.makedirs(os.path.dirname(self._cache_path), exist_ok=True)
        with open(self._cache_path, 'w', encoding='utf-8') as f:
            json.dump({
                'rhyme_groups': dict(self.rhyme_groups),
                'char_to_rhyme': self.char_to_rhyme,
            }, f, ensure_ascii=False, indent=2)

    def get_final(self, char):
        """获取汉字的韵母（韵腹+韵尾）"""
        if not char or len(char) != 1:
            return None

        if char in self.char_to_rhyme:
            return self.char_to_rhyme[char]

        if not HAS_PYPINYIN:
            return 'unknown'

        try:
            py = pinyin(char, style=Style.FINALS)
            if py and py[0]:
                final = py[0][0]
            else:
                final = 'unknown'
        except Exception:
            final = 'unknown'

        self.char_to_rhyme[char] = final
        return final

    def build_from_corpus(self, poems):
        """
        从诗歌语料中构建韵部

        统计偶句末字的韵母共现情况，建立韵部分组
        """
        print('正在构建韵部字典...')

        rhyme_cooccur = defaultdict(lambda: defaultdict(int))

        for poem in poems:
            sentences = poem.get('sentences', [])
            if len(sentences) < 4:
                continue

            # 提取偶数句(1-based: 2,4,6,8...)的末字
            rhyme_chars = []
            for i in range(1, len(sentences), 2):  # 0-based: 1,3,5... 是偶数句
                s = sentences[i]
                if s:
                    rhyme_chars.append(s[-1])

            # 记录这些韵脚字的韵母共现
            for i, c1 in enumerate(rhyme_chars):
                final1 = self.get_final(c1)
                if not final1:
                    continue
                self.rhyme_groups[final1].append(c1)
                for c2 in rhyme_chars[i + 1:]:
                    final2 = self.get_final(c2)
                    if final2:
                        rhyme_cooccur[final1][final2] += 1
                        rhyme_cooccur[final2][final1] += 1

        # 去重
        for final in self.rhyme_groups:
            self.rhyme_groups[final] = list(set(self.rhyme_groups[final]))

        # 合并可通押的韵部
        self._merge_similar_rhymes(rhyme_cooccur)

        total_chars = sum(len(v) for v in self.rhyme_groups.values())
        print(f'构建完成: {len(self.rhyme_groups)} 个韵部, 覆盖 {total_chars} 个汉字')
        self.save_cache()

    def build_from_jsonl(self, jsonl_path, total_poems=None):
        """从 JSONL 文件流式构建韵部字典（内存高效，逐行读取）"""
        print('正在构建韵部字典...')

        if not HAS_PYPINYIN:
            print('警告: pypinyin 未安装')
            return

        from pypinyin import pinyin, Style

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc='构建韵部', total=total_poems):
                poem = json.loads(line)
                sentences = poem.get('sentences', [])
                if len(sentences) < 4:
                    continue

                rhyme_chars = []
                for i in range(1, len(sentences), 2):
                    s = sentences[i]
                    if s:
                        rhyme_chars.append(s[-1])

                for c in rhyme_chars:
                    if c in self.char_to_rhyme:
                        final = self.char_to_rhyme[c]
                    else:
                        try:
                            py = pinyin(c, style=Style.FINALS)
                            final = py[0][0] if py and py[0] else 'unknown'
                        except Exception:
                            final = 'unknown'
                        self.char_to_rhyme[c] = final

                    if final and final != 'unknown':
                        self.rhyme_groups[final].append(c)

        # 去重
        for final in self.rhyme_groups:
            self.rhyme_groups[final] = list(set(self.rhyme_groups[final]))

        total_chars = sum(len(v) for v in self.rhyme_groups.values())
        print(f'构建完成: {len(self.rhyme_groups)} 个韵部, 覆盖 {total_chars} 个汉字')
        self.save_cache()

    def _merge_similar_rhymes(self, rhyme_cooccur):
        """合并经常通押的韵部"""
        # 常见的通押对（基于《平水韵》与现代拼音的对应）
        merge_pairs = [
            ('ong', 'eng'), ('ing', 'eng'), ('en', 'eng'),
            ('in', 'ing'), ('an', 'ang'), ('ian', 'iang'),
            ('ui', 'ei'), ('ou', 'iu'),
        ]
        for f1, f2 in merge_pairs:
            if f1 in self.rhyme_groups and f2 in self.rhyme_groups:
                self.rhyme_groups[f1].extend(self.rhyme_groups[f2])
                self.rhyme_groups[f1] = list(set(self.rhyme_groups[f1]))
                # 不删除f2，在查询时会通过merge信息找到

    def get_rhyme_group(self, char):
        """获取某个字的韵部（韵母）"""
        return self.get_final(char)

    def get_rhyme_chars(self, rhyme_key, limit=None):
        """
        获取某个韵部下的所有字

        Args:
            rhyme_key: 韵母 或 一个代表字
            limit: 最多返回字数
        """
        # 如果传入的是字，先获取其韵部
        if len(rhyme_key) == 1 and '一' <= rhyme_key <= '鿿':
            rhyme_key = self.get_final(rhyme_key)

        chars = self.rhyme_groups.get(rhyme_key, [])
        if limit:
            return chars[:limit]
        return chars

    def check_rhyme(self, char1, char2):
        """
        检查两个字是否押韵（韵母相同或属于可通押的韵部）

        Returns:
            True 如果押韵
        """
        f1 = self.get_final(char1)
        f2 = self.get_final(char2)

        if not f1 or not f2:
            return False

        if f1 == f2:
            return True

        # 检查是否在同一韵部组
        if f1 in self.rhyme_groups and char2 in self.rhyme_groups[f1]:
            return True
        if f2 in self.rhyme_groups and char1 in self.rhyme_groups[f2]:
            return True

        return False

    def find_rhyming_chars(self, char, count=10, corpus_char_freq=None):
        """
        找到与给定字押韵的字列表，按常见度排序

        Args:
            char: 参考字
            count: 返回数量
            corpus_char_freq: 字频字典 {char: freq}

        Returns:
            [(字, 频率), ...] 按频率降序
        """
        rhyme_key = self.get_final(char)
        chars = self.get_rhyme_chars(rhyme_key)

        if corpus_char_freq:
            chars_with_freq = [(c, corpus_char_freq.get(c, 0)) for c in chars if c != char]
            chars_with_freq.sort(key=lambda x: -x[1])
            return chars_with_freq[:count]
        else:
            return [(c, 0) for c in chars[:count] if c != char]


if __name__ == '__main__':
    rd = RhymeDict()
    # 测试
    print('韵母测试:')
    for c in ['春', '风', '明', '月', '花', '红', '东', '同']:
        print(f'  {c}: {rd.get_final(c)}')

    print(f"\n'红'与'东'押韵? {rd.check_rhyme('红', '东')}")
    print(f"'红'与'明'押韵? {rd.check_rhyme('红', '明')}")
