"""
平仄字典模块
基于 pypinyin 获取汉字声调，映射为平仄
规则：1声（阴平）和2声（阳平）→ 平；3声（上声）和4声（去声）→ 仄
参考：普通话四声与中古四声的对应关系
"""

import json
import os

try:
    from pypinyin import pinyin, Style
    HAS_PYPINYIN = True
except ImportError:
    HAS_PYPINYIN = False
    print('警告: pypinyin 未安装，将使用简化平仄字典。请运行: pip install pypinyin')

# 平仄模板
# 五言绝句四格式（以首句为准）
WUYAN_TEMPLATES = {
    '仄起首句不入韵': [
        '仄仄平平仄',
        '平平仄仄平',
        '平平平仄仄',
        '仄仄仄平平',
    ],
    '仄起首句入韵': [
        '仄仄仄平平',
        '平平仄仄平',
        '平平平仄仄',
        '仄仄仄平平',
    ],
    '平起首句不入韵': [
        '平平平仄仄',
        '仄仄仄平平',
        '仄仄平平仄',
        '平平仄仄平',
    ],
    '平起首句入韵': [
        '平平仄仄平',
        '仄仄仄平平',
        '仄仄平平仄',
        '平平仄仄平',
    ],
}

# 七言绝句四格式
QIYAN_TEMPLATES = {
    '仄起首句不入韵': [
        '仄仄平平平仄仄',
        '平平仄仄仄平平',
        '平平仄仄平平仄',
        '仄仄平平仄仄平',
    ],
    '仄起首句入韵': [
        '仄仄平平仄仄平',
        '平平仄仄仄平平',
        '平平仄仄平平仄',
        '仄仄平平仄仄平',
    ],
    '平起首句不入韵': [
        '平平仄仄平平仄',
        '仄仄平平仄仄平',
        '仄仄平平平仄仄',
        '平平仄仄仄平平',
    ],
    '平起首句入韵': [
        '平平仄仄仄平平',
        '仄仄平平仄仄平',
        '仄仄平平平仄仄',
        '平平仄仄仄平平',
    ],
}



class ToneDict:
    """平仄字典"""

    def __init__(self, cache_path=None):
        self._tone_cache = {}
        self._cache_path = cache_path or os.path.join(
            os.path.dirname(__file__), '..', 'data', 'processed', 'tone_cache.json'
        )
        self._load_cache()

    def _load_cache(self):
        """从文件加载缓存的平仄数据"""
        if os.path.exists(self._cache_path):
            try:
                with open(self._cache_path, 'r', encoding='utf-8') as f:
                    self._tone_cache = json.load(f)
            except Exception:
                pass

    def save_cache(self):
        """保存平仄缓存到文件"""
        os.makedirs(os.path.dirname(self._cache_path), exist_ok=True)
        with open(self._cache_path, 'w', encoding='utf-8') as f:
            json.dump(self._tone_cache, f, ensure_ascii=False)

    def get_tone(self, char):
        """
        获取单个汉字的声调
        返回: '平' 或 '仄'
        """
        if not char or len(char) != 1:
            return '平'

        if char in self._tone_cache:
            return self._tone_cache[char]

        if not HAS_PYPINYIN:
            # 无pypinyin时的简化处理
            return '平'

        try:
            py_list = pinyin(char, style=Style.TONE3)
            if py_list and py_list[0]:
                tone_str = py_list[0][0]
                # 提取数字声调
                tone_num = None
                for ch in tone_str:
                    if ch.isdigit():
                        tone_num = int(ch)
                        break
                if tone_num is None:
                    tone = '平'
                elif tone_num in [1, 2]:
                    tone = '平'
                elif tone_num in [3, 4]:
                    tone = '仄'
                else:
                    tone = '平'
            else:
                tone = '平'
        except Exception:
            tone = '平'

        self._tone_cache[char] = tone
        return tone

    def get_tones(self, text):
        """获取字符串的平仄序列"""
        return [self.get_tone(c) for c in text if '一' <= c <= '鿿']

    def get_tone_pattern(self, text):
        """获取字符串的平仄模式字符串，如 '仄仄平平仄'"""
        tones = self.get_tones(text)
        return ''.join(tones)

    def check_compliance(self, line, template):
        """
        检查诗句是否符合平仄模板
        返回: (匹配数, 总字数, 符合率)
        """
        if len(line) != len(template):
            return 0, len(line), 0.0

        tones = self.get_tones(line)
        if len(tones) != len(template):
            return 0, len(line), 0.0

        matches = sum(1 for t, tmpl in zip(tones, template) if t == tmpl)
        return matches, len(line), matches / len(line)

    def score_tone_match(self, char, position_in_line, template):
        """
        给单个字在特定位置的平仄匹配打分

        Returns:
            1.0 完全匹配, 0.5 可接受, 0.0 不匹配
        """
        if template is None:
            return 0.5  # 无模板约束

        if position_in_line >= len(template):
            return 0.5

        expected = template[position_in_line]
        actual = self.get_tone(char)

        if expected == actual:
            return 1.0
        # 某些位置可以宽松（如第1、3、5字在七言中常有"一三五不论"）
        return 0.3  # 不匹配但可以接受


if __name__ == '__main__':
    td = ToneDict()
    # 测试
    tests = ['春', '风', '明', '月', '花', '落', '知', '多', '少']
    for c in tests:
        print(f'{c}: {td.get_tone(c)}')

    print(f"\n'春风明月花落知多少': {td.get_tone_pattern('春风明月花落知多少')}")

    # 测试模板匹配
    line = '春风又绿江南岸'
    template = '平平仄仄平平仄'
    matches, total, rate = td.check_compliance(line, template)
    print(f'"{line}" vs "{template}": {matches}/{total} = {rate:.1%}')
