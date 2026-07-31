"""
古诗生成系统 —— 主程序入口
支持两种模式：
1. 关键词生成: python main.py keyword --words 春风 明月 --type 五言绝句
2. 藏头诗生成: python main.py acrostic --head 生日快乐 --type 七言绝句
3. 交互模式: python main.py interactive
4. 训练模式: python main.py train
"""

import os
import sys
import argparse

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.tone_dict import ToneDict
from src.analyzer import PoetryAnalyzer
from src.generator import PoemGenerator, format_poem_output


def cmd_train(args):
    """预处理模式：构建统计数据（Transformer 训练请使用 cloud_train.py 上传云端）"""
    from src.train_pipeline import train_all

    raw_dir = args.raw_dir or os.path.join(os.path.dirname(__file__), 'raw_poetry')
    processed_dir = args.processed_dir or os.path.join(os.path.dirname(__file__), 'data', 'processed')

    print(f'\n原始数据目录: {raw_dir}')
    print(f'处理后数据目录: {processed_dir}')
    print('注意: 此命令仅构建统计数据。Transformer 模型训练请使用 cloud_train.py 上传 AutoDL 云端训练。')
    train_all(raw_dir, processed_dir)


def cmd_keyword(args):
    """关键词生成模式"""
    # 加载模型
    generator = _load_generator()

    keywords = args.words
    poem_type = args.type or '五言绝句'

    print(f'\n关键词: {" ".join(keywords)}')
    print(f'诗体: {poem_type}')

    # 生成多首候选
    for i in range(args.count):
        print(f'\n--- 候选 {i+1} ---')
        result = generator.generate_from_keywords(
            keywords=keywords,
            poem_type=poem_type,
            beam_width=args.beam,
            temperature=args.temperature,
        )
        format_poem_output(result, mode='keyword')


def cmd_acrostic(args):
    """藏头诗生成模式"""
    generator = _load_generator()

    head = args.head
    poem_type = args.type  # None = 自动选择

    if poem_type:
        print(f'\n藏头字: {head}')
        print(f'诗体: {poem_type}')
    else:
        print(f'\n藏头字: {head} (自动选择诗体)')

    for i in range(args.count):
        print(f'\n--- 候选 {i+1} ---')
        result = generator.generate_acrostic(
            head_chars=head,
            poem_type=poem_type,
            beam_width=args.beam,
            temperature=args.temperature,
        )
        format_poem_output(result, mode='acrostic')


def cmd_interactive(args):
    """交互模式"""
    print('=' * 60)
    print('  古诗生成系统 —— 交互模式')
    print('  输入 "quit" 或 "q" 退出')
    print('=' * 60)

    # 预加载模型
    print('正在加载模型...')
    generator = _load_generator(verbose=True)
    print('模型加载完成!\n')

    while True:
        print('-' * 40)
        print('请选择模式:')
        print('  1. 关键词生成古诗')
        print('  2. 藏头诗生成')
        print('  q. 退出')
        choice = input('> ').strip()

        if choice.lower() in ('q', 'quit', 'exit'):
            print('再见!')
            break

        if choice == '1':
            words_input = input('请输入关键词（空格分隔）: ').strip()
            if not words_input:
                print('关键词不能为空')
                continue
            keywords = words_input.split()

            print('请选择诗体:')
            print('  1. 五言绝句  2. 五言律诗  3. 七言绝句  4. 七言律诗')
            type_choice = input('> ').strip()
            type_map = {'1': '五言绝句', '2': '五言律诗', '3': '七言绝句', '4': '七言律诗'}
            poem_type = type_map.get(type_choice, '五言绝句')

            result = generator.generate_from_keywords(
                keywords=keywords,
                poem_type=poem_type,
                beam_width=15,
                temperature=0.8,
            )
            format_poem_output(result, mode='keyword')

        elif choice == '2':
            head = input('请输入藏头字（如"生日快乐"）: ').strip()
            if not head:
                print('藏头字不能为空')
                continue

            result = generator.generate_acrostic(
                head_chars=head,
                beam_width=15,
                temperature=0.8,
            )
            format_poem_output(result, mode='acrostic')

        else:
            print('无效选择，请重试')


def _load_generator(verbose=False):
    """加载生成器（Transformer 模型）"""
    tone_dict = ToneDict()
    analyzer = PoetryAnalyzer()
    if not analyzer.load():
        if verbose:
            print('  警告: 未找到统计数据')

    from src.transformer_model import TransformerModel
    transformer_model = TransformerModel()
    if transformer_model.load('transformer_model'):
        if verbose:
            params = sum(p.numel() for p in transformer_model.model.parameters())
            print(f'  Transformer 模型已加载 ({params/1e6:.1f}M 参数, {transformer_model.vocab_size} 词汇)')
    else:
        if verbose:
            print('  警告: 未找到 Transformer 模型，请先在云端训练: python cloud_train.py')

    return PoemGenerator(
        transformer_model=transformer_model,
        tone_dict=tone_dict,
        analyzer=analyzer,
    )


def main():
    parser = argparse.ArgumentParser(
        description='古诗生成系统 - 基于N-gram的古诗自动生成',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python main.py train
  python main.py keyword --words 春风 明月 --type 五言绝句
  python main.py acrostic --head 生日快乐
  python main.py interactive
        '''
    )

    subparsers = parser.add_subparsers(dest='command', help='子命令')

    # train 子命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--raw-dir', help='原始CSV数据目录')
    train_parser.add_argument('--processed-dir', help='处理后数据输出目录')

    # keyword 子命令
    kw_parser = subparsers.add_parser('keyword', help='关键词生成古诗')
    kw_parser.add_argument('--words', nargs='+', required=True, help='关键词（空格分隔）')
    kw_parser.add_argument('--type', default='五言绝句',
                           choices=['五言绝句', '五言律诗', '七言绝句', '七言律诗'],
                           help='诗体类型')
    kw_parser.add_argument('--count', type=int, default=1, help='生成候选数量')
    kw_parser.add_argument('--beam', type=int, default=15, help='Beam Search 宽度')
    kw_parser.add_argument('--temperature', type=float, default=0.8, help='采样温度')

    # acrostic 子命令
    ac_parser = subparsers.add_parser('acrostic', help='生成藏头诗')
    ac_parser.add_argument('--head', required=True, help='藏头字')
    ac_parser.add_argument('--type', default=None, help='诗体（默认自动选择）')
    ac_parser.add_argument('--count', type=int, default=1, help='生成候选数量')
    ac_parser.add_argument('--beam', type=int, default=15, help='Beam Search 宽度')
    ac_parser.add_argument('--temperature', type=float, default=0.8, help='采样温度')

    # interactive 子命令
    subparsers.add_parser('interactive', help='交互模式')

    args = parser.parse_args()

    if args.command == 'train':
        cmd_train(args)
    elif args.command == 'keyword':
        cmd_keyword(args)
    elif args.command == 'acrostic':
        cmd_acrostic(args)
    elif args.command == 'interactive':
        cmd_interactive(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
