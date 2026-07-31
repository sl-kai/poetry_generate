# -*- coding: utf-8 -*-
"""云GPU训练脚本（AutoDL专用，支持断点续训+定期存盘）"""
import sys, os, time
sys.path.insert(0, 'src')
from src.transformer_model import TransformerModel, TransformerConfig

corpus_path = 'corpus_all.txt'

config = TransformerConfig(
    vocab_size=8000,
    block_size=80,
    n_layer=4,
    n_head=4,
    n_embd=256,
    dropout=0.1,
    learning_rate=3e-4,
    max_epochs=2,          # 云GPU多跑一轮
    batch_size=512,
    device='cuda',
)

print(f'设备: {config.device}, batch={config.batch_size}, epochs={config.max_epochs}')
model = TransformerModel(config=config)
model.train(corpus_path=corpus_path)
print('\n云训练完成！下载 transformer_model.pt 回本地即可。')
