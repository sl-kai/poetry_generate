"""
Transformer 语言模型模块
基于 GPT 风格的 decoder-only Transformer，字符级古诗建模
替换原有的 N-gram 模型，提供更好的语义连贯性和生成多样性

架构: Mini-GPT (decoder-only Transformer)
- 字符级 tokenization (vocab_size ≈ 12000)
- 6 层 Transformer Decoder
- 384 维 embedding，6 个 attention heads
- 上下文窗口: 128 字符
- 支持 top-k / top-p / temperature 采样
"""

import os
import json
import math
import random
from collections import Counter

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("警告: PyTorch 未安装。请运行: pip install torch")


# ============================================================
# 模型配置
# ============================================================

class TransformerConfig:
    """Transformer 模型超参数"""
    def __init__(self,
                 vocab_size=12000,
                 block_size=128,        # 最大上下文长度
                 n_layer=6,             # Transformer 层数
                 n_head=6,              # 注意力头数
                 n_embd=384,            # Embedding 维度
                 dropout=0.1,
                 learning_rate=3e-4,
                 max_epochs=10,
                 batch_size=64,
                 device='cpu'):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.device = device


# ============================================================
# Transformer 组件
# ============================================================

class CausalSelfAttention(nn.Module):
    """带因果掩码的多头自注意力"""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # 因果掩码缓存
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C = x.size()  # batch, seq_len, embed_dim

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Transformer 中的前馈网络"""
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """单个 Transformer 层"""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MiniGPT(nn.Module):
    """
    小型 GPT 风格 Transformer，字符级语言模型
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        """前向传播，返回 logits"""
        B, T = idx.size()
        assert T <= self.block_size, f"序列长度 {T} 超过 block_size {self.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(pos)
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        """
        自回归生成

        Args:
            idx: 起始 token 序列 (1, T)
            max_new_tokens: 最多生成 token 数
            temperature: 温度 (>0)
            top_k: Top-K 采样
            top_p: Nucleus (Top-P) 采样

        Returns:
            生成的 token 序列
        """
        self.eval()
        for _ in range(max_new_tokens):
            # 截断到 block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]

            logits = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            # Top-K
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Top-P (nucleus)
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        self.train()
        return idx


# ============================================================
# 数据集
# ============================================================

class PoetryDataset(Dataset):
    """古诗数据集：将文本切分为固定长度的训练样本"""
    def __init__(self, data, block_size):
        self.block_size = block_size
        self.data = data  # 1D tensor of token ids

    def __len__(self):
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, i):
        x = self.data[i:i + self.block_size]
        y = self.data[i + 1:i + self.block_size + 1]
        return x, y


# ============================================================
# Transformer 模型包装器（与 NgramModel 接口兼容）
# ============================================================

class TransformerModel:
    """
    Transformer 诗歌模型

    与 NgramModel 保持兼容的接口，可以直接替换到 generator.py 中使用
    """

    def __init__(self, config=None, cache_dir=None):
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(__file__), '..', 'data', 'processed'
        )
        os.makedirs(self.cache_dir, exist_ok=True)

        self.config = config or TransformerConfig()
        self.model = None
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0

        # 特殊 token
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'

    def build_vocab(self, poems_or_text):
        """从诗歌数据构建词汇表"""
        chars = Counter()

        # 支持两种输入：诗歌列表 或 纯文本
        if isinstance(poems_or_text, str):
            text = poems_or_text
            for c in text:
                if '一' <= c <= '鿿':
                    chars[c] += 1
        elif isinstance(poems_or_text, list):
            for poem in poems_or_text:
                content = poem.get('content', '') if isinstance(poem, dict) else str(poem)
                for c in content:
                    if '一' <= c <= '鿿':
                        chars[c] += 1

        # 常见字 + 高频字
        vocab_chars = [self.PAD_TOKEN, self.UNK_TOKEN] + \
                      [c for c, _ in chars.most_common(self.config.vocab_size - 2)]

        self.char_to_idx = {c: i for i, c in enumerate(vocab_chars)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        self.vocab_size = len(vocab_chars)
        self.config.vocab_size = self.vocab_size

        print(f'词汇表构建完成: {self.vocab_size} 个 token')
        return self.vocab_size

    def encode(self, text):
        """字符串 → token id 列表"""
        return [self.char_to_idx.get(c, self.char_to_idx[self.UNK_TOKEN])
                for c in text if '一' <= c <= '鿿']

    def decode(self, ids):
        """token id 列表 → 字符串"""
        return ''.join(self.idx_to_char.get(i, self.UNK_TOKEN) for i in ids
                       if i not in (self.char_to_idx.get(self.PAD_TOKEN, -1),))

    def _pre_encode_corpus(self, corpus_path):
        """
        将语料文件预编码为 numpy 二进制文件（内存映射友好）
        返回编码后的 numpy 数组和总 token 数
        """
        import numpy as np
        bin_path = corpus_path + '.npy'

        # 如果已存在预编码文件，直接加载
        if os.path.exists(bin_path):
            print(f'加载预编码文件: {bin_path}')
            data = np.load(bin_path, mmap_mode='r')
            print(f'  {len(data)} tokens')
            return data

        print(f'预编码语料: {corpus_path}')
        print('  Step 1/3: 构建词汇查找表...')

        # 构建快速查找表：Unicode code point → token id
        # 中文字符范围在 0x4E00-0x9FFF
        lookup = np.full(0x10000, self.char_to_idx.get(self.UNK_TOKEN, 1), dtype=np.int32)
        for char, idx in self.char_to_idx.items():
            if len(char) == 1:
                code = ord(char)
                if code < 0x10000:
                    lookup[code] = idx

        print('  Step 2/3: 编码到临时文件...')
        tmp_path = bin_path + '.tmp'
        buffer_size = 10_000_000  # 10M tokens per write

        buffer = []
        total = 0
        with open(corpus_path, 'r', encoding='utf-8') as f_in, \
             open(tmp_path, 'wb') as f_out:

            while True:
                chunk = f_in.read(5_000_000)  # 5M chars per read
                if not chunk:
                    break

                codes = np.frombuffer(chunk.encode('utf-32-le'), dtype=np.uint32)
                ids = lookup[np.clip(codes, 0, 0xFFFF)]
                # Filter: keep only valid Chinese chars (token id != UNK or being common)
                valid_mask = ids != lookup[ord('\n')]  # also exclude newlines
                # Actually just keep all non-UNK, non-newline
                ids_filtered = ids[(ids > 1)]  # skip PAD(0) and UNK(1)

                buffer.append(ids_filtered)
                total += len(ids_filtered)

                if len(buffer) >= 5:
                    combined = np.concatenate(buffer)
                    combined.tofile(f_out)
                    buffer = []

            if buffer:
                combined = np.concatenate(buffer)
                combined.tofile(f_out)
                buffer = []

        print(f'  Step 3/3: 转换为 numpy 数组 ({total} tokens)...')

        # 从二进制文件创建 numpy 数组
        data = np.fromfile(tmp_path, dtype=np.int32)

        # 保存为 .npy 格式（支持 mmap）
        np.save(bin_path, data)
        os.remove(tmp_path)

        print(f'  预编码完成: {len(data)} tokens')
        return data

    def train(self, poems=None, corpus_path=None):
        """
        内存高效训练 Transformer 模型 (mmap + 预编码)

        Args:
            poems: 诗歌列表（可选）
            corpus_path: 纯文本语料路径
        """
        if not HAS_TORCH:
            print("ERROR: PyTorch 未安装")
            return

        import numpy as np

        device = torch.device(self.config.device if torch.cuda.is_available() else 'cpu')
        print(f'训练设备: {device}')

        if corpus_path is None:
            corpus_path = os.path.join(self.cache_dir, 'corpus_all.txt')
        if not os.path.exists(corpus_path):
            print(f"ERROR: 语料文件不存在: {corpus_path}")
            return

        # 构建词汇表（从文件头部采样）
        if not self.char_to_idx:
            print('从语料采样构建词汇表...')
            sample_size = 10_000_000
            with open(corpus_path, 'r', encoding='utf-8') as f:
                sample_text = f.read(sample_size)
            self.build_vocab(sample_text)

        # 预编码
        data = self._pre_encode_corpus(corpus_path)
        data = torch.from_numpy(data.astype(np.int64))

        # 训练/验证分割 (95/5)
        split_idx = int(len(data) * 0.95)
        train_data = data[:split_idx]
        print(f'训练集: {len(train_data)} tokens')

        # 数据集和 DataLoader
        dataset = PoetryDataset(train_data, self.config.block_size)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )

        # 模型
        self.model = MiniGPT(self.config).to(device)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f'模型参数量: {total_params / 1e6:.1f}M')
        print(f'每 epoch 步数: {len(dataloader)}')

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.max_epochs * len(dataloader)
        )

        best_loss = float('inf')

        for epoch in range(self.config.max_epochs):
            print(f'\n--- Epoch {epoch+1}/{self.config.max_epochs} ---')
            self.model.train()
            total_loss = 0
            batch_count = 0

            for x, y in dataloader:
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                logits = self.model(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    ignore_index=self.char_to_idx.get(self.PAD_TOKEN, 0)
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                batch_count += 1

                if batch_count % 500 == 0:
                    avg = total_loss / batch_count
                    lr = scheduler.get_last_lr()[0]
                    print(f'  Step {batch_count}/{len(dataloader)}, Loss: {avg:.4f}, LR: {lr:.6f}')

                # 每 5000 步存一次 checkpoint（防止断连白跑）
                if batch_count % 5000 == 0:
                    self.save('transformer_checkpoint')
                    print(f'  [Checkpoint] Step {batch_count} 已保存')

            avg_loss = total_loss / batch_count
            print(f'  Epoch {epoch+1} 完成, Avg Loss: {avg_loss:.4f}')

            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save('transformer_model')

        print(f'\n训练完成! Best loss: {best_loss:.4f}')
        self.save('transformer_model')

    def predict_next(self, prefix, n=20, temperature=0.8, top_k=50, top_p=0.9):
        """
        预测下一个字（与 NgramModel 接口兼容）

        Args:
            prefix: 上文（字符串）
            n: 候选数量
            temperature: 温度
            top_k: Top-K
            top_p: Top-P

        Returns:
            [(字, 概率), ...]
        """
        if self.model is None:
            return []

        device = next(self.model.parameters()).device
        self.model.eval()

        # 编码前缀
        prefix_chars = [c for c in prefix if '一' <= c <= '鿿']
        if not prefix_chars:
            # 随机选常见字
            return [(c, 1.0/n) for c in list(self.char_to_idx.keys())[:n]]

        prefix_ids = [self.char_to_idx.get(c, self.char_to_idx[self.UNK_TOKEN])
                      for c in prefix_chars]
        prefix_tensor = torch.tensor([prefix_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            logits = self.model(prefix_tensor)
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            # Top-K + Top-P
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cum_probs > top_p
                sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
                sorted_mask[:, 0] = False
                mask = sorted_mask.scatter(1, sorted_indices, sorted_mask)
                logits[mask] = float('-inf')

            probs = F.softmax(logits, dim=-1)

        # 获取 top-n
        top_probs, top_indices = torch.topk(probs[0], min(n, probs.size(-1)))
        candidates = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            char = self.idx_to_char.get(idx, self.UNK_TOKEN)
            candidates.append((char, prob))

        return candidates

    def generate_line(self, start_char=None, line_length=5, beam_width=15, temperature=0.8):
        """
        生成一行诗（与 NgramModel 接口兼容）

        Args:
            start_char: 起始字（用于藏头诗）
            line_length: 行长度
            beam_width: 在此接口中作为重复生成次数（Transformer 用随机采样代替 beam search）
            temperature: 温度

        Returns:
            [(诗句, 得分), ...]
        """
        if self.model is None:
            return []

        device = next(self.model.parameters()).device

        candidates = []
        for _ in range(beam_width):
            if start_char:
                start_ids = [self.char_to_idx.get(start_char, 0)]
                gen_ids = [start_ids[0]]
                # 从 start_char 开始生成
                for pos in range(line_length - 1):
                    idx_tensor = torch.tensor([gen_ids], dtype=torch.long, device=device)
                    with torch.no_grad():
                        logits = self.model(idx_tensor)[:, -1, :] / max(temperature, 1e-8)
                        probs = F.softmax(logits, dim=-1)
                        next_id = torch.multinomial(probs, num_samples=1).item()
                    gen_ids.append(next_id)
                line = self.decode(gen_ids)
            else:
                # 无起始字，从头生成
                start_tensor = torch.tensor([[0]], dtype=torch.long, device=device)
                gen_tensor = self.model.generate(
                    start_tensor, line_length,
                    temperature=temperature, top_k=50, top_p=0.9
                )
                line = self.decode(gen_tensor[0].tolist())[:line_length]

            if len(line) >= line_length:
                line = line[:line_length]
                # 简单得分：长度完整性
                score = len(line) / line_length
                candidates.append((line, score))

        # 去重并按得分排序
        seen = set()
        unique = []
        for line, score in candidates:
            if line not in seen and len(line) == line_length:
                seen.add(line)
                unique.append((line, score))

        unique.sort(key=lambda x: -x[1])
        return unique[:beam_width]

    def score_sequence(self, text):
        """
        计算序列的对数概率（流畅度，与 NgramModel 兼容）

        Returns:
            平均对数概率
        """
        if self.model is None or len(text) < 2:
            return float('-inf')

        device = next(self.model.parameters()).device
        ids = self.encode(text)
        if len(ids) < 2:
            return float('-inf')

        ids_tensor = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = self.model(ids_tensor)
            log_probs = F.log_softmax(logits, dim=-1)

        total_log_prob = 0.0
        for i in range(len(ids) - 1):
            total_log_prob += log_probs[0, i, ids[i + 1]].item()

        return total_log_prob / (len(ids) - 1)

    def save(self, name='transformer_model'):
        """保存模型"""
        if self.model is None:
            print('无模型可保存')
            return

        model_path = os.path.join(self.cache_dir, f'{name}.pt')
        vocab_path = os.path.join(self.cache_dir, f'{name}_vocab.json')
        config_path = os.path.join(self.cache_dir, f'{name}_config.json')

        # 保存权重
        torch.save(self.model.state_dict(), model_path)

        # 保存词汇表
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump({
                'char_to_idx': self.char_to_idx,
                'idx_to_char': {str(k): v for k, v in self.idx_to_char.items()},
                'vocab_size': self.vocab_size,
            }, f, ensure_ascii=False)

        # 保存配置
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump({
                'vocab_size': self.config.vocab_size,
                'block_size': self.config.block_size,
                'n_layer': self.config.n_layer,
                'n_head': self.config.n_head,
                'n_embd': self.config.n_embd,
                'dropout': self.config.dropout,
            }, f)

        print(f'模型已保存: {model_path}')

    def load(self, name='transformer_model'):
        """加载模型"""
        if not HAS_TORCH:
            print("ERROR: PyTorch 未安装")
            return False

        model_path = os.path.join(self.cache_dir, f'{name}.pt')
        vocab_path = os.path.join(self.cache_dir, f'{name}_vocab.json')
        config_path = os.path.join(self.cache_dir, f'{name}_config.json')

        if not os.path.exists(model_path):
            print(f'未找到 Transformer 模型: {model_path}')
            return False

        # 加载配置
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            self.config = TransformerConfig(**cfg)

        # 加载词汇表
        if os.path.exists(vocab_path):
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vcb = json.load(f)
            self.char_to_idx = vcb['char_to_idx']
            self.idx_to_char = {int(k): v for k, v in vcb['idx_to_char'].items()}
            self.vocab_size = vcb['vocab_size']
            self.config.vocab_size = self.vocab_size

        # 加载模型
        self.model = MiniGPT(self.config)
        device = torch.device(self.config.device if torch.cuda.is_available() else 'cpu')
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

        params = sum(p.numel() for p in self.model.parameters())
        print(f'Transformer 模型已加载: {params/1e6:.1f}M 参数, {self.vocab_size} 词汇')
        return True


if __name__ == '__main__':
    print('Transformer 模型模块')
    print(f'PyTorch 可用: {HAS_TORCH}')

    if HAS_TORCH:
        config = TransformerConfig(
            vocab_size=12000,
            block_size=128,
            n_layer=6,
            n_head=6,
            n_embd=384,
        )
        model = MiniGPT(config)
        params = sum(p.numel() for p in model.parameters())
        print(f'MiniGPT 参数量: {params/1e6:.1f}M')

        # 测试前向传播
        x = torch.randint(0, 1000, (2, 32))
        logits = model(x)
        print(f'输入 shape: {x.shape}, 输出 shape: {logits.shape}')
