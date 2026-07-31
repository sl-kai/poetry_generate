# GitHub 公开发布设计

## 目标

将古诗生成系统发布为 GitHub 公开仓库 `poetry_generate`。仓库应包含完整源码、两张界面截图和直接运行推理所需的模型文件，同时避免提交体积过大的训练语料与生成数据。

## 仓库内容

保留以下内容：

- Python 源码与依赖声明
- `data/processed/transformer_model.pt`
- `data/processed/transformer_model_config.json`
- `data/processed/transformer_model_vocab.json`
- `data/processed/statistics.json`
- `data/processed/rhyme_cache.json`
- 两张 Web 界面截图
- 中文 README

排除以下内容：

- `raw_poetry/` 原始训练数据
- `data/processed/all_poems.jsonl`
- `data/processed/corpus_all.txt`
- Python 缓存、虚拟环境、编辑器配置、日志和本地环境变量文件

## 截图与文档

截图放在 `assets/screenshots/`，分别命名为 `keyword-generation.png` 和 `acrostic-generation.png`。README 在项目介绍之后展示两张截图，并说明仓库已包含推理模型，但重新训练需要用户自行准备原始数据和语料。

## GitHub 发布

本地仓库使用 `main` 作为默认分支，远程公开仓库名称为 `poetry_generate`。首次提交发布前检查受 Git 跟踪的文件大小、忽略规则、潜在凭据以及 README 图片路径。当前不添加许可证，避免未经用户选择授予代码使用权限。

## 验收标准

- Git 不跟踪任何超过 100 MB 的文件。
- 两张截图能够在 GitHub README 中直接显示。
- 模型权重、词表、配置和生成评分所需统计文件均被跟踪。
- 原始训练数据和大型中间文件均被忽略。
- GitHub 仓库为公开仓库，默认分支为 `main`。
