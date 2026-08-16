# 贡献指南

[English](CONTRIBUTING.md)

贡献的首要问题是：它是否让学习者更容易运行、观察或理解这个项目，同时保持实验结果
可以核对？技术上正确但难以解释的功能，可能需要先缩小范围。

## 开始编码前

行为或架构变化先开 issue，说明学习者遇到的问题、预期结果、替代方案、数据/网络影响
和验证方式。不要把 Oxford 原始数据、下载的模型权重、密钥或完整本地产物加入 Git。

## 本地检查

```bash
uv sync --extra dev
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
uv build
uv run twine check dist/*
```

新增或修复行为时先写失败测试。测试和 CI 必须离线，使用合成 fixture 和
`weights=None`。不要绕过 pre-commit hooks。

## 文档

中英文核心文档都要有入口并描述相同行为，但可以按各自语言组织标题、例子和排错内容。
涉及命令、配置名、标签值或指标定义时必须一致。新增字段要更新配置参考，教学示例要指向
正式模块，实测数字要同时记录条件和限制。

模型二进制放 GitHub Release 或模型制品存储，仓库只保留配置、哈希、指标和少量代表图。

## Pull Request

说明改了什么、解决哪个学习问题、运行了哪些检查、产物怎样变化，以及仍有哪些限制。
无关重构拆成其他 PR。提交使用英文 Conventional Commits。

提交贡献即表示同意作品按仓库的 [MIT License](LICENSE) 发布。
