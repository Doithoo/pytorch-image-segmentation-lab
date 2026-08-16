# 贡献指南

[English](CONTRIBUTING.md)

贡献应保持项目对初学者可读的行为、可复现契约以及英文/简体中文内容对齐。

## 编码前

行为或架构变化请先开 issue。说明学习者问题、预期行为、替代方案、数据/网络影响和
测试方式。绝不能把 Oxford 数据、下载的模型权重或密钥加入 Git。

## 本地流程

```bash
uv sync --extra dev
uv run ruff check .
uv run pytest
uv build
uv run twine check dist/*
```

修改生产行为前先写失败测试。保持提交聚焦，并使用英文 Conventional Commits。
不要绕过 hooks。测试和 CI 必须保持离线：使用合成 fixture 和 `weights=None`。

## 文档

英文和 `.zh-CN.md` 对应文件必须同步更新，标题层级和 shell 命令块完全一致。新增
字段要写入配置参考，学习示例要链接到生产模块。绝不能发布未实测指标，也不能省略
解释指标所需的条件。

## Pull Request

说明改了什么、为何帮助学习者、运行了哪些测试、产物或界面变化以及已知限制。无关
重构应拆开。即使代码技术上正确，维护者也可能要求更小的示例或更清楚的命名。

提交贡献即表示你同意作品按仓库的 [MIT License](LICENSE)发布。
