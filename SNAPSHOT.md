# 完整项目快照

仓库文件页提供源代码、配置、Notebook、测试和复现入口。数据快照、实验逐步记录、完整成果包及原始目录中的其他文件随 [GitHub Releases](https://github.com/BabyofStar/EV_Rolling_Horizon_Charging_Optimization/releases) 中的分卷提供。完整复现应从还原后的项目目录运行。

## 校验与解包

下载同一 Release 的 `SHA256SUMS` 和全部 `EV_project.tar.gz.part-*` 分卷，在 Linux、macOS 或 WSL 中将它们放在同一目录，再执行：

```bash
set -euo pipefail
# Linux / WSL
sha256sum -c SHA256SUMS
# macOS 可使用 shasum -a 256 -c SHA256SUMS
cat EV_project.tar.gz.part-* | tar -xzf -
```

请使用新的目录解包。解包后阅读项目的 `README.md`。历史虚拟环境用于存档，复现时按 README 使用固定依赖创建环境。
