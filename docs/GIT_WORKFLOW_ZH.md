# VLA-PCRV1 本地版本管理

本地目录：`/personal_fast_storage/VLA-PCRV1`。主要开发项目位于
`versions/v1_evolution/`，使用方式见该目录的 `README.md`。
保留完整仓库结构，以支持共享工具和跨版本实验脚本。

## 已完成的配置

- 保留源仓库完整提交历史。
- `upstream`：`git@github.com:cgznb/breast-mri-jepa-versions.git`。
- `main`：跟踪 `upstream/main`。
- `work/vla-pcrv1`：当前开发分支，未设置远端跟踪分支。
- `vla-pcrv1-baseline`：原始源码基线标记，对应提交 `6f69744`。
- 提交身份仅在本仓库生效：`cgznb` / `74957389+cgznb@users.noreply.github.com`。
- 本仓库已配置专用 SSH 密钥，后续拉取无需额外指定密钥。
- 拉取使用 `pull.ff=only`，需要合并的分支由开发者显式处理。

## 日常保存版本

以下命令从仓库根目录执行：

```bash
cd /personal_fast_storage/VLA-PCRV1
git status
git diff
# 将下面路径替换为本次修改的具体文件，可列出多个文件。
git add versions/v1_evolution/实际修改的文件
git diff --cached
git commit -m "说明本次修改的内容"
git log --oneline --decorate -10
```

每个提交保存一个可以说明清楚的修改。现有 `.gitignore` 已排除 MRI 数据、
模型权重、虚拟环境和常见运行产物；本地配置放入 `local_configs/`，
训练输出放入 `private_outputs/`。

查看相对原始源码的修改：

```bash
git diff vla-pcrv1-baseline -- versions/v1_evolution/
```

查看原始版本而不切换当前工作分支：

```bash
git show vla-pcrv1-baseline:versions/v1_evolution/README.md
```

查看源仓库更新：

```bash
git fetch upstream
git log --oneline HEAD..upstream/main
```

确认需要引入源仓库更新后，在工作区干净时执行 `git merge upstream/main`。

## 可选：连接独立的 GitHub 备份仓库

本地提交已经可用。当前没有配置独立的 `origin` 远端，也没有推送本地修改。
需要云端备份时，在有权限的 GitHub 账号下创建一个空的私有仓库
`VLA-PCRV1`，创建时不要初始化 README、许可证或 `.gitignore`。

拿到新仓库的 SSH 地址后，将下方账号替换为实际账号，再执行：

```bash
git remote add origin git@github.com:你的账号/VLA-PCRV1.git
git push -u origin work/vla-pcrv1
git push origin vla-pcrv1-baseline
```

连接后，在 `work/vla-pcrv1` 分支运行 `git push` 即可同步到新仓库。
`upstream` 继续用于获取源仓库更新。

Git 备份包含受版本管理的代码和文档；被忽略的数据、权重和训练输出需要单独备份。
