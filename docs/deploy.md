# 部署与同步（deploy）

本地仓库是**唯一真源**，服务器只接收部署产物，不允许长期手工改代码。

## 三个保险

| # | 保险 | 实现位置 | 行为 |
|---|---|---|---|
| 1 | 本地工作区必须干净 | `runs/scripts/deploy.ps1` | 有未提交改动 → 拒绝打包，列出改动文件 |
| 2 | 上传后逐文件校验 | `runs/scripts/deploy.sh`（服务器） | 任一 hash 不一致 → 列出差异文件并以 **exit 1** 中止，不写 `DEPLOYED_COMMIT` |
| 3 | 记录部署版本 | `runs/scripts/deploy.sh` | 校验通过后写 `~/Reca-Audio/DEPLOYED_COMMIT`（commit / 时间 / manifest hash / 文件数） |

## 用法

**半自动（当前可用路径）**

```powershell
cd D:\Homework\2026Fall\paperLearn\Reca-Audio
powershell -NoProfile -ExecutionPolicy Bypass -File .\runs\scripts\deploy.ps1
# 输出三条命令：上传 tar / 上传 manifest / 远端 deploy.sh
```

**全自动（本机 `ssh`/`scp` 可用且免密时）**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\runs\scripts\deploy.ps1 -Push
```

同步范围：`src/`、`Audio-Omni/audio_omni/`、`runs/scripts/`、`runs/Benchmark/`（只取 `git ls-files`，
数据与输出目录已被 `.gitignore` 排除）。

## 临时调试（服务器上直接改代码）的处理

规则：**改完立刻回灌本地，git diff → commit**，不要等下次部署。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\runs\scripts\pull_from_server.ps1 `
  -Files Audio-Omni/audio_omni/api.py -ShowDiff
```

反例（真实事故）：曾在服务器用 python 改 `api.py`，又在本地用 PowerShell 补一次，结果本地
UTF-8 中文注释被按 ANSI 读写损坏（18296 → 19107 字节），本地/远端 hash 不一致。正确做法是
改完服务器直接 `pull_from_server.ps1` 拉回，而不是再手工改一遍。

## 已验证记录（2026-11-05）

| 测试 | 结果 |
|---|---|
| 正常部署 | 480 个文件校验通过，写入 `DEPLOYED_COMMIT=c4edf66` |
| manifest 篡改一个 hash | `❌ 校验失败`，列出 `Audio-Omni/audio_omni/__init__.py`，exit 1 |
| 本地有未提交改动 | 拒绝部署并列出 7 个改动文件 |

## 注意事项

1. **首次引导**：`deploy.sh` 自身必须先在服务器上存在（第一次用 `ssh_upload` 单独传），
   之后随 tar 一起部署。
2. **PowerShell 5.1**：脚本必须保持 **纯 ASCII**（无 BOM 的 UTF-8 会被按 ANSI 解析导致语法错误）；
   运行需要 `-ExecutionPolicy Bypass`（脚本未签名）。
3. **`-AllowDirty`** 只用于演练，会绕过保险 1；此时 `DEPLOYED_COMMIT` 记录的 commit 并不包含
   实际部署的代码。
4. `DEPLOYED_COMMIT` 目前只是**记录**，不会阻止训练/评测启动；如需强约束，可在启动脚本里
   比对 `git rev-parse` 与 `DEPLOYED_COMMIT`。
5. 本机 `~/.ssh` 受安全策略保护，脚本不读取其中的配置；`-Push` 依赖系统 `ssh/scp` 的默认配置。
