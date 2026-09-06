# 本机Codex开发与GPU服务器实验工作流

## 方案结论

采用私有Git仓库作为代码同步的唯一通道：本机负责修改、检查、提交和推送；GPU服务器只执行快进更新和实验，不直接修改代码。模型权重、原始大数据、密钥和实验输出不进入Git。

当前200条正式MISSING/CONFLICT输入合计约2 MB，作为冻结的小型论文数据快照随仓库同步；其余可重建的中间调用记录已被 `.gitignore` 排除。

## 首次配置

### 1. 创建私有远程仓库

在GitHub、GitLab或Gitee创建一个空的私有仓库，不要在网页端初始化README。推荐使用SSH地址。

本机执行：

```powershell
git config --global user.name "你的名字"
git config --global user.email "你的邮箱"
./ops/configure_remote.ps1 -Url git@github.com:yuzhengzhou1-spec/refusal-research.git
./ops/push_code.ps1 -Message "chore: initialize reproducible experiment workspace"
```

不要把服务器密码、SSH私钥或API Key发给Codex，也不要写进YAML。API Key只放在服务器环境变量或未跟踪的 `.env` 文件中。

### 2. 服务器只读克隆

为服务器创建只读Deploy Key，然后执行：

```bash
git clone git@github.com:yuzhengzhou1-spec/refusal-research.git /home/zyz/evidence-posttrain
cd /home/zyz/evidence-posttrain
cp ops/server.local.env.example ops/server.local.env
cp evidence_posttrain_baselines/configs/experiment.server.example.yaml \
  evidence_posttrain_baselines/configs/experiment.server.yaml
```

修改两个未跟踪文件中的仓库、模型和输出路径。服务器不提交这两个文件。

### 3. 初始化服务器环境

```bash
bash ops/bootstrap_server.sh check
bash ops/bootstrap_server.sh install
```

脚本创建互相隔离的 `.venv-train` 和 `.venv-vllm`，避免vLLM对PyTorch版本的约束污染训练环境。安装前先根据服务器驱动确认 `TORCH_INDEX_URL`。

## 日常循环

本机：

```powershell
./ops/push_code.ps1 -Message "feat: describe the change"
```

服务器：

```bash
cd /home/zyz/evidence-posttrain
bash ops/update_code.sh main
bash ops/run_experiment.sh prepare
bash ops/run_experiment.sh preflight
```

启动vLLM和运行评测建议放在两个终端：

```bash
# 终端1
bash ops/run_experiment.sh serve

# 终端2
RUN_NAME=qwen25_zero_shot bash ops/run_experiment.sh eval
```

训练：

```bash
RUN_NAME=qwen25_sft bash ops/run_experiment.sh sft
RUN_NAME=qwen25_sft_train EVAL_SPLIT=train bash ops/run_experiment.sh eval
bash ops/run_experiment.sh dpo-build
RUN_NAME=qwen25_dpo bash ops/run_experiment.sh dpo
```

## 可复现性规则

1. 一次正式实验只对应一个已提交Git哈希；服务器工作区有改动时禁止更新和正式运行。
2. `run_with_manifest.py` 自动保存提交哈希、命令、配置副本及SHA256、GPU、Python和依赖列表。
3. 公共配置进入Git；服务器真实路径放在 `experiment.server.yaml`，密钥只放环境变量。
4. 模型目录只保存权重；训练结果写入独立实验盘，不写回仓库。
5. 不在服务器热修代码。发现问题回本机修改、测试、提交，再让服务器快进更新。
6. 正式论文结果记录确切模型revision或权重SHA256，不能只写 `main`。

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `.gitignore` | 排除密钥、权重、大型中间数据和输出，仅放行200条冻结输入 |
| `.gitattributes` | 固定Windows/Linux脚本行尾，避免服务器Shell脚本失效 |
| `ops/check_repository.py` | 提交前拦截大文件、权重和常见密钥 |
| `ops/configure_remote.ps1` | 本机设置私有Git远程地址 |
| `ops/push_code.ps1` | 本机暂存、安全检查、提交并推送 |
| `ops/bootstrap_server.sh` | 服务器检查GPU并创建训练/vLLM隔离环境 |
| `ops/update_code.sh` | 服务器无本地改动时执行fast-forward更新 |
| `ops/run_experiment.sh` | 服务器统一的prepare、eval、SFT和DPO入口 |
| `ops/run_with_manifest.py` | 包装正式实验并保存完整运行清单 |
| `ops/server.local.env.example` | 服务器本地路径模板 |
| `configs/experiment.server.example.yaml` | GPU服务器实验配置模板 |

## 不使用Git同步的内容

- 模型权重：直接在服务器下载到 `MODEL_ROOT`，或从已有模型盘软链接。
- 大规模原始数据：使用对象存储、NAS或一次性 `rsync`，再记录校验和。
- Checkpoint与日志：写到 `EXPERIMENT_ROOT`，需要备份时传对象存储。
- API Key和SSH私钥：只由服务器密钥管理或环境变量提供。
