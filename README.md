# DIBO 使用与运行指南

DIBO 是面向 vLLM 的自动调参研究原型。它把 15 个底层参数组织为 8 个可解释的 Action，用六个 `Action 系数 -> Metric` 高斯过程 F 和一个 `Metrics -> TPS` 高斯过程 G 逐轮选择新配置。

设计与 CPU 阶段验收见 [工程计划](DIBO_v10_env机开发与CPU测试工程计划.md) 和 [验收记录](ACCEPTANCE.md)。

## 1. 最重要的运行结论

### 1.1 能否自动把 BO 参数应用到 vLLM？

可以。real 模式下每个候选自动执行：

```text
BO 产生 z
-> Compiler 用当前 W 和固定 x_base 生成 15 个最终参数
-> 保存 spec/CompileTrace
-> 启动新的 vllm serve
-> 检查 /health 和 /v1/models
-> warmup
-> 启动 Prometheus + NVML 后台采样
-> 发送正式 HTTP workload
-> 聚合 m01-m06 和 TPS
-> 停止采样器
-> finally 停止本次 vLLM 进程组
-> 保存结果并更新 F/G
-> 选择下一候选
```

使用者不需要手工停止旧服务、改参数、再启动。每个 Trial 只操作自己创建的进程组，不扫描或终止其他用户进程。

### 1.2 哪些模型需要单独启动？

1. Qwen 被调模型由每个 Trial 自动加载，不要在完整 `dibo tune` 前手动启动。
2. F/G 是 DIBO 进程内的 scikit-learn 对象，不需要独立服务。
3. 更新 W 的外部 LLM 是可选服务。完整 BO+LLM 模式需要 env 机的文件桥 worker；只测试 BO 自动重启时使用 `--no-llm`。

### 1.3 如何看到每次优化？

```bash
tail -f runs/<experiment_id>/events.jsonl
```

每个 `trial_completed` 事件包含 Trial 状态、Action 版本、所选 Metric/Action、配置 Hash、六项 Metric、TPS 和清理结果。完整文件见第 11 节。

运行时还会在终端逐步输出并写入相同的 JSONL 文件：`trial_started`、`engine_starting`、`engine_ready`、`warmup_completed`、`metrics_sampling_started`、`benchmark_completed`、`metrics_aggregated`、`engine_stopped`、`trial_result_saved`、`model_fit_completed`、`round_selected` 和 `llm_review`。日志只记录 ID、状态、指标和配置 Hash，不记录 API key 或请求正文。

文件桥等待外部 LLM 时，会额外记录 `llm_request_published`、每 30 秒一次的 `llm_waiting` 心跳、`llm_response_received`、`llm_wait_timeout` 或 `llm_wait_aborted`。这使得“worker 没响应”不会是静默等待。

### 1.4 如何提前安全停止？

在另一个终端执行：

```bash
RUN=runs/<experiment_id>
printf 'operator requested stop\n' > "$RUN/STOP"
```

Controller 会在当前 Trial 完成清理后，或文件桥等待的下一次轮询时停止后续工作，并写 `stop_requested`/`llm_wait_aborted` 事件和 `summary.json`。它不会杀死其他人的进程；当前 Trial 的 vLLM 仍由其 `finally` 清理。

## 2. 服务器边界

| 入口 | 用途 |
|---|---|
| `h20-env-qb-lpl` | 联网下载、编辑、安装依赖、运行 LLM worker |
| `h20-gpu-qb-lpl` | 在管理员分配的 GPU 上运行 preflight、vLLM、metrics smoke 和 real tune |

共享项目目录：`/home/nice/qinbin/stu/lpl/code/dibo`。

共享模型目录：`/home/nice/qinbin/stu/lpl/models`。

必须遵守：

- 不要从 env 机内部再 `ssh h20`。
- 不能自行选择“任意空闲 GPU”，必须取得管理员分配的 UUID、卡数和时段。
- API key 只在 env worker 进程环境中设置，不写入文件。
- 手动 vLLM 与 metrics smoke/full tune 不要同时占用同一 GPU 或端口。


## 3. 代码框架

### 3.1 配置

| 文件 | 作用 |
|---|---|
| `configs/parameters.yaml` | 15 个参数的类型、范围、初始符号值和物理尺度 |
| `configs/actions_v1.yaml` | 初始 8x15 矩阵 W、Action 语义和锚点 |
| `configs/graph.yaml` | 8 Action 到 6 Metric 的 24 条边 |
| `configs/experiment_smoke.yaml` | CPU synthetic 配置，只允许 fake adapter |
| `configs/experiment.yaml` | real 模板，故意保留 GPU、revision、workload Hash 和 m04 占位符 |

### 3.2 核心模块

| 文件 | 作用 |
|---|---|
| `schemas.py` | 严格 Schema，拒绝未知字段、错维度、错 adapter 和 real 占位符 |
| `compiler.py` | 把 x_base、W、z 编译为最终 15 参数，负责 HALF_UP、网格、投票、约束和 Hash |
| `trace_encoder.py` | 生成相对 x_base 的 phi，只用于证据、复现和 LLM，不进入 F |
| `engine.py` | 生成完整 argv，显式传递 bool 正/负 flag，设置 UUID 与 V1/V0，启动、检查和停止 vLLM |
| `benchmark.py` | 读取 JSONL，按速率/并发发送流式 chat 请求，测 TTFT 和实际 token |
| `metrics.py` | 后台抓取 `/metrics` 和指定 UUID 的 NVML，写原始样本并聚合 m01-m06 |
| `trial.py` | 管理一个 Trial 的保存、启动、warmup、采样、benchmark、清理和单次结果写入 |
| `store.py` | 原子保存 JSON，禁止覆盖已有 Trial |
| `initial_design.py` | 生成全零、八个 `+0.8` probe、Sobol，按最终 Hash 去重，选一次 x_base |
| `models.py` | 拟合六个邻接 z F 和一个六维 G，旧 W 样本用 2 倍噪声 |
| `selector.py` | 硬阈值优先；否则 G 反事实；最后不确定性/轮转后备 |
| `optimizer.py` | 在完整邻接并集中做 MC-EI，真实编译去重，G 只作并列次级排序 |
| `llm_update.py` | 构造全历史 Evidence、校验 W 修改、保存版本并实现文件桥 |
| `controller.py` | 串联完整闭环，按 run mode 选择 FakeTrialRunner 或 RealTrialRunner |
| `report.py` | 从磁盘离线重建 Markdown/CSV，不启动引擎 |
| `cli.py` | `dibo` 命令入口 |

### 3.3 脚本

| 脚本 | 作用 |
|---|---|
| `download_model.py` | 下载固定 revision 模型并记录来源 |
| `prepare_workload.py` | 生成 chat JSONL 和 SHA256 |
| `prepare_real_config.py` | 合并模板、模型、workload、UUID、V1/V0 和 m04 值 |
| `gpu_preflight.py` | 在获配 GPU 节点检查 UUID、Torch CUDA、vLLM、模型和 workload |
| `print_vllm_command.py` | 打印零 Action 对应的完整 vLLM 命令 |
| `gpu_metrics_smoke.py` | 自动跑一个真实 Trial 并打印 metrics 输出路径 |
| `run_real_tuning.sh` | preflight 后运行完整 real tune |
| `llm_worker_watch.py` | 持续处理一个 run 的 LLM 文件请求 |

## 4. 环境

```bash
source /home/nice/qinbin/stu/lpl/tools/activate.sh
conda activate dibo
cd /home/nice/qinbin/stu/lpl/code/dibo
python --version
python -m pip check
dibo --version
```

重装完整依赖：

```bash
python -m pip install -e '.[test,gpu]'
```

CPU 回归：

```bash
python -m ruff check src tests scripts
python -m pytest -q tests/unit
python -m pytest -q tests/interface
python -m pytest -q tests/e2e/test_cpu_closed_loop.py
```

CPU 测试使用 fake subprocess/HTTP/NVML，并禁止真实 socket、子进程和 GPU 信号。

## 5. 下载 Qwen2.5-7B-Instruct

在联网 env 机运行：

```bash
ssh h20-env-qb-lpl
source /home/nice/qinbin/stu/lpl/tools/activate.sh
conda activate dibo
cd /home/nice/qinbin/stu/lpl/code/dibo

python scripts/download_model.py \
	--repo Qwen/Qwen2.5-7B-Instruct \
	--output /home/nice/qinbin/stu/lpl/models/Qwen2.5-7B-Instruct
```

下载器固定 commit，不覆盖非空目录，并写 `DIBO_MODEL_INFO.json`。

本次官方 Hugging Face TLS 连接被对端重置，随后经用户授权通过镜像完成下载：

```bash
python scripts/download_model.py \
	--endpoint https://hf-mirror.com \
	--repo Qwen/Qwen2.5-7B-Instruct \
	--output /home/nice/qinbin/stu/lpl/models/Qwen2.5-7B-Instruct
```

已下载并离线校验：

```text
repo: Qwen/Qwen2.5-7B-Instruct
revision: a09a35458c702b33eeacc393d103063234e8bc28
endpoint: https://hf-mirror.com
size: 15.24 GB
weights: 4 个 safetensors 分片，索引无缺失
model: qwen2，28 heads，28 layers，hidden size 3584
tokenizer: Qwen2TokenizerFast，本地 chat template 可用
```

下载完成后：

```bash
du -sh /home/nice/qinbin/stu/lpl/models/Qwen2.5-7B-Instruct
test -f /home/nice/qinbin/stu/lpl/models/Qwen2.5-7B-Instruct/config.json
cat /home/nice/qinbin/stu/lpl/models/Qwen2.5-7B-Instruct/DIBO_MODEL_INFO.json
```

## 6. Workload

已提供：`data/requests/qwen_smoke.jsonl`。每行都是：

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."}]}
```

正式 `num_prompts=128` 时会稳定循环使用文件行。生成新文件：

```bash
python scripts/prepare_workload.py data/requests/qwen_mixed_128.jsonl --rows 128
```

脚本打印绝对路径、行数和 SHA256，已有文件拒绝覆盖。

## 7. 生成 real 配置

先取得管理员分配的 GPU UUID、卡数和时段。不能以“任意一个 GPU”代替分配。

Metrics smoke 配置，以下 m04 仅为链路测试值：

```bash
python scripts/prepare_real_config.py \
	--output configs/qwen25_7b_metrics_smoke.yaml \
	--experiment-id qwen25_7b_metrics_smoke \
	--gpu-uuid GPU-替换为管理员分配的UUID \
	--model-dir /home/nice/qinbin/stu/lpl/models/Qwen2.5-7B-Instruct \
	--request-file data/requests/qwen_smoke.jsonl \
	--execution-mode V1 \
	--llm-model disabled \
	--m04-trigger 8 --m04-target 4 --m04-scale 2 \
	--max-total-trials 1
```

正式调参配置必须使用 pilot 确认的 m04：

```bash
python scripts/prepare_real_config.py \
	--output configs/qwen25_7b_real.yaml \
	--experiment-id qwen25_7b_real_run01 \
	--gpu-uuid GPU-替换为管理员分配的UUID \
	--model-dir /home/nice/qinbin/stu/lpl/models/Qwen2.5-7B-Instruct \
	--request-file data/requests/qwen_smoke.jsonl \
	--execution-mode V1 \
	--llm-model 你的审核模型名 \
	--m04-trigger PILOT_TRIGGER \
	--m04-target PILOT_TARGET \
	--m04-scale PILOT_SCALE
```

脚本自动写入模型 revision、绝对路径、workload SHA256、卡数和 UUID。

## 8. GPU 预检

以下命令只能在管理员分配的 GPU 入口和时段运行：

```bash
ssh h20-gpu-qb-lpl
source /home/nice/qinbin/stu/lpl/tools/activate.sh
conda activate dibo
cd /home/nice/qinbin/stu/lpl/code/dibo

python scripts/gpu_preflight.py \
	--config configs/qwen25_7b_metrics_smoke.yaml \
	--output runs/qwen25_7b_gpu_preflight.json
```

它检查配置 UUID 是否可见、Torch CUDA、vLLM 可执行文件、模型、tokenizer、workload Hash 和模型结构，但不启动模型。

## 9. 直接部署一次 vLLM

打印完整命令：

```bash
python scripts/print_vllm_command.py \
	--config configs/qwen25_7b_metrics_smoke.yaml \
	> /tmp/start_dibo_qwen.sh
cat /tmp/start_dibo_qwen.sh
```

文件包含：

```bash
export CUDA_VISIBLE_DEVICES=GPU-...
export VLLM_USE_V1=1
vllm serve ...全部15个参数及显式bool正/负flag...
```

前台启动：

```bash
bash /tmp/start_dibo_qwen.sh
```

另一个 GPU 终端检查：

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/v1/models | python -m json.tool
curl -s http://127.0.0.1:8000/metrics | head
```

测试后 `Ctrl+C`。不要让手动服务与自动测试同时运行。

## 10. GPU Metrics Smoke

无需预先启动 vLLM：

```bash
python scripts/gpu_metrics_smoke.py \
	--config configs/qwen25_7b_metrics_smoke.yaml \
	--run-root runs \
	--prompts 16 \
	--warmup 4
```

脚本自动启动 vLLM、warmup、采样、请求、停止，并打印：

```text
runs/metrics_smoke_<时间>/trials/trial_001/metrics.jsonl
runs/metrics_smoke_<时间>/trials/trial_001/metrics_summary.json
runs/metrics_smoke_<时间>/trials/trial_001/benchmark.json
runs/metrics_smoke_<时间>/trials/trial_001/result.json
```

查看：

```bash
cat runs/metrics_smoke_*/trials/trial_001/metrics.jsonl
python -m json.tool runs/metrics_smoke_*/trials/trial_001/metrics_summary.json
python -m json.tool runs/metrics_smoke_*/trials/trial_001/result.json
```

| ID | 含义 | 来源 |
|---|---|---|
| m01 | KV cache usage P95 | vLLM Prometheus |
| m02 | 抢占 counter 增量 / 完成请求数 | Prometheus + benchmark |
| m03 | GPU utilization 均值 | 分配 UUID 的 NVML |
| m04 | waiting queue P95 | vLLM Prometheus |
| m05 | GPU memory I/O utilization 均值，不是显存占用 | 分配 UUID 的 NVML |
| m06 | TTFT P95 | 流式首 token |

样本不足时值为 `null`，`missing_reason` 说明原因，不会填 0。

## 11. 可选 LLM Worker

只验证 BO 自动重启时跳过本节并使用 `--no-llm`。

完整 BO+LLM 模式在 env 机启动 watcher。它可在 GPU tune 前启动，目录不存在时只等待：

```bash
ssh h20-env-qb-lpl
source /home/nice/qinbin/stu/lpl/tools/activate.sh
conda activate dibo
cd /home/nice/qinbin/stu/lpl/code/dibo

export DIBO_LLM_API_KEY='只在当前终端设置'

python scripts/llm_worker_watch.py \
	runs/qwen25_7b_real_run01/llm_io \
	--endpoint https://你的服务/v1/chat/completions \
	--model 审核模型名 \
	--timeout 60
```

API 错误会产生 error 响应，Controller 会 keep 当前 W 并继续。

## 12. 完整自动调参

BO-only：

```bash
dibo tune --config configs/qwen25_7b_real.yaml --no-llm
```

BO+LLM，先启动第 11 节 worker：

```bash
dibo tune --config configs/qwen25_7b_real.yaml
```

或使用包装脚本：

```bash
bash scripts/run_real_tuning.sh configs/qwen25_7b_real.yaml
```

默认运行 10 个初始 Trial 和 4 轮 x 3 个 BO Trial，共 22 次尝试。失败也计入预算。每轮 W、Metric、Action、目标和权重固定；合法新 W 只在下一轮生效。

## 13. 实时结果和输出目录

```bash
RUN=runs/qwen25_7b_real_run01
tail -f "$RUN/events.jsonl"
```

打印所有 Trial 的 TPS/Metric：

```bash
python - <<'PY'
import json
from pathlib import Path
for path in sorted(Path("runs/qwen25_7b_real_run01/trials").glob("*/result.json")):
		row = json.loads(path.read_text())
		print(row["trial_id"], row["status"], row["throughput_tps"], row["metrics"])
PY
```

离线报告：

```bash
dibo report --run "$RUN"
```

Run 级文件：

```text
experiment_snapshot.yaml  实际配置快照
environment.json          Python、Compiler、模型/workload 上下文
graph.yaml                图快照
x_init.json               解析后的初始15参数
base_trial.json           固定 x_base
summary.json              尝试数、停止原因、base/best ID
events.jsonl              可 tail 的实时事件
encoded_history.jsonl     phi 证据
trials.csv / report.md    汇总输出
action_versions/          W 版本
selections/               每轮选择、F/G 状态、BO建议、预测误差
llm_reviews/              Evidence、提案和拒绝原因
llm_io/                   文件桥请求/响应
```

每 Trial：

```text
spec.json                 启动对象
compile_trace.json        z 到最终15参数的全部决策
engine.log                vLLM 日志
benchmark.json            每请求 token、TTFT、成功状态
metrics.jsonl             原始 Prometheus/NVML 时间序列
metrics_summary.json      六项聚合、样本数、缺失原因
result.json               TrialResult 总入口
```

## 14. CLI 速查

```bash
dibo env-check --config CONFIG
dibo initial-run --config CONFIG
dibo tune --config CONFIG [--no-llm]
dibo compile --run RUN --base-trial ID --actions-version N --round N --z Z.yaml
dibo llm-review --run RUN --round N
dibo report --run RUN
```

`compile` 不用 GPU。`llm-review` 不追加 Trial。已有 run 拒绝覆盖，系统没有 resume。

## 15. 算法变量

BO 搜索 8 维 Action 系数 z。每个 F 只读邻接子向量：

| F | 输入维度和 Action |
|---|---|
| F01 -> m01 | 4：A01、A02、A05、A07 |
| F02 -> m02 | 3：A01、A02、A07 |
| F03 -> m03 | 5：A02、A03、A04、A06、A08 |
| F04 -> m04 | 4：A02、A03、A04、A05 |
| F05 -> m05 | 3：A03、A07、A08 |
| F06 -> m06 | 5：A03、A04、A05、A06、A08 |

15 维最终参数和 phi 用于真实执行、Hash、复现、投票审计和 LLM，不作为 F 输入。

## 16. 常见故障

### Placeholder 未替换

用 `prepare_real_config.py` 生成新配置。real 配置必须有模型 revision、真实 UUID、workload Hash、V1/V0、LLM 模型名和 m04 数值。

### Hugging Face 连接重置

先检查 env 网络/代理。官方源不可用时，经授权后使用 `--endpoint https://hf-mirror.com`。

### GPU preflight 找不到 UUID

说明 UUID 未分配或入口错误。不要改用其他卡，联系管理员。

### vLLM ready 超时

查看 `trials/<trial>/engine.log`。常见原因：模型路径、显存、参数兼容、V1/V0 或端口占用。

### Metrics 为 null

查看 `metrics.jsonl` 和 `metrics_summary.json`，检查 `/metrics` 字段、采样数、counter reset 和 NVML UUID。

### Run 已存在

修改 `experiment_id`。不覆盖、不自动恢复旧 run。

## 17. 当前验证状态

已验证：CPU synthetic 完整闭环；真实 argv/bool/V1 环境；mock 流式 TTFT/token；fake Prometheus/NVML 后台采样与 JSONL；real 配置的模型结构/tokenizer/workload Hash/UUID 解析；Qwen2.5-7B-Instruct 固定 revision 的模型分片、索引、配置和 tokenizer 离线完整性。

尚未验证：实际 H20 加载 Qwen、驱动与 wheel 兼容、真实 Prometheus 字段、NVML 实采样、m04 pilot、真实 LLM API 和真实吞吐提升。必须取得 GPU UUID/时段并执行第 8-12 节后，才能把这些项目标记为已验证。