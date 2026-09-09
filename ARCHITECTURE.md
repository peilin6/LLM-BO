# DIBO 架构与运行交接说明

本文是给下一位 AI、维护者和实验操作者的项目总览。它描述当前代码的真实控制流、运行模式、每个脚本职责、自动调参过程、输出位置与实机运行前提。

权威算法和约束仍以 [DIBO v10 工程计划](DIBO_v10_env机开发与CPU测试工程计划.md) 为准；CPU 验收历史见 [ACCEPTANCE.md](ACCEPTANCE.md)；面向操作者的命令手册见 [README.md](README.md)。

## 1. 项目目标

DIBO 为 vLLM 的 15 个运行参数做自动实验与贝叶斯优化。系统不是直接在 15 维参数空间训练 GP，而是采用三层变量：

```text
W: 8 x 15 的 Action 矩阵
z: 8 维 Action 系数，BO 实际搜索的变量
x: Compile(x_base, W, z) 生成的 15 项最终 vLLM 配置
```

- BO 修改 `z`。
- Compiler 用 `W` 和 `z` 从固定 `x_base` 生成 `x`。
- 每个 Trial 真正启动 vLLM 使用的是 `x`。
- 六个 F 模型从 `z` 的邻接子向量预测六项运行 Metric。
- 一个 G 模型从六项实测 Metric 预测 TPS。
- `phi` 是最终 `x` 相对 `x_base` 的 15 维实际变化，只用于 Trace、配置去重、复现和 LLM Evidence，不是 F 的输入。
- LLM 只提议下一轮的 `W` 修改，不能直接指定 BO 系数。

## 2. 模型与图

Action 固定为 `A01` 到 `A08`，完整矩阵在 [configs/actions_v1.yaml](configs/actions_v1.yaml)。Metric 固定为 `m01` 到 `m06`，图在 [configs/graph.yaml](configs/graph.yaml)。

| F 模型 | 输出 Metric | 固定输入 Action 顺序 | 维度 |
|---|---|---|---:|
| F01 | m01 KV cache usage P95 | A01, A02, A05, A07 | 4 |
| F02 | m02 preemption rate | A01, A02, A07 | 3 |
| F03 | m03 GPU utilization mean | A02, A03, A04, A06, A08 | 5 |
| F04 | m04 queue length P95 | A02, A03, A04, A05 | 4 |
| F05 | m05 GPU memory I/O utilization mean | A03, A07, A08 | 3 |
| F06 | m06 TTFT P95 | A03, A04, A05, A06, A08 | 5 |

当 Selector 同时选择多个 Metric 时，BO 搜索这些 Metric 邻接 Action 的完整并集。例如 m01+m02 的空间是 A01/A02/A05/A07，维度为 4；m03+m06 的空间为 A02/A03/A04/A05/A06/A08，维度为 6。并集之外的 z 必须精确为 0。

## 3. 两个运行模式

| 模式 | 配置 | Runner | 用途 |
|---|---|---|---|
| `synthetic` | [configs/experiment_smoke.yaml](configs/experiment_smoke.yaml) | `FakeTrialRunner` | CPU 测试；不启动进程、不访问网络、不使用 GPU |
| `real` | 从 [configs/experiment.yaml](configs/experiment.yaml) 生成的新 YAML | `RealTrialRunner` | 真实 vLLM、HTTP benchmark、Prometheus/NVML 采集 |

Schema 会强制隔离两者：synthetic 只允许 fake adapter；real 只允许 `vllm_subprocess`、`prometheus_nvml` 与文件桥 LLM transport。real 配置在启动前还必须通过模型 revision、真实 GPU UUID、workload SHA256、m04 阈值等占位符检查。

## 4. 自动调参完整控制流

调用 `dibo tune --config <real.yaml>` 后，逻辑在 [src/dibo/controller.py](src/dibo/controller.py) 的 `run_experiment()` 中顺序执行：

```text
加载并校验配置、Action 矩阵、图、参数表、workload、模型配置
  -> 解析 x_init
  -> 创建新的 runs/<experiment_id>/
  -> 生成 10 个初始 z 点
       全零 + A01..A08 各 +0.8 + 一个 Sobol 混合点
  -> 对每一个点编译最终 x，并运行一个 Trial
  -> 从成功初始 Trial 中按 TPS、TTFT、trial_id 选定一次 x_base
  -> 对每个 BO 小轮：
       拟合 F01..F06 和 G
       -> Selector 选择 1-3 个 Metric
          硬阈值优先；否则 G 在可信区间搜索 m_k*
       -> 取图邻接 Action 的完整并集 A*
       -> Optimizer 在 A* 中生成 Sobol 候选
       -> Compiler 编译真实配置、按最终 config_hash 去重
       -> F 后验采样计算 Metric Loss MC-EI
       -> 运行 3 次新的 Trial
       -> 轮末建立全历史 Evidence
       -> LLM 合法提案才生成下一版 W；下一轮才生效
  -> 写 summary、phi 编码、CSV 和报告
```

默认预算是 10 次初始化加 4 轮乘以 3 次 BO，即 22 次实际 Trial 尝试。失败也消耗预算。若 `max_total_trials` 较小，Controller 会在初始或某一轮内提前截断，绝不超出上限。

### 4.1 一个真实 Trial 如何工作

`RealTrialRunner` 位于 [src/dibo/controller.py](src/dibo/controller.py)，但真实生命周期统一由 [src/dibo/trial.py](src/dibo/trial.py) 的 `run_trial()` 控制：

1. 写入 `spec.json` 和 `compile_trace.json`。
2. [src/dibo/engine.py](src/dibo/engine.py) 用 `create_subprocess_exec` 启动新的 `vllm serve`。
3. 只设置配置内的 `CUDA_VISIBLE_DEVICES` UUID；设置 `VLLM_USE_V1=1` 或 `0`。
4. Engine 轮询 `/health`，再验证 `/v1/models` 的 served model name，并从日志确认 V1/V0。
5. [src/dibo/benchmark.py](src/dibo/benchmark.py) 发送 warmup 请求，但不将 warmup 计入结果。
6. [src/dibo/metrics.py](src/dibo/metrics.py) 启动后台采样线程，周期性请求 `/metrics` 并读取配置 UUID 的 NVML utilization。
7. Benchmark 按 request rate、burstiness 和 max concurrency 发送正式 OpenAI Chat Completions 请求。
8. 流式响应的第一个 `delta.content` 记录 TTFT；最后 usage 中的 `completion_tokens` 记录真实输出 token。
9. Metrics 聚合 m01-m06，Benchmark 聚合 TPS 和请求成功率。
10. 无论成功、timeout 或失败，`finally` 都停止本次 sampler 和自己创建的 vLLM 进程组。
11. 写入一次 `result.json`，不自动重跑相同配置。

因此，**当前 real 模式实现会自动重启并应用 BO 新配置到 vLLM**。此前 CPU 阶段仅验证了 fake 生命周期；真实 H20 硬件执行尚未验证，必须先运行 GPU preflight 与 metrics smoke。

### 4.2 运行时日志与安全停止

所有运行事件写入 `runs/<id>/events.jsonl`，并同步打印精简终端日志。Trial 生命周期会记录启动、ready、warmup、采样、benchmark、Metric 聚合、异常、清理和结果落盘。Controller 记录模型拟合、Metric/Action 选择、BO 候选、LLM review 和实验结束。

文件桥会写 request 发布、每 30 秒等待心跳、响应、超时和中止事件。若 worker 无响应，操作员可以从另一终端创建 `runs/<id>/STOP`；Controller 在当前 Trial 清理后或文件桥下一个轮询周期结束后安全停止，并把 `stop_requested` 写入事件流和 summary。不要使用全局 kill 命令。

## 5. 组件职责

| 路径 | 关键对象/函数 | 职责 |
|---|---|---|
| [src/dibo/schemas.py](src/dibo/schemas.py) | Pydantic models, `load_*` | 固定 ID 顺序、严格 YAML/JSON、real/synthetic 隔离和占位符门禁 |
| [src/dibo/compiler.py](src/dibo/compiler.py) | `compile_config` | `x_base + scale * sum(z*W)`、分类投票、约束、Hash 与 CompileTrace |
| [src/dibo/trace_encoder.py](src/dibo/trace_encoder.py) | `encode_trace` | 15 维 phi、真实参数增量与复现证据 |
| [src/dibo/engine.py](src/dibo/engine.py) | `build_argv`, `start`, `wait_ready`, `stop` | 单个 vLLM 服务的启动、ready 检查与本进程组清理 |
| [src/dibo/benchmark.py](src/dibo/benchmark.py) | `VLLMHttpBenchmark` | JSONL 请求、Gamma burstiness、并发、流式 TTFT、completion token |
| [src/dibo/metrics.py](src/dibo/metrics.py) | `PrometheusNvmlMetrics` | vLLM Prometheus + 指定 NVML UUID 的后台原始采样和 m01-m06 聚合 |
| [src/dibo/trial.py](src/dibo/trial.py) | `run_trial` | 一个 Trial 的严格 I/O 顺序、失败状态和单次落盘 |
| [src/dibo/store.py](src/dibo/store.py) | `save_json_atomic`, `save_trial` | 原子 JSON 写入和不可覆盖 Trial 保存 |
| [src/dibo/initial_design.py](src/dibo/initial_design.py) | `generate_initial_candidates`, `choose_base_trial` | 初始化点、候选 Hash 去重、固定基础选择 |
| [src/dibo/models.py](src/dibo/models.py) | `fit_f`, `fit_g` | 六个邻接 z F、一个 G、scaler、Matérn 5/2 ARD、跨 W 版本噪声 |
| [src/dibo/selector.py](src/dibo/selector.py) | `select_metrics`, `union_neighbor_actions` | 阈值、G 反事实、探索后备、完整 Action 并集 |
| [src/dibo/optimizer.py](src/dibo/optimizer.py) | `suggest`, `metric_loss` | 候选池、编译去重、MC-EI、零 EI 不确定性探索、受限 G tie-break |
| [src/dibo/llm_update.py](src/dibo/llm_update.py) | Evidence, validator, file bridge | 全历史 Evidence、W 更新限制、requests/responses 文件协议 |
| [src/dibo/controller.py](src/dibo/controller.py) | `run_experiment` | 整个顺序闭环、实时 event、模型/版本/报告输入管理 |
| [src/dibo/report.py](src/dibo/report.py) | `write_report` | 从已保存结果离线生成 `report.md` 与 `trials.csv` |
| [src/dibo/cli.py](src/dibo/cli.py) | Typer app | 六个终端命令的入口 |

## 6. 运行脚本

| 脚本 | 在哪里运行 | 输入 | 输出/作用 |
|---|---|---|---|
| [scripts/download_model.py](scripts/download_model.py) | env 机 | Hugging Face repo/endpoint/output | 固定 revision 模型和 `DIBO_MODEL_INFO.json` |
| [scripts/prepare_workload.py](scripts/prepare_workload.py) | env 机 | 输出 JSONL、rows | 生成确定性 chat 请求并打印 SHA256 |
| [scripts/prepare_real_config.py](scripts/prepare_real_config.py) | env 或 GPU 机 | 模板、模型、workload、UUID、m04、V1/V0 | 生成无 placeholder 的 real YAML |
| [scripts/gpu_preflight.py](scripts/gpu_preflight.py) | GPU 机 | real YAML | 检查配置 UUID、Torch CUDA、vLLM、模型、tokenizer、Hash；不启动模型 |
| [scripts/print_vllm_command.py](scripts/print_vllm_command.py) | GPU 机 | real YAML | 打印零 Action 对应的完整 vLLM 命令，不执行 |
| [scripts/gpu_metrics_smoke.py](scripts/gpu_metrics_smoke.py) | GPU 机 | real YAML | 一个自动 vLLM Trial，写真实 metrics 文件 |
| [scripts/run_real_tuning.sh](scripts/run_real_tuning.sh) | GPU 机 | real YAML | 先 preflight，再 `dibo env-check`，最后完整 `dibo tune` |
| [scripts/llm_worker.py](scripts/llm_worker.py) | env 机 | 一个 request 文件、API endpoint/model | 单次处理一个 LLM 文件请求 |
| [scripts/llm_worker_watch.py](scripts/llm_worker_watch.py) | env 机 | 一个 run 的 `llm_io/`、API endpoint/model | 持续处理每轮 LLM 请求；目录未创建时等待 |

## 7. 操作者的推荐步骤

### 7.1 不用 GPU 的开发验证

```bash
source /home/nice/qinbin/stu/lpl/tools/activate.sh
conda activate dibo
cd /home/nice/qinbin/stu/lpl/code/dibo
python -m ruff check src tests scripts
python -m pytest -q tests/unit tests/interface tests/e2e/test_cpu_closed_loop.py
```

这组测试不能启动真实 socket、真实子进程或 GPU。最近完整无 GPU 联测为 196 passed。

### 7.2 模型和 workload 状态

下载好的模型：

```text
/home/nice/qinbin/stu/lpl/models/Qwen2.5-7B-Instruct
repo: Qwen/Qwen2.5-7B-Instruct
revision: a09a35458c702b33eeacc393d103063234e8bc28
size: about 15 GB
```

提供的可用 workload：

```text
data/requests/qwen_smoke.jsonl
```

新 workload：

```bash
python scripts/prepare_workload.py data/requests/qwen_mixed_128.jsonl --rows 128
```

### 7.3 获得 GPU 分配后

管理员必须提供准确 UUID、卡数和时间窗口。不得选择任意空闲 GPU。

在 GPU 机生成 smoke 配置：

```bash
python scripts/prepare_real_config.py \
  --output configs/qwen25_7b_metrics_smoke.yaml \
  --experiment-id qwen25_7b_metrics_smoke \
  --gpu-uuid GPU-<管理员分配的完整UUID> \
  --model-dir /home/nice/qinbin/stu/lpl/models/Qwen2.5-7B-Instruct \
  --request-file data/requests/qwen_smoke.jsonl \
  --execution-mode V1 \
  --llm-model disabled \
  --m04-trigger 8 --m04-target 4 --m04-scale 2 \
  --max-total-trials 1
```

`m04` 的这些数值仅用于 metrics 链路 smoke，不能用作正式研究阈值。

然后运行：

```bash
python scripts/gpu_preflight.py \
  --config configs/qwen25_7b_metrics_smoke.yaml \
  --output runs/qwen25_7b_gpu_preflight.json

python scripts/gpu_metrics_smoke.py \
  --config configs/qwen25_7b_metrics_smoke.yaml \
  --run-root runs \
  --prompts 16 \
  --warmup 4
```

只有 smoke 成功并确认 Metrics 字段后，才准备正式调参配置。

### 7.4 完整 BO-only 调参

为正式实验重新生成 YAML，使用 pilot 得到的 m04 trigger/target/scale。然后在 GPU 机：

```bash
bash scripts/run_real_tuning.sh configs/qwen25_7b_real.yaml
```

该包装脚本默认调用完整 LLM 文件桥，因此若不启动 worker，四轮末会等待配置中的 LLM timeout 后 keep。纯 BO 验证可改用：

```bash
dibo tune --config configs/qwen25_7b_real.yaml --no-llm
```

`--no-llm` 不会关闭 BO。它只让轮末 W 保持不变，BO 和自动 vLLM 重启仍完整运行。

### 7.5 完整 BO + LLM 更新 W

在 env 机，先设置 API key 并启动 worker。不要把 key 写进 YAML、shell history、run 文件或聊天内容。

```bash
export DIBO_LLM_API_KEY='<在终端中安全设置>'
python scripts/llm_worker_watch.py \
  runs/qwen25_7b_real_run01/llm_io \
  --endpoint https://<OpenAI兼容服务>/v1/chat/completions \
  --model <审核模型名> \
  --timeout 60
```

再在 GPU 机运行：

```bash
dibo tune --config configs/qwen25_7b_real.yaml
```

每轮的 3 个 Trial 结束后，GPU Controller 写 request；env worker 只处理一次并写 response；Controller 检查证据、权重变更、锚点、5/2 修改上限和版本号。非法响应、超时或 API 错误会 `keep` 当前 W，不会中断已完成 Trial。

## 8. 输出目录和观察方法

给定 `<id>`：

```bash
RUN=runs/<id>
tail -f "$RUN/events.jsonl"
```

结构：

```text
runs/<id>/
  experiment_snapshot.yaml  实际配置快照
  environment.json          Python、Compiler 和模型/workload 上下文
  graph.yaml                图快照
  x_init.json               解析后的初始 15 参数
  base_trial.json           固定 x_base
  summary.json              尝试数、停止原因、base/best Trial ID
  events.jsonl              实时事件：trial_completed/round_selected/llm_review
  encoded_history.jsonl     phi 证据
  trials.csv                所有 Trial 汇总
  report.md                 离线最终报告
  action_versions/          actions_v1.yaml, actions_v2.yaml...
  selections/               每轮选择、模型状态、BO z、F 后验、预测误差
  llm_reviews/              全历史 Evidence、提案和拒绝原因
  llm_io/                   env/GPU 文件桥 requests/responses
  trials/<trial_id>/
    spec.json
    compile_trace.json
    engine.log
    benchmark.json
    metrics.jsonl
    metrics_summary.json
    result.json
```

常用查看命令：

```bash
# 每次结果
python - <<'PY'
import json
from pathlib import Path
for path in sorted(Path("runs/<id>/trials").glob("*/result.json")):
    row = json.loads(path.read_text())
    print(row["trial_id"], row["status"], row["throughput_tps"], row["metrics"])
PY

# 原始 Prometheus/NVML 样本
cat "$RUN/trials/trial_001/metrics.jsonl"

# 聚合后的六项指标
python -m json.tool "$RUN/trials/trial_001/metrics_summary.json"

# 离线更新报告
dibo report --run "$RUN"
```

最后一条命令应写作 `dibo report --run "$RUN"`。

## 9. 指标定义

| Metric | 含义 | 原始来源 |
|---|---|---|
| m01 | KV cache usage P95 | vLLM Prometheus `vllm:kv_cache_usage_perc` |
| m02 | preemption counter 增量 / 正式完成请求数 | vLLM Prometheus `vllm:num_preemptions`，兼容旧 `_total` 名称 |
| m03 | GPU utilization 平均值 | 配置 UUID 的 NVML `utilization.gpu` / 100 |
| m04 | waiting request queue P95 | vLLM Prometheus `vllm:num_requests_waiting` |
| m05 | GPU memory I/O utilization 平均值 | 配置 UUID 的 NVML `utilization.memory` / 100，不是显存容量使用率 |
| m06 | TTFT P95 | benchmark 流式响应的首个输出 token |
| TPS | output token throughput | 正式窗口 completion token 总数 / 正式窗口秒数 |

`metrics.jsonl` 保存每次原始样本或采样错误。若样本数不足、counter reset、TTFT 缺失或指标不存在，聚合值为 `null` 并保留 `missing_reason`，不会填 0。

## 10. 当前验证状态和未完成实机步骤

已验证：

- 196 项无 GPU 联合测试通过。
- synthetic 完整 22 Trial 闭环。
- vLLM argv、全部 bool 的显式正/负 flag、V1/V0 环境设置。
- mock 流式 HTTP TTFT/completion token、fake Prometheus/NVML 后台采样和 JSONL 输出。
- real 配置对本地 Qwen 模型、tokenizer、workload Hash 和 UUID 字符串的严格加载。
- Qwen2.5-7B-Instruct 四个 safetensors 分片、索引、配置和 tokenizer 的离线完整性。

尚未验证：

- 在获配 H20 上实际加载 Qwen；
- 当前 GPU 驱动与安装的 Torch/vLLM wheel 的兼容性；
- 真实 vLLM `/metrics` 输出、真实 NVML 采样与 Metrics smoke；
- m04 的正式 pilot 阈值；
- 真实 LLM API 更新；
- 真实吞吐提升。

GPU metrics smoke 是这些实机验证的第一步。它仍被管理员分配的 GPU UUID、卡数和时段阻塞。没有这些信息时，不要运行或声称完成 GPU 测试。
