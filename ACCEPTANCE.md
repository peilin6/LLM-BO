# v10 CPU 阶段验收对照

日期：2026-09-09。对照 [工程计划](DIBO_v10_env机开发与CPU测试工程计划.md) 的 Day 1–14、M01–M14、IF01–IF04 与 E2E-01。

## 范围与授权

- 工作限于 env 机；未登录 GPU 机、未加载模型、未启动 vLLM、未调用 NVML 或真实 LLM API。
- 用户额外授权安装完整 GPU 软件依赖，覆盖 v10 文档的 CPU-only 安装限制；仅安装不执行。
- 单元、接口与 CPU E2E 已获授权直接运行。原始工程计划未改写成运行日志。

## 逐项对照

| Day | 内容 | 验证文件/门禁 |
|---|---|---|
| 1 | Python 3.10 环境、包及 CLI | test_package_smoke.py、test_cli.py、pip check |
| 2 | Schema、参数、W、图及 fake 隔离 | test_schemas.py、test_config_snapshots.py |
| 3–4 | 数值/分类编译、约束与 Hash | test_compiler_numeric.py、test_compiler_categorical.py、test_compiler_constraints.py |
| 5 | phi 与原子本地存储 | test_trace_encoder.py、test_store.py |
| 6 | Engine 注入式进程/HTTP 边界 | test_engine.py（fake） |
| 7 | Benchmark/六项 Metric 聚合 | test_benchmark.py、test_metrics.py（fixture） |
| 8 | 单 Trial 生命周期、失败与清理 | test_trial.py、test_compile_to_trial.py |
| 9 | 10 点初始化、去重、基础选择 | test_initial_design.py |
| 10 | 邻接 z F、六维 G、版本噪声 | test_f_models.py、test_g_model.py、test_store_to_models.py |
| 11 | 阈值/G/后备、全部邻接并集 | test_selector_threshold.py、test_selector_counterfactual.py、test_graph_union.py |
| 12 | 四种 Loss、MC-EI、受限 G 次级排序 | test_metric_loss.py、test_optimizer.py、test_selector_to_optimizer.py |
| 13 | Evidence、提案规则、文件桥、版本保存 | test_llm_evidence.py、test_llm_validator.py、test_llm_file_bridge.py、test_llm_update.py、test_trial_to_llm_update.py |
| 14 | Controller、报告、六个 CLI、CPU 闭环 | test_controller.py、test_report.py、test_cli.py、test_full_interfaces.py、test_cpu_closed_loop.py |

## 最终门禁结果

以下命令于 2026-09-09 在 dibo Python 3.10 环境中执行，退出码均为 0：

| 命令 | 结果 |
|---|---|
| `python -m pip check` | No broken requirements found |
| `python -m ruff check src tests scripts` | All checks passed |
| `python -m pytest -q tests/unit/test_controller.py tests/unit/test_report.py tests/unit/test_cli.py` | 18 passed |
| `python -m pytest -q tests/unit` | 168 passed |
| `python -m pytest -q tests/interface` | 10 passed |
| `python -m pytest -q tests/e2e/test_cpu_closed_loop.py` | 6 passed |
| `python -m pytest -q tests/unit tests/interface tests/e2e/test_cpu_closed_loop.py` | 184 passed in 130.01s |

联合计数与分组计数一致：168 + 10 + 6 = 184。没有 skip、xfail 或真实 GPU 测试；测试结果写入 pytest 临时目录，未创建正式 GPU 实验记录。

### Real 运行链路扩展验收（2026-09-09）

在 CPU 阶段验收后，补充了真实 OpenAI chat benchmark、Prometheus/NVML 后台采样、每 Trial `metrics.jsonl`/`benchmark.json`、RealTrialRunner、GPU preflight、metrics smoke、完整调参包装脚本和实时 `events.jsonl`。随后重新执行：

| 命令 | 结果 |
|---|---|
| `python -m ruff check src tests scripts` | All checks passed |
| `python -m pip check` | No broken requirements found |
| `bash -n scripts/run_real_tuning.sh` | 通过 |
| `python -m pytest -q tests/unit tests/interface tests/e2e/test_cpu_closed_loop.py` | 196 passed in 139.74s |

新增 real adapter 使用 mock HTTP/NVML 和本地模型 fixture 验证，联合测试仍未连接 GPU。下载的 `Qwen/Qwen2.5-7B-Instruct` 固定 revision 为 `a09a35458c702b33eeacc393d103063234e8bc28`，位于 `/home/nice/qinbin/stu/lpl/models/Qwen2.5-7B-Instruct`；四个 safetensors 分片、索引、配置与 tokenizer 已离线校验，总大小 15,242,808,990 bytes。

GPU metrics smoke 仍待管理员提供明确的 GPU UUID、卡数和允许时段。用户表示“任意一张”，但服务器规范禁止自行挑卡，因此未执行 GPU 命令，也未生成伪造的真实 metrics 输出。

## 闭环验收内容

- 默认算法预算 10+4×3=22；完整合成 E2E 实际产生 22 个唯一配置结果。
- 受控场景覆盖硬阈值、G 反事实、未就绪模型、零 EI、失败计数和总预算截断。
- 每轮 W/Metric/邻接并集/目标固定，下一轮才使用合法新 W；初始 x_base 不移动。
- 重编译候选的 Hash 与真实执行的 synthetic Trace 相等，集合外 z 为零。
- E2E 中不同轮次出现不同邻接维度，并在受控函数上验证 Metric 目标距离改善。
- 旧 W 成功历史继续进入邻接 z F，观测噪声加倍；失败记录不作性能改善证据。
- CLI 集成验证 tune、compile、llm-review、report；人工 review 不增加 Trial。
- 报告拒绝 mixed real/synthetic 数据，最佳记录仅来自成功 Trial，预测 TPS 不冒充观测。
- 资源守卫覆盖全测试集；没有 GPU E2E 或 GPU skip 充当验收。

## 已修正的集成问题

- 总预算 Schema 从强制等于日程改为允许提前截断。
- LLM neutral 可用于 weaken/set_zero；缺失目标 Metric 不得支持性能修改。
- 文件桥严格校验协议版本/文件身份，已有响应的请求不重复调用 reviewer。
- 每次建议保存 F 预测均值、方差、观测误差和投影策略；轮记录保存模型状态与训练 ID。

## 不在本阶段的实机验收

真实模型初始化、实际 V1/V0、vLLM CLI 全参数兼容性、NVML 实采样、硬件内存可行性、真实 LLM API、m04 pilot 阈值和真实吞吐结果仍待后续 GPU 验证。synthetic 初始化、fake 日志/采样不替代这些验证。

测试通过证明列出的实现路径在固定 fixture 与合成场景下可联合工作，不是统计性能结论或对所有外部版本的兼容性证明。