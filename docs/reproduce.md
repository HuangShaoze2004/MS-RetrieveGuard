# 首个实施包的重建与接续

先阅读 `AGENTS.md`、`docs/status.md`、`manifests/access_policy.json`。本入口只支持已知分子式条件与U机制的工程闭环；不存在正式calibration/test评估命令。前十四天到期也不会自动解封。

## 环境

从仓库根目录执行。使用数据盘，不激活或修改服务器现有Conda环境。uv自身需已经安装。

```bash
export UV_CACHE_DIR="$PWD/artifacts/cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/artifacts/envs/python"
export TMPDIR="$PWD/artifacts/tmp"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
mkdir -p "$TMPDIR"
uv venv --managed-python --python 3.11.16 --no-project artifacts/envs/core
uv pip sync --python artifacts/envs/core/bin/python --require-hashes requirements.lock
uv pip check --python artifacts/envs/core/bin/python
artifacts/envs/core/bin/python -m pytest -q
```

已有环境先执行依赖检查；不要清空它或重新下载大型包。依赖锁包含传递依赖及wheel哈希；模型实现是对ms-mole核心配方的本地重实现，没有调用上游自动test流程。

## 顺序运行

每条命令都写配置、源码内容hash、Git状态、PID、耗时和成功/失败回执。请给新运行取新的目录名，原始日志不覆盖。

```bash
artifacts/envs/core/bin/python -u -m ms_retrieve_guard download --run-dir artifacts/runs/download-rebuild
artifacts/envs/core/bin/python -u -m ms_retrieve_guard audit --run-dir artifacts/runs/audit-rebuild
artifacts/envs/core/bin/python -u -m ms_retrieve_guard build-smoke --run-dir artifacts/runs/build-rebuild
artifacts/envs/core/bin/python -u -m ms_retrieve_guard smoke --run-dir artifacts/runs/smoke-rebuild
artifacts/envs/core/bin/python -u -m ms_retrieve_guard select-smoke --run-dir artifacts/runs/smoke-rebuild
```

数据下载固定HF revision并验证原始LFS SHA256。审计可复用SQLite化学缓存，但必须重新建立划分和候选汇总；中断后不能把旧的complete文件作为新运行成功。身份标准化或RDKit版本改变时使用新审计目录。多个审计写任务不要同时运行。

先完成全量身份及候选合法性审计，再建立1000train/100fit/100dev的一分子一谱输入。主K只按样本量选择，不查看calibration性能。模型启动前自动检查GPU计算进程和项目GPU锁；不终止其他进程。资源账本在 `artifacts/resource_ledger.json`。

工程smoke只训练1000步，取最后保存的checkpoint验证恢复与评分，不是正式R@1选模实验。正式阶段还需全训练集、独立dev_ranker早停/选模、选择器内部独立阈值部分及候选机制诊断。

## 恢复与报告

中断训练保留最近checkpoint和状态。复制原配置到新的配置文件，只添加 `ranker.resume_from`、增加目标步数并指定新run目录；数据hash、温度、seed、batch与模型结构等不可改变。先核对累计GPU预算；禁止把种子或重复训练步骤当独立样本。配置仍须保留 `scope: bootstrap_100_queries`。

完成smoke和select-smoke后生成公开摘要：

```bash
artifacts/envs/core/bin/python -c 'from ms_retrieve_guard.reporting import write_bootstrap_report; print(write_bootstrap_report("artifacts/processed", "artifacts/runs/smoke-rebuild"))'
```

`reports/bootstrap.md/json`是公开安全摘要；原始NPZ、逐查询CSV、SQLite、训练日志和checkpoint保留在本地artifacts。零接受必须显示未定义风险。下载环境包的网络量采用缓存所选wheel元数据核算，重试历史不能伪称精确流量计。
