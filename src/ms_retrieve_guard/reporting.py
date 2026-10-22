"""Generate a small public engineering receipt from verified local artifacts."""

import json
from pathlib import Path

from .io import sha256_file, write_json


def write_bootstrap_report(audit_dir, run_dir, destination="reports/bootstrap.md"):
    audit_dir, run_dir, destination = Path(audit_dir), Path(run_dir), Path(destination)
    audit = json.loads((audit_dir / "audit_summary.json").read_text())
    benchmark = json.loads((run_dir / "benchmark.json").read_text())
    selection = json.loads((run_dir / "selection_results.json").read_text())
    if audit["status"] != "complete" or benchmark["budget_stopped"] or benchmark["completed_steps"] != benchmark["requested_steps"]:
        raise ValueError("Incomplete runs cannot produce a successful bootstrap report")
    if selection["npz_sha256"] != benchmark["npz_sha256"]:
        raise ValueError("Report inputs disagree on dataset/candidate identity")
    rows = ["# 首个100查询实施包", "", "本报告是工程闭环和资源实测，不是完整训练基线、论文数值复现或风险认证。",
            "每个评估分子仅一张谱，分别构造present/absent版本；100个分子不能按200个独立样本计数。",
            "calibration/test仅进行了身份与候选合法性审计，没有模型分数。", "",
            "## 数据与身份审计", "", f"- 数据行数：{audit['query_rows']:,}；通过身份审计的二维分子：{audit['molecules']:,}。",
            f"- 隔离行数：{audit['quarantined_rows']:,}；跨官方fold身份：{audit['cross_fold_identities']}；身份内分子式/质量冲突：{audit['conflicting_query_identities']}。",
            f"- 当前主K：{audit['main_k']}；性能无关的样本量门槛通过：{audit['certification_sample_gate_passed']}（不代表已认证）。",
            "", "| 角色 | 分子数 | K32保留 | K64保留 | K128保留 |", "|---|---:|---:|---:|---:|"]
    for role, value in sorted(audit["retention"].items()):
        values = [str(value["by_k"][str(k)]["eligible_molecules"]) for k in [32, 64, 128]]
        rows.append(f"| {role} | {value['total_molecules']} | {' | '.join(values)} |")
    rows += ["", "## 小规模运行", "", "训练使用1000个训练分子，温度/临时阈值使用100个fit_selector分子，评估使用100个dev_ranker分子。",
             "选择器临时拟合复用了这100个fit分子，仅用于贯通流程；正式方法比较前需要独立内部阈值开发部分。",
             "", "| 项目 | 实测 |", "|---|---:|",
             f"| 训练步数 | {benchmark['completed_steps']} |",
             f"| 训练耗时 | {benchmark['training_seconds']:.2f}秒 |",
             f"| 稳态步耗时，中位／P95 | {benchmark['training_step_seconds_median']:.5f}／{benchmark['training_step_seconds_p95']:.5f}秒 |",
             f"| 峰值分配显存 | {benchmark['peak_gpu_memory_bytes']/2**30:.3f}GiB |",
             f"| 峰值进程RSS | {benchmark['peak_process_rss_bytes']/2**30:.3f}GiB |",
             f"| 输入校验与读取 | {benchmark['input_validation_and_read_seconds']:.2f}秒 |",
             f"| NPZ磁盘大小／加载数组 | {benchmark['input_npz_bytes']/2**20:.2f}／{benchmark['loaded_array_bytes']/2**20:.2f}MiB |",
             f"| checkpoint预测恢复通过 | {benchmark['checkpoint_validation']['prediction_equal']} |",
             f"| 优化器与RNG恢复通过 | {benchmark['checkpoint_validation']['optimizer_resume_equal']} |",
             "", "## 开发查询的描述性输出", "", "| 检索器 | R@1 | 分差接受率 | 分差present错误率 | 分差unknown FAR |", "|---|---:|---:|---:|---:|"]
    def fmt(value):
        return "未定义" if value is None else f"{100*value:.1f}%"
    for name, model in selection["models"].items():
        metric = model["selectors"]["margin"]["eval"]
        rows.append(f"| {name} | {fmt(model['retrieval_present']['eval']['recall_at_1'])} | {fmt(metric['present_acceptance_rate'])} | {fmt(metric['present_accepted_error_rate'])} | {fmt(metric['unknown_far'])} |")
    rows += ["", "这组数值仅验证输出链路；模型训练数据与查询均为小子集，不能据此证明研究缺口或比较论文方法。",
             "", "## 溯源", "", f"- 原始运行目录：`{run_dir}`。", f"- 检索源码commit：`{benchmark['source_commit']}`；实际源码内容hash另存运行回执。",
             f"- 输入NPZ SHA256：`{benchmark['npz_sha256']}`。", f"- checkpoint SHA256：`{benchmark['checkpoint_sha256']}`。",
             f"- 候选／划分清单SHA256：`{audit['manifest_sha256']}`。",
             "- 环境与下载：`manifests/environment_core.json`、`environment_downloads.json`、`downloads.json`。",
             "- 中断和失败保留在`artifacts/runs/failures.jsonl`及各运行回执，不能按成功实验计数。", ""]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(rows))
    public = {"kind": "engineering_smoke_not_scientific_validation", "audit": {key: audit[key] for key in ["source_sha256", "query_rows", "molecules", "quarantined_rows", "row_reason_counts", "cross_fold_identities", "conflicting_query_identities", "retention", "main_k", "certification_sample_gate_passed", "manifest_sha256"]},
              "benchmark": benchmark, "selection": selection, "source_artifacts_sha256": {
                  name: sha256_file(path) for name, path in {"audit_summary": audit_dir/"audit_summary.json", "benchmark": run_dir/"benchmark.json", "selection": run_dir/"selection_results.json"}.items()}}
    write_json(destination.with_suffix(".json"), public)
    return {"report": str(destination), "sha256": sha256_file(destination)}
