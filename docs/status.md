# MS-RetrieveGuard 当前状态

更新：2026-10-22 UTC。**首个100查询实施包已完成**；完整基线与研究缺口尚未验证。

## 接续入口

- 实际结果与限制：`reports/bootstrap.md`、`reports/bootstrap.json`。
- 重建/恢复命令：`docs/reproduce.md`；已确认阶段计划：`docs/plan.md`。
- 分层证据：`docs/evidence.jsonl`；原始配置、日志、模型及失败保存在本地`artifacts/runs/`。
- 当前开发分支`main`；首包提交`497ce17`已快进合入。按项目所有者要求，后续直接在`main`开发并推送到`origin/main`，常规工作不另建分支。起始commit为`65f5a613d5399601f2e1dc3d8e6495432ba0b65f`；实际训练源码commit为`40aa6fa5ca200379bf7ea027087cd3df0f625c8a`。

## 已完成验收

- [x] 独立Python3.11.16/Torch2.3.0+cu121环境，36包带hash锁定，未修改base或其他环境。
- [x] 固定HF版本两文件完整下载并校验LFS SHA256，共631,503,764bytes。
- [x] 231,104行全量身份审计：28,895个有效分子；隔离289行；272身份不符、58质量不符、18身份内分子式/质量冲突行（原因重叠）；跨官方fold身份交集0。
- [x] 各角色K保留表与分子清单：`manifests/data_audit.json`、`manifests/molecule_splits.csv`。K128校准791低于预设800门槛，因此主K64，校准可用926；没有使用校准模型表现选K。
- [x] 1000train/100fit_selector/100dev_ranker分子，一分子一谱；显式身份、等K配对、原池含真值、absent无真值等价体、候选指纹索引检查。
- [x] 重复审查：133组跨角色/身份原始峰列表相同，联合前体/加合物后12组；不是已确认泄漏。所选1200条查询无触发≥4峰复合冲突规则的交集。近重复和来源级审计尚未完成。
- [x] 32项测试通过，真实评分的8组候选排列检查通过，checkpoint预测及优化器/RNG恢复差异为0。
- [x] 单种子1000步MLP工程运行、训练谱库NN、分差/概率/熵/温度选择流程贯通。100个开发查询MLP/NN的R@1均4%；严格分差规则全拒绝，接受风险为未定义。不能据此证明研究缺口或与论文方法比较。
- [x] 资源核算和失败记录：GPU共32.93秒；训练26.68秒，p50/P95为23.72/25.94ms；Torch分配峰值0.455GiB，设备采样峰值1039MiB；RSS峰值1.527GiB；下载payload约3.624GB，项目磁盘约13.5GiB。

## 封存与暴露边界

calibration/test没有检索器、选择器或阈值性能输出。必要的身份、候选、精确重复和结构暴露审计已做；未在处理SQLite存储这些部分的谱峰，也未把它们加入smoke输入。

封存自`2026-10-22T06:38:11+00:00`开始，最早`2026-11-05T06:38:11+00:00`才具备时间条件；时间到期不会自动解封，还需协议/模型/特征/阈值全部固定。新会话不得重置封存起点。

训练候选出现4个test、4个calibration身份，属于公开候选结构暴露。没有使用外部谱图预训练权重；不声称所有结构完全未见或任意分布下可靠。

## 成功运行与失败记录

- `audit-initial`：主动中断串行版本以换用经等价测试的4进程处理；保留failed回执及化学缓存，不能作为成功实验。
- `audit-parallel`：完整身份与候选审计完成；并行重启1751秒，之前串行中断554秒。原始日志`artifacts/runs/audit_parallel.log`。
- `build-initial`：输入与精确重复审计完成，55.51秒，NPZ11.64MiB；输入hash与模型输出绑定。
- `bootstrap-20261022`：MLP/NN及选择流程完成；原始分数`scores.npz`、逐查询`selection_per_query.csv`、训练`train.jsonl`、模型`checkpoint.pt`。
- 成功运行使用的代码已按内容hash从本地Git验证并归档，见`reports/source_provenance.json`。初始中断尝试的未提交中间代码仅有hash/日志，不承担定量检索结论；未来命令启动时自动归档实际工作树代码。
- 下载量来自所选wheel/数据元数据核算，不是精确网络计数；假设与余量见`manifests/environment_downloads.json`。原始数据、缓存、环境、checkpoint没有提交Git。

## 下一阶段及预算

下一步是阶段1剩余工作：全train的流式/索引化输入、完整训练谱库NN、单种子MLP，严格按dev_ranker R@1选checkpoint；比较已预注册温度网格，先做完整数据流标定。当前smoke取最后1000步checkpoint，只证明链路可运行。

正式选择器比较前，fit_selector必须再分出独立内部阈值部分。之后才进行U/H候选机制、候选先验、不看/打乱质谱等缺口验证；目前不实施稳健选择器、不宣称方法贡献。

首包上限2 GPU小时/6GB/20GiB未超；前两周累计上限20 GPU小时/15GB/80GiB。MSAlign依赖、权重/许可与吞吐尚未跑通；缺任务头和MolDeBERTa条款不阻塞核心路线。启动任何新GPU任务前先核对账本、进程及剩余预算；只管理本项目进程。
