# 本地股票关系研究

这是本地研究代码，不是生产服务或自动交易入口。数据、权重和论文实验结果保留在 Quant 工作区，不加入源码仓库。

- `model.py`：保留原结构，并显式隔离身份直连、身份路由、状态路由、均匀汇聚。
- `train.py`：固定对照训练，记录每次更新，保存验证选择与末轮 checkpoint，执行回测并归档。
- `reference.py`：从现有 v1 论文记录加载按 SHA256 固定的输入/回测适配代码，校验后仅在内存中设置工作区路径。避免更换预处理导致对照失真。
- `config-v2.json`、`PROTOCOL.md`：事前配置和研究边界。
- `test_model.py`：通道隔离和汇聚语义检查。

在工作区根执行：

```bash
./scripts/quant-own-exec python -m unittest discover -s /app/research/stock_relations -p test_model.py -v
./scripts/quant-own-exec python /app/research/stock_relations/train.py \
  --workspace /home/lyy/workspace/Quant --run-id stock-relation-v2-YYYYMMDD-NNN
```

需要既有 `stock-relation-20261120-001`、`factor-correlation-20261119-001` 和最初因子实验的输入引用。依赖的归档源码哈希写入配置；不从当前可变脚本路径静默加载，不自动下载缺失数据。这个研究入口只在配置的工作区运行，单独克隆平台仓库不自带实验数据或旧记录。

每个新运行独立建目录；既有 ID 拒绝覆盖。失败保留现场及论文记录。结果置于 `experiments/runs/<run_id>`，论文置于 `experiments/paper-records/<run_id>`，模型置于 `models/<run_id>`。checkpoint 从验证期选择，不能把验证期选后表现当成独立测试结果。

## 身份诊断与复核

`diagnose_stock_identity.py` 在冻结 v1 权重上隔离直接身份与分组路由；`test_identity_audit.py` 检查加权均值及干预语义。`verify_audit.py` 从原实验预测独立重算 MSE。`verify_training.py` 在独立进程检查 v2 checkpoint 选择与 16 个日期的全部种子/集成预测。

```bash
./scripts/quant-own-exec python -m unittest discover \
  -s /app/research/stock_relations -p 'test_*.py' -v
./scripts/quant-own-exec python /app/research/stock_relations/diagnose_stock_identity.py \
  --workspace /home/lyy/workspace/Quant --run-id stock-identity-audit-NEW
./scripts/quant-own-exec python /app/research/stock_relations/verify_audit.py \
  --workspace /home/lyy/workspace/Quant --audit-id stock-identity-audit-20261122-001
```

2026-11-22 的身份诊断实际由工作区脚本执行，执行版本保存在该实验 `code/`；此处是其校验后、按固定父代码加载的版本。原诊断结果及 51 项独立损失复核见对应论文；数学解释和后续决策见 [结题分析](CLOSEOUT.md)。旧 checkpoint 不含新增的 `state_projection.weight`，诊断只允许缺失这一项；它不参与 v1 五种结构的前向。

v2 同时回测 best 与固定末轮集成，共 42 个场景。源代码不承诺任何结构一定改善泛化；完整实测结论以新运行记录为准。
