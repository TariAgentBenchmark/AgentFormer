# 主体-客体改动与可视化说明

## 背景
- 任务从“预测所有行人未来 12 帧”改为“随机选 1 个主体，其余为客体；输入主体/客体 8 帧历史 + 主体未来 6 帧，只预测客体未来 12 帧”。

## 主要改动
1. **数据预处理（`data/preprocessor.py`）**
   - 随机选主体并将其放在索引 0，记录 `agent_ids`/`object_ids`。
   - 记录主体未来 6 帧观测 `subject_future_obs`，并对主体未来其他帧 mask 置 0，避免参与 loss。
   - `pred_mask`/`heading` 按新顺序重排，输出增加 `subject_*` 字段。

2. **模型管线（`model/agentformer.py`）**
   - 保存 `subject_idx`、`subject_future_*`，构造 `object_mask`。
   - 解码前若干步用主体真实轨迹替换；推理输出默认只保留客体。

3. **损失与 DLow（`model/agentformer_loss.py`, `model/dlow.py`）**
   - MSE/sample/recon/diverse 均按 `object_mask` 只累积客体误差。
   - DLow 推理返回也只保留客体，避免主体误评。

4. **测试与评估（`test.py`）**
   - 默认只评估客体；`--include_subject_eval` 可把主体写入结果/评估以便与原版对比。
   - 新增可视化：`--plot_results [--plot_limit N]` 生成末帧误差散点 + 95% 椭圆（等权样本），保存到 `results/.../figures/`。`--plot_limit <= 0` 表示不限制数量。

5. **脚本与依赖**
   - 新增 `scripts/run_eth_ucy.sh` 批量跑 ETH/UCY 两阶段训练+测试，可选清理/评估/绘图。
   - `requirements.txt` 增加 `matplotlib` 以支持绘图。

## 使用示例
- 训练：  
  - Stage1：`uv run --python .venv/bin/python train.py --cfg eth_agentformer_pre --gpu 0`  
  - Stage2：`uv run --python .venv/bin/python train.py --cfg eth_agentformer --gpu 0`
- 测试：  
  - 只客体：`python test.py --cfg eth_agentformer --gpu 0`  
  - 含主体：`python test.py --cfg eth_agentformer --gpu 0 --include_subject_eval`
- 可视化：  
  - `python test.py --cfg eth_agentformer --gpu 0 --plot_results --plot_limit 15`  
  - 不限数量：`--plot_limit 0`；含主体：加 `--include_subject_eval`。

## 已知差异
- 任务定义已变，指标与原论文/README 不可直接对齐。`--include_subject_eval` 仅用于可比性参考（主体轨迹直接用 GT 拼入结果）。

