# 主体-客体改动与可视化说明

（概述近期主体-客体改动与可视化功能，便于同事了解使用。）

## 背景
- 任务由“预测所有行人未来 12 帧”改为“随机选 1 个主体，其余为客体；输入主体/客体 8 帧历史 + 主体未来 6 帧，只预测客体未来 12 帧”。

## 主要改动概览
1. **数据预处理（`data/preprocessor.py`）**
   - 随机选主体并放到索引 0，记录 `agent_ids`/`object_ids`。
   - 保存主体未来 6 帧观测 `subject_future_obs`，并将主体后续帧 mask 置 0，避免参与 loss。
   - `pred_mask`/`heading` 按新顺序重排，输出增加 `subject_*` 字段。

2. **模型管线（`model/agentformer.py`）**
   - 保存 `subject_idx`、`subject_future_*`，构造 `object_mask`。
   - 解码前若干步用主体真实轨迹替换；推理输出默认只保留客体。

3. **损失与 DLow（`model/agentformer_loss.py`, `model/dlow.py`）**
   - MSE/sample/recon/diverse 均按 `object_mask` 只累积客体误差。
   - DLow 推理返回也只保留客体，避免主体误评。

4. **测试与评估（`test.py`）**
   - 默认只评估客体；`--include_subject_eval` 可把主体也写入结果/评估（主体轨迹直接用 GT 拼入）。
   - 可视化：`--plot_results` 生成末帧误差散点 + 椭圆，`--plot_traj` 生成历史/未来轨迹对比。`--plot_limit<=0` 不限张数。

5. **脚本与依赖**
   - `scripts/run_eth_ucy.sh` 批量跑 ETH/UCY 两阶段训练+测试，可选清理/评估/绘图。
   - `requirements.txt` 增加 `matplotlib` 以支持绘图。

## 使用示例
- 训练  
  - Stage1：`uv run --python .venv/bin/python train.py --cfg eth_agentformer_pre --gpu 0`  
  - Stage2：`uv run --python .venv/bin/python train.py --cfg eth_agentformer --gpu 0`
- 测试  
  - 只客体：`python test.py --cfg eth_agentformer --gpu 0`  
  - 含主体：`python test.py --cfg eth_agentformer --gpu 0 --include_subject_eval`
- 可视化  
  - 误差椭圆：`python test.py --cfg eth_agentformer --gpu 0 --plot_results --plot_limit 15`  
  - 轨迹对比（主体 + 客体，客体用全部 K 条预测）：`python test.py --cfg eth_agentformer --gpu 0 --plot_traj --plot_limit 15`  
  - 不限数量：`--plot_limit 0`；含主体：加 `--include_subject_eval`。

## 概率椭圆与采样数 K
- 椭圆使用所有采样的末帧误差（当前等权重 `p=1/K`），`sample_k` 由 cfg 决定。如果把 `sample_k` 调到 100，代码无需修改，会自动用 100 条样本计算均值/协方差并绘制椭圆。
- 轨迹对比同样绘制全部 K 条客体预测（浅色虚线）；增减 K 只需在 cfg 中改 `sample_k`。
- 如需按采样概率加权，可在 `plot_error_ellipse` 中将 `p = np.ones(K)/K` 替换为实际概率，再用该权重计算 `mu` 与 `Sigma`。

