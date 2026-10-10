# VTMOT A1 单帧诊断与粗阶段适配

## 固定参照与只读诊断

固定模型是 RoadScene 上选出的无 DCN A1：`fusion_hierarchical_2x2/best_A1.pth`。本轮只读检查了既定的 4 条 VTMOT 验证序列、800 帧，没有访问测试序列。逐帧数值在 [diagnosis_per_frame.csv](diagnosis_per_frame.csv)，完整汇总在 [diagnosis_val.json](diagnosis_val.json)。预测方向为当前可见光目标坐标 → 红外源坐标，所有 EPE 与位移幅度均换算成 512×512 图像像素。16/32 网格原始 flow 是 256px 单位，报告前乘 2；64/128 网格原始 flow 已是 512px 单位。128 网格上采样就是最终 flow，没有额外的 512 网格预测层。

| 验证序列 | 有效像素 | 零 flow | 粗 16 | 局部 32 | 局部 64 | 128/最终 | GT 平均位移 | 粗预测位移 | 最终预测位移 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| photo-0306-02 | 51,109,405 | 8.681 | 35.763 | 33.729 | 32.756 | 32.098 | 8.681 | 34.905 | 31.218 |
| photo-0319-18 | 51,215,569 | 7.581 | 24.831 | 18.261 | 16.155 | 13.829 | 7.581 | 27.188 | 17.884 |
| photo-0319-25 | 51,079,529 | 9.927 | 16.079 | 9.607 | 8.154 | 7.465 | 9.927 | 17.078 | 11.663 |
| wurenji-0302-02 | 50,905,052 | 9.937 | 11.011 | 5.315 | 3.957 | 2.950 | 9.937 | 15.031 | 10.494 |
| **全 800 帧** | **204,309,555** | **9.030** | **21.934** | **16.741** | **15.268** | **14.098** | **9.030** | **23.562** | **17.823** |

局部相关搜索半径在 32/64/128 网格各为 4 个网格单元，折合 512px 坐标中每轴 ±64/±32/±16px。分母是各网格中 GT 有效面积占比 >80% 的查询点；中心采用该层前实际用于 warp 的 flow，而不是仅用上一层输出近似。

| 序列 | 32 窗口覆盖率 | 64 窗口覆盖率 | 128 窗口覆盖率 | 粗/最终方向余弦均值 | 粗/最终同向像素比例 |
|---|---:|---:|---:|---:|---:|
| photo-0306-02 | 84.0% | 60.8% | 51.8% | 0.041 / 0.018 | 51.3% / 48.9% |
| photo-0319-18 | 95.8% | 88.5% | 80.0% | 0.370 / 0.607 | 71.0% / 84.1% |
| photo-0319-25 | 100.0% | 99.9% | 94.4% | 0.382 / 0.752 | 73.5% / 92.7% |
| wurenji-0302-02 | 100.0% | 99.9% | 94.7% | 0.616 / 0.933 | 85.8% / 98.7% |

方向统计只纳入 GT 和预测位移均不小于 1px 的有效像素。`photo-0306-02` 的 200 帧粗 EPE 全部高于零 flow，196 帧最终 EPE 仍高于零 flow；其粗匹配已产生过大的错误位移，细化只小幅修正。`wurenji-0302-02` 的 32/64/128 级连续改善，最终只有 3 帧高于零 flow。故优先适配粗阶段，不先改局部网络或时序策略。`photo-0319-18` 也从粗层开始失配，细层虽然持续降低 EPE，但仍未追上零 flow。

`visible_gt` 严格在预测完成后读取，只用于核查 `gt_h` 与当前 `visible_mis`。按 GT flow 扭曲 `visible_gt` 后，800 帧有效像素上的 RGB MSE（0–255 值域）为 **1.128**，零变换为 **608.900**；800/800 帧中 GT 扭曲误差均更小。这支持几何方向、帧配对和 GT 缩放正确。该同模态核查不能单独证明红外与 `visible_gt` 的全部语义边缘完全重合。

## 唯一适配分支

从同一 A1 权重启动，仅开放 VGG `level_4`、DNS、全局 SA/CA 与粗 flow 解码器；旧局部层、预训练 flow 上采样层及全部 BN 统计冻结。不改网络、`MutualMatching` 非负保护、分层 Charbonnier 损失（`L_final + 0.1 L_32 + 0.2 L_64`）或关键帧阈值。每轮在 38 条训练序列按步长 5 取帧，相位逐轮轮换；每 5 轮覆盖所有 7600 帧。最多 30 轮，批量 2，AdamW 学习率 `1e-5`，梯度裁剪 1，验证集最终 EPE 提前停止（耐心 6 轮，最小实质进步 0.02px）。每轮在完整 800 帧上验证，按全局有效像素加权最终 EPE 保存独立最佳检查点，包含优化器、调度器、随机状态和累计步数。

本地仅完成 1 训练帧、1 验证帧的两次更新检查：损失和梯度有限，旧局部层、上采样权重与 BN 状态未改变。启用粗层参数求导后，直接推理输出与冻结 A1 的逐像素平均绝对差为 0.0080px、最大差为 0.0776px；单纯切换训练模式或粗层计算图开关时差为 0。这是当前 CUDA 路径的数值差异，**不能声称逐位相同**。每轮正式验证会临时关闭全部参数梯度，以复现部署时的前向路径，然后恢复粗阶段训练状态。单帧的第 0 轮 EPE 与诊断相差约 0.0021px；正式训练前还会重测完整 800 帧的第 0 轮，并要求与固定参照的 EPE 相差不超过 0.01px。

### 在服务器运行

以下命令在 `G:\cxj\REG\GLU-Net` 中执行。先把 `G:\cxj\VTMOT_misaligned` 改为服务器上真实的数据目录。检查命令只使用各 1 帧验证前向、反传及冻结状态；训练命令才使用 38 条训练序列与固定的 4 条验证序列。两个输出目录都应是新的空目录。

```bat
python vtmot_a1_adapt.py --stage check --data-root "G:\cxj\VTMOT_misaligned" --split-file experiments\vtmot_keyframe\split.json --pretrained pre_trained_models\GLUNet_DPED_CityScape_ADE.pth --coarse-checkpoint roadscene_runs\no_dcn_v1\best_coarse.pth --a1-checkpoint roadscene_runs\fusion_hierarchical_2x2\best_A1.pth --diagnosis-report experiments\vtmot_a1_adapt\diagnosis_val.json --output roadscene_runs\vtmot_a1_coarse_check
python vtmot_a1_adapt.py --stage train --data-root "G:\cxj\VTMOT_misaligned" --split-file experiments\vtmot_keyframe\split.json --pretrained pre_trained_models\GLUNet_DPED_CityScape_ADE.pth --coarse-checkpoint roadscene_runs\no_dcn_v1\best_coarse.pth --a1-checkpoint roadscene_runs\fusion_hierarchical_2x2\best_A1.pth --diagnosis-report experiments\vtmot_a1_adapt\diagnosis_val.json --roadscene-root "G:\cxj\RoadScence" --output roadscene_runs\vtmot_a1_coarse_adapt
```

训练完成后读取 `report_val.json`、`history.json`、`vtmot_val_per_frame.csv`、`vtmot_val_paired.csv`、`roadscene_val_per_pair.csv` 及 `best_adapted_a1.pth`。只有完整 VTMOT 验证集的适配 A1 最终 EPE 低于零 flow 的 9.030px，且至少两条序列各改善 0.1px，才重新测试关键帧传播。本轮旧过渡分支的 0/122 次放行应记为**“未实际测试到”**。已锁定的 RoadScene 22 对测试图像和 VTMOT 测试序列不用于选模型。
