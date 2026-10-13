# VTMOT 因果匹配状态：固定关键帧 A/B 对照

## 已固定的部分

- 两组使用同一 `best_fine_adapted_a1.pth`，程序要求 SHA-256 为 `af245ea28cd0fc33bff7aa9dd83cb1402c1108de7870c044a9960f791cbac435`，并冻结完整 A1（含 BN）。
- 验证集的 4 条序列、800 帧及关键帧位置从 `experiments/vtmot_local_correction/frozen_keyframes.json` 读取。A、B 共用 256 网格 Farneback 双模态时间位移、前后向检查、VI→IR 几何传播、512×512 输入与 GT 掩码。B 的输出不回写 A 的传播链。
- 本轮仅训练 B 的状态残差头。VTMOT 训练集为 38 条序列；每条每轮预先固定 20、70、120、170 四个锚点及随后最多 4 帧。训练关键帧由旧固定间隔和输入驱动的失败回退规则在训练集生成，不使用 GT。30 轮、AdamW `1e-4`、余弦退火、裁剪梯度范数 1；每轮用完整 4 条验证序列的有效像素加权 512px EPE 选最佳头（第 0 轮的零头也是候选）。A 无需训练。
- 损失沿用最终 flow 的有效像素 Charbonnier：`sqrt(||F_pred-F_GT||²+0.01)`。不加时间损失、未来帧、DCN 或轮内循环更新。

## 新增的唯一变量

冻结 VGG 的 `level_2` 提取当前 VI、IR 的 128×128 特征；围绕 A 给出的 VI→IR flow 建立 5×5 局部相关。上一帧保存的相关状态沿当前 VI→上一 VI 时间位移和有效掩码对齐到当前 VI 网格，与当前相关证据均值融合。共享的轻量卷积头预测 128 网格残差，上采样并乘 4 后加到 A 的 512px flow。最后一层权重和偏置为零，训练前 B 逐像素等于 A。关键帧上状态重置，直接使用 A1 结果。

这种状态只借鉴 VideoFlow 的历史状态按运动对齐思想；**不是 TROF 或 MOP 复现**。TROF 的三帧双向光流网络、MOP 的学习式多方向运动传播均未加入。此实现只有过去与当前，不读取未来帧。

## 评测及采用规则

- 主指标：相同 GT 有效掩码上的 512px flow EPE；补充 PCK@1/3/5px、逐序列和逐帧 EPE、距离关键帧的误差、移动区域和低置信区域、最差帧。
- 跨模态辅助指标沿用项目实现：`nmi`、`lncc17_squared`、`edge_overlap_f1_3x3`。后两者是实现明确的代理指标，不声称与其他论文表格的 LNCC/边缘指标完全同定义。`visible_gt_mse/ncc` 仅在预测后评测。
- 时间误差是上一帧已配准 IR 图经当前 VI→上一 VI 时间位移补偿后，与当前已配准 IR 图在双方有效区域的灰度 MAE。总推理时间含四次 Farneback、几何传播、关键帧 A1，以及 B 的 VGG、局部相关、历史对齐与残差头；磁盘 IO 和 GT/指标计算另列端到端墙钟时间。
- 预设判断：B 相对 A 的完整验证集 EPE 至少下降 0.1px，且至少 3/4 序列改善；若 EPE 未达到该值而计算时间增加，停止此分支。与每帧 A1 的质量、成本另行并列，不以锁定的 6 条测试序列选择模型。

## 本地可运行的检查

在 `GLU-Net` 目录运行：

```bat
python vtmot_motion_state_ab.py --stage check
```

该命令只运行合成几何、时间 flow、相关峰位置、历史状态对齐、两帧零残差和梯度检查，不访问 VTMOT。带齐真实路径和权重时，`check` 额外只读验证集首序列前 5 帧的 A/B 初始一致性，不读取 GT。

正式服务器命令见交付说明；训练输出使用新空目录，评测时读取其中 `best_head.pth`。当前仓库只包含代码与协议，不包含训练数值或 800 帧评测结论。

## 手动复制到服务器

本次新增两项：`GLU-Net/vtmot_motion_state_ab.py`、`GLU-Net/experiments/vtmot_motion_state/PROTOCOL.md`。脚本还依赖服务器已有的 `datasets/vtmot_video.py`、`vtmot_geometry.py`、`vtmot_keyframe.py`、`vtmot_local_correction_ab.py` 和验证关键帧 JSON；未修改这些依赖文件。
