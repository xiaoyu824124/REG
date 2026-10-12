# VTMOT：每帧 A1 与 SmoothFusion 式 AKRF 的代码协议

本轮仅实现两组配准代码并运行合成数据检查，**未运行完整 800 帧验证、训练、缓存准备、融合或测试序列**。A、B 都加载同一冻结的 `best_fine_adapted_a1.pth`（SHA-256 `af245ea28cd0fc33bff7aa9dd83cb1402c1108de7870c044a9960f791cbac435`）。B 只借鉴 [SmoothFusion 论文](https://doi.org/10.1016/j.optlastec.2026.115692)第 3.1 节可核实的 AKRF 策略；这不是论文完整的配准加融合系统复现。

## 论文公式与代码

| 论文位置 | 本实验实现 | 代码位置 |
|---|---|---|
| §3.1.1 式(1)，历史光流相邻差的 `exp(-λ·L1)` 均值 | 使用当前 VI→前一 VI 的同模态光流历史；以每个 XY 分量的平均绝对差作为归一化 L1 | `vtmot_akrf_paper.py:61` `motion_consistency` |
| §3.1.1 式(2)，`max(8, floor(20·C_m))`，首帧关键帧 | 在当前帧计算完时间光流后决定是否调用 A1 | `vtmot_akrf_paper.py:74`、`:123` |
| §3.1.2 式(3)，关键帧新 flow 空间 Gaussian | 对 512px XY flow 两通道做同一 Gaussian | `vtmot_akrf_paper.py:87` |
| §3.1.2 式(4)，`α F_new + (1-α) F_k`，`α=3u²−2u³` | 先把旧关键帧 flow 经 VI 时间位移复合到当前 VI 坐标，再在有效区域加权；仅一侧有效时使用该侧 | `vtmot_akrf_paper.py:81`、`:96`、`:123` |
| §3.1.2 非关键帧的可见光 Farneback 更新 | 计算当前 VI→前一 VI 的反向 XY 光流，并沿历史关键帧 VI 网格复合，不直接逐数组相加 | `vtmot_akrf_paper.py:51`、`:123` |
| §3.1.2 式(5)，反向双线性采样 IR | 复用 `vtmot_geometry.backward_warp`；输出当前 VI→当前 IR 的 512px 位移和有效区域 | `vtmot_akrf_paper.py:123`、`vtmot_geometry.py` |

**实现选择，不是论文给定数值：**历史长度 `N=5`、`λ=0.1`（每 512px XY 分量的平均 L1）、Gaussian `σ=1px`/核 5、过渡周期 `T=20` 帧，以及现有 OpenCV Farneback 参数。见 `vtmot_akrf_paper.py:26` 的 `AKRFConfig`。论文只明确了基础间隔 20 与最小间隔 8；`N` 在本地 SmoothFusion 脚本中出现，但不能据此确认是论文表格所用的完整设置。

本地 `E:\res and fus\SmoothFusion\AKRF\utils\predict_video_GLU_keyframe2.py` 并非可证明版本来源的带 Git 元数据作者仓库，且与论文正文不一致：其中运动一致性为 `1/(1+差值)`，关键帧用固定 0.95 混合，非关键帧更新位置是 `pass`，未找到正文所述 Gaussian 与可见光 Farneback 传播实现。本实验因此以论文式(1)–(5)为主，不能称为逐项运行该本地脚本。尚未发现可核实的补充材料给出遗漏的系数。

**几何限制：**论文的非关键帧更新仅说明可见光时间光流。这里用它将旧注册场移到当前 VI 网格；将先前 IR 源坐标解释为当前 IR 源坐标，隐含 IR 成像网格跨帧稳定。若 IR 目标本身发生不同运动，该近似可能失败。正式验证须同时检查 GT EPE 与重影，不能因 ITF/T-SSIM 改善就判定配准更准。

## 预设评测口径

只在后续明确运行 `--stage eval --allow-800-frame-eval` 时，固定 4 条验证序列、800 帧、同一图像预处理、同一 A1 权重、同一独立于预测的 GT 有效掩码。保存各组逐帧 CSV、逐序列和全量加权 512px 最终 EPE、关键帧率、有效比例、完整配准计算秒数与计算 FPS。`visible_gt` 的同模态 MSE/NCC 只在模型输出和计时之后读取。

论文注册表列 NMI、`Q_AB/F`、`R_edge`、LNCC、ITF、T-SSIM，但正文及本地脚本没有足够完整的实现定义来逐项复现。现有工程的 64 桶归一化 `nmi`、`edge_information_retention`、3×3 容差 `edge_overlap_f1_3x3`、平方 `lncc17_squared`、`itf_255` 和 `t_ssim` 均保留**代理指标**名称，不能改名为论文同名指标或直接与论文表格并列。尤其 `Q_AB/F` 涉及边缘信息保留和融合输出的定义，当前配准输出没有可核实的论文同口径实现。正式报告模式中的 `paper_table_exact_metrics` 因此为 `null`，`proxy_metric_fields` 明列可计算指标。ITF/T-SSIM 只针对配准后的 IR 视频，不作为融合输出指标。

## 本地已执行检查

运行 `python vtmot_akrf_paper.py --stage check`，仅生成少量合成帧，不加载 VTMOT 或模型权重。已通过：首帧与 20/8 触发边界、式(1)高低运动变化、512 网格 Farneback 方向约 `(-3,+2)`、恒等和平移传播、旋转几何复合与越界掩码、Gaussian 常量场与 WST 端点、两组相同模拟 GT 掩码、输出后才读取模拟 `visible_gt`、现有可用代理指标数值有限。

**仍待正式实验验证：**所选 `N/λ/σ/T` 对真实 VTMOT 的影响；只靠可见光传播时移动 IR 内容的误差；关键帧切换处重影、最终 EPE、时间开销；以及论文配准表六项指标的精确定义。六条锁定测试序列不用于选择。
