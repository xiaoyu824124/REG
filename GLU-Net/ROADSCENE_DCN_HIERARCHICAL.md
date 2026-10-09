# RoadScene 两尺度 DCN 与损失对照

本实验沿用 `fusion_hierarchical_2x2` 的固定 `no_dcn_v1/best_coarse.pth`、176 对训练图像、23 对验证图像、旧局部层初始权重、增强、随机种子和选权口径。锁定的 22 对测试图像不会被读取。已有 A0/A1 权重只读，报告时在同一设备上重新计时和复核 EPE。

| 组别 | 32、64 网格的真实 DCN | 损失 |
| --- | --- | --- |
| A0（既有） | 无 | `L_final` |
| A1（既有） | 无 | `L_final + 0.1 L_32 + 0.2 L_64` |
| D0 | 各一次 | `L_final` |
| D1 | 各一次 | `L_final + 0.1 L_32 + 0.2 L_64` |
| D2 | 各一次 | `L_final + 0.1 L_res32 + 0.2 L_res64 + 0.001 (L_offset32 + L_offset64)` |

`L_final` 和分层损失均为当前有效像素 Charbonnier flow 损失。D2 的 `L_res` 直接监督 DCN 输出的增量，使其接近 `GT flow－进入该层前的 flow`。先把 GT 和 flow 转换成相同的 512px 位移单位；只在 GT 有效、当前源图采样在界内且目标残差处于该 DCN 可更新的 ±4 个特征格范围内计算辅助损失。`L_offset` 是 offset 绝对值均值，单位为特征格。辅助损失的梯度仅进入 DCN 参数；共同的最终损失仍训练旧局部层和 DCN。辅助项的有效比例、残差损失与 offset 大小会逐轮记录。验证和选权只使用**全部有效像素**的最终 512px EPE，不使用 GT 作为推理输入。

两层 DCN 的 `offset` 与 `Δflow` 输出头都从零初始化，训练前必须逐像素复现 A0。32 网格的 flow 原本是 256px 单位，计算损失时乘 2；64 网格的 flow 已是 512px 单位。warp 使用 VI 目标位置到 IR 源采样位置的 flow。原有 `dc_conv*` 是空洞卷积，仍按既有局部解码器使用；预训练 flow 上采样层继续冻结。

D0/D1/D2 使用既有开发集确定的旧局部层学习率 `5e-6`、新模块学习率 `1e-5`、AdamW、相同余弦调度、训练图像顺序及 **71 轮、每组 6248 次更新**。这些数字来自已完成的 A0/A1 训练记录，不根据 D 组验证结果重设。每组分别按 23 对验证集最终 EPE 保存最佳权重，不以末轮替代最佳轮。重点比较 D0−A0、D1−A1、D2−D0、D2−D1，并查看 23 对逐图改善数、边缘与大位移区域、推理时间。若 D2 只有训练损失降低或少数图像改善，不将其认定为有效。

## 服务器运行

先把下方“本次改动文件”同步到服务器的 `G:\cxj\REG\GLU-Net`。服务器还须有 `roadscene_runs\fusion_hierarchical_2x2` 中的 A0/A1 最佳权重、报告、逐图 CSV、训练曲线和锁定配方。每次使用**新的空输出目录**。

**第一步：检查零初始化、flow 方向与尺度、两层梯度。**

```bat
python roadscene_dcn_hierarchical.py --stage check --data-root "G:\cxj\RoadScence" --pretrained "pre_trained_models\GLUNet_DPED_CityScape_ADE.pth" --coarse-checkpoint "roadscene_runs\no_dcn_v1\best_coarse.pth" --reference-dir "roadscene_runs\fusion_hierarchical_2x2" --output "roadscene_runs\dcn_hierarchical_check"
```

**第二步：用一对训练图像和一对验证图像试跑更新与存档流程。**

将 `--stage check` 改为 `--stage smoke`，并将输出改为新的 `roadscene_runs\dcn_hierarchical_smoke`。

**第三步：正式训练和比较。**

```bat
python roadscene_dcn_hierarchical.py --stage train --data-root "G:\cxj\RoadScence" --pretrained "pre_trained_models\GLUNet_DPED_CityScape_ADE.pth" --coarse-checkpoint "roadscene_runs\no_dcn_v1\best_coarse.pth" --reference-dir "roadscene_runs\fusion_hierarchical_2x2" --output "roadscene_runs\dcn_hierarchical_71"
```

生成 `best_D0/D1/D2.pth`、`history_D0/D1/D2.json`、`report_val.json` 与逐图 CSV。正式结果尚未运行；`check` 与 `smoke` 的数值不可当作实验结果。

本地已通过 Python 语法检查、两层 DCN 的零位移/方向/尺度/越界几何检查，以及专用损失的张量反传检查。本机 CuPy 初始化未完成，因此完整模型的 `check` 和 `smoke` 尚未通过；服务器必须先完成这两步再正式训练。

本次改动文件：

1. `models/our_models/GLUNet.py`：增加可选 64 网格 DCN 和两层更新轨迹。
2. `roadscene_fusion_hierarchical.py`：可选返回 DCN 更新轨迹；原 A0/A1 路径保持默认关闭。
3. `roadscene_dcn_hierarchical.py`：三种 DCN 损失对照、训练检查和同口径报告。
4. `ROADSCENE_DCN_HIERARCHICAL.md`：实验协议与中文运行说明。
