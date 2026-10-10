# VTMOT A1 细配准适配

## 固定方案

从 `roadscene_runs/vtmot_a1_coarse_adapt/best_adapted_a1.pth` 开始，只训练原 GLU-Net 的 32、64、128 网格局部解码器。冻结 VGG、DNS、全局 SA/CA、粗 flow 解码器、预训练 flow 上采样层和所有 BN 统计。名字含 `dc_conv` 的旧层是空洞卷积；本模型没有可变形卷积及 offset 分支。输入为当前可见光目标和红外源，flow 方向为可见光坐标到红外坐标，监督统一换算为 512×512 像素单位。

仅使用 38 条 VTMOT 训练序列更新参数；4 条验证序列共 800 帧选择检查点。每轮每条序列按步长 5 取帧，相位逐轮轮换。损失沿用 A1 的 `L_final + 0.1 L_32 + 0.2 L_64` Charbonnier，批量 2，AdamW 学习率 `5e-6`，梯度范数裁剪到 1，余弦调度，最多 30 轮。验证最终 EPE 若连续 6 轮未获得至少 0.02px 的进步则提前停止；最佳轮的模型、优化器、调度器、随机状态和累计步数独立保存。第 0 轮的模型是已适配粗配准的权重，并须在完整验证集复现原 5.8841px 结果。

EPE 用于训练期几何选权，不单独作为视频配准结论。训练后对粗适配与细适配模型在相同 800 帧、有效掩码下做逐帧配对，报告四个网格的 512px EPE、零 flow、逐序列及困难帧、改善帧数和 RoadScene 验证集保持情况。随后才在固定的视频协议下重算关键帧和传播的 NMI、跨模态边缘信息保持、边缘重叠率、LNCC、ITF、T-SSIM、评测专用 `visible_gt` MSE/NCC、关键帧率和完整耗时。视频指标若只是来自更多 A1 重估或增加计算量，需要如实呈现。RoadScene 的 22 对独立测试图像及 VTMOT 测试序列不用于选择。

## 服务器运行

手动复制唯一新增的 `GLU-Net/vtmot_a1_fine_adapt.py` 到服务器相同位置；它复用服务器已有的 `vtmot_a1_adapt.py`、`datasets/vtmot_video.py`、`vtmot_geometry.py`、`experiments/vtmot_a1_adapt/split.json` 和既有权重。两个输出目录必须是新目录。在 `G:\cxj\REG\GLU-Net` 中先执行一帧检查：

```bat
python vtmot_a1_fine_adapt.py --stage check --data-root "G:\cxj\VTMOT_misaligned" --split-file experiments\vtmot_a1_adapt\split.json --pretrained pre_trained_models\GLUNet_DPED_CityScape_ADE.pth --coarse-checkpoint roadscene_runs\no_dcn_v1\best_coarse.pth --a1-checkpoint roadscene_runs\fusion_hierarchical_2x2\best_A1.pth --coarse-adapt-checkpoint roadscene_runs\vtmot_a1_coarse_adapt\best_adapted_a1.pth --output roadscene_runs\vtmot_a1_fine_check
```

检查通过后执行完整训练：

```bat
python vtmot_a1_fine_adapt.py --stage train --data-root "G:\cxj\VTMOT_misaligned" --split-file experiments\vtmot_a1_adapt\split.json --pretrained pre_trained_models\GLUNet_DPED_CityScape_ADE.pth --coarse-checkpoint roadscene_runs\no_dcn_v1\best_coarse.pth --a1-checkpoint roadscene_runs\fusion_hierarchical_2x2\best_A1.pth --coarse-adapt-checkpoint roadscene_runs\vtmot_a1_coarse_adapt\best_adapted_a1.pth --roadscene-root "G:\cxj\RoadScence" --output roadscene_runs\vtmot_a1_fine_adapt
```

训练后将 `report_val.json`、`history.json`、`vtmot_val_per_frame.csv`、`vtmot_val_paired.csv` 和 `best_fine_adapted_a1.pth` 复制回本地同名输出目录，以便运行后续固定协议的视频指标评测。若服务器缺少前述已存在的依赖文件或权重，应逐一按相同相对路径补齐，不要直接复制整份旧工程覆盖它。

## 本地代码检查

在本地 `vfbench` 环境以训练集和验证集各一帧执行 `--stage check`：初始最终 flow 最大绝对差为 0，粗 flow 在一次局部更新后的差为 0，非训练参数及 BN 状态未变化，局部梯度存在且有限。首帧验证 EPE 与已保存的粗适配逐帧记录相差 0.00168px；这是本地 CUDA 路径的微小数值差异。尚未运行完整训练，不填写任何细适配效果数值。
