# 无 DCN 细配准 2×2 对照

本实验固定 `no_dcn_v1/best_coarse.pth`，只使用 RoadScene 的训练集和 23 对验证图像。22 对独立测试图像不会被读取。两条结构路线均从该粗权重中保存的相同 GLU-Net 预训练局部层出发；粗网络和 BatchNorm 统计量冻结，预训练 flow 上采样层保持冻结。文件名 `dc_conv*` 表示原 GLU-Net 的空洞卷积，不是可变形卷积。

| 组别 | 局部结构 | 损失 |
| --- | --- | --- |
| A0 | 原 GLU-Net 局部层 | `L_final` |
| A1 | 原 GLU-Net 局部层 | `L_final + 0.1 L_32 + 0.2 L_64` |
| B0 | 128/256 局部特征在 `decoder1` 输入前融合 | `L_final` |
| B1 | 与 B0 相同 | `L_final + 0.1 L_32 + 0.2 L_64` |

`L` 使用现有的有效掩码 Charbonnier flow 损失。32 网格的 flow 从 256px 单位乘 2；64 网格和最终 flow 已是 512px 单位；三项都与同一份 512×512 GT 比较。128 flow 只是最终 flow 的低分辨率输出，不重复加损失。没有粗损失、真 DCN、循环迭代或新上采样层。

B 组先分别汇聚 VI 和 IR 的 128/256 VGG 特征，再做零门控 DNS、有限窗口自注意力。按当前 VI→IR flow 在 128 网格搜索半径 2、256 网格搜索半径 1 的源特征窗口，做局部交叉注意力。两个尺度的信号投影后，以零初始化残差加到原 `decoder1` 输入，因而参与原局部解码器的 flow 预测。训练前逐级比较四组输出，且检查新增投影的梯度。CUDA 确定性卷积设置也在脚本内固定。

## 服务器运行

在服务器 `G:\cxj\REG\GLU-Net` 中运行下列一条命令。输出目录必须是新的空目录；粗权重须位于所列路径：

```bat
python roadscene_fusion_hierarchical.py --stage all --data-root "G:\cxj\RoadScence" --pretrained "pre_trained_models\GLUNet_DPED_CityScape_ADE.pth" --coarse-checkpoint "roadscene_runs\no_dcn_v1\best_coarse.pth" --output "roadscene_runs\fusion_hierarchical_2x2"
```

该命令依次完成：

1. **退化诊断**：用旧的 `1e-5` 局部层学习率，在第 0、1、2、3 轮记录训练和验证逐级 EPE、更新前梯度范数、局部参数相对变化。这里只诊断现象，不单独宣称学习率是原因。
2. **训练集内部开发选择**：固定 24 对训练图像作开发集，其余训练图像用于试跑；对同一套成对水平/垂直翻转，比较较小的旧局部层学习率 `1e-6`、`2e-6`、`5e-6`。只根据开发集选择一档，写入 `locked_recipe.json`。翻转时同步改变两幅图像、GT flow 分量的符号和有效掩码。
3. **四组正式训练**：全部 176 对训练图像使用相同批次顺序、增强、选定学习率、最多 80 轮。融合模块的学习率固定为 `1e-5`。四组同步早停：只有四组都连续 12 轮未达到至少 `0.01px` 的验证改善才停止，保证更新步数相同。各组按完整 23 对验证集最终有效像素加权 EPE 独立保存最佳权重，包括第 0 轮；最后一轮不会覆盖最佳权重。

主要文件：`diagnosis_old_local.json`、`locked_recipe.json`、`coverage_val.json`、`history_A0/A1/B0/B1.json`、`best_A0/A1/B0/B1.pth`、`per_image_paired.csv`、`report_val.json`。逐图 CSV 另含各级 EPE、低误差像素比例、边缘和大位移区域误差。覆盖率包括真实 32 网格第一局部窗口中心以及 128/256 窗口，单列 `000002` 和 `000014`。

正式结果尚未运行。`--stage smoke` 只在各一对训练和验证图像上检查代码、梯度、选权和报告流程，不能当作实验结果。结构作用看 B0−A0；分层损失作用看 A1−A0 和 B1−B0。只有最终 EPE 与逐图稳定性都支持 B1，才考虑保留组合；中间层或训练损失单独改善不构成采用理由。
