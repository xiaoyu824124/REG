# VTMOT 单帧 A1 与视频关键帧配准

本实验只做配准，不运行融合模块。固定 `fusion_hierarchical_2x2/best_A1.pth`，并校验它对应的 `no_dcn_v1/best_coarse.pth` 与基础 GLU-Net 权重 SHA-256。A1 全程 `eval()`，全部参数冻结。`split.json` 沿用本地 VTMOT 的 38 条训练、4 条验证、6 条锁定测试序列；程序只接受 `train`/`eval`，不会打开测试序列。

## 几何约定

512×512 中所有 flow 为 `(dx,dy)` 像素位移。A1 输出 `当前 VI 目标 → 当前 IR 源` 的反向采样场。非关键帧按当前 VI → 上一 VI → 上一 IR → 当前 IR 的顺序复合，每一步在新坐标双线性取值并交集有效掩码。时间位移由同模态 Farneback 光流估计：VI 当前→上一及其反向、IR 上一→当前及其反向。前后向误差、复合有效比例、两模态运动分歧只依赖输入和模型输出。首帧及时间估计失效时强制 A1 重估。

输入 IR 和 `visible_mis` 共同进行中心裁剪及 PIL 双线性缩放到 512×512；GT 单应矩阵作相同仿射共轭，得到 VI→IR 的 XY GT flow。原始 640×480 图裁成 480×480 后缩放；因此这些 EPE 数字是 **512×512 方形裁剪的像素单位**，不能与原 640×480 指标直接比较。所有方法使用相同 GT 边界掩码评价，预测的传播有效掩码只用于触发和单列报告，不能让困难像素从 EPE 分母消失。`visible_gt` 只在已经决定当前帧预测后，为同模态 MSE/NCC 读取。

## 依次运行

以下命令在 `GLU-Net` 目录执行。把数据路径改成服务器上 `VTMOT_misaligned` 的实际位置。已有报告不会覆盖，请为重跑另取输出目录。这里是推理、评测与训练集阈值校准，**没有模型训练**。

1. 几何与 GT 方向检查：

   `python vtmot_keyframe.py --stage check --data-root "G:\cxj\VTMOT_misaligned" --split-file experiments\vtmot_keyframe\split.json --pretrained pre_trained_models\GLUNet_DPED_CityScape_ADE.pth --coarse-checkpoint roadscene_runs\no_dcn_v1\best_coarse.pth --a1-checkpoint roadscene_runs\fusion_hierarchical_2x2\best_A1.pth --output roadscene_runs\vtmot_keyframe_v1`

2. 先跑验证集每帧 A1 与零 flow 基线：把上一条命令的 `--stage check` 改为 `--stage baseline`，其他参数保持相同。

3. 在 **训练序列**固定自适应阈值：改为 `--stage calibrate`，建议加 `--calibration-pairs-per-sequence 4`。38 条训练序列各取 4 对相邻帧；只用输入图像、A1 预测和时间光流，不读取训练 GT。阈值保存为 `train_thresholds.json`，不会被验证结果修改。

4. 验证固定间隔与自适应传播：改为 `--stage compare`。程序输出 `fixed/` 与 `adaptive/` 的逐帧 CSV、逐序列报告、与每帧 A1 的配对差异及困难帧重影图。

5. 最后做自适应关键帧的单帧过渡对照：改为 `--stage transition`。只在传播与新 A1 同处当前坐标、双方有效且至少 90% 有效像素差异≤2px，且新 A1 的边缘对齐没有明显更好时进行一次 50/50 过渡；历史始终重置为新 A1。程序报告与不经渡自适应方案在切换帧处的配对误差。

固定关键帧间距为 5 帧（帧号 0、5、10…），动态最长间距为 8 帧（首帧 0，最迟第 8 帧重估）；作为实验协议预先固定，未依据验证集 GT 调整。额外紧急重估阈值用训练序列时间置信度的分位数固定。若适配后接近每帧都重估，应按关键帧率与完整耗时如实报告。

验证方案采用预先固定的成本与精度门槛：相对每帧 A1 的像素加权 EPE 最多增加 0.5px、纯计算耗时最多为 A1 的 80%、关键帧比例最多为 50%；合格方案中取 EPE 最低者，否则保留每帧 A1。完整验证结果仍会报告，即使三种方案都不优于零 flow。过渡对照只检查切换过程，不更改这条主方案规则。

## 指标范围

主指标为相同 GT 掩码下的 512px EPE，分别给出像素加权、逐帧平均与逐序列平均。NMI 使用固定 64 桶灰度互信息/平均熵；LNCC 为 17×17 窗口平方局部互相关；边缘信息保持是 VI 边缘能量被配准 IR 保留的比例，边缘重叠是 3×3 容差、两侧 top-quartile Sobel 边缘的 F1。ITF 为相邻配准 IR 帧的 0–255 灰度绝对差，T-SSIM 是相邻配准 IR 帧的 SSIM。MSE/NCC 在评测阶段使用 `visible_gt` 经预测 flow warp 后与 `visible_mis` 比较。这些是**配准输出上的统一代理指标**；SmoothFusion 原库中的 NMI/R_edge/ITF/T-SSIM 主要为融合输出定义，不能将本实验数值等同于其论文表格。

逐帧 `frame_ms` 包含光流、坐标复合、触发决策及关键帧 A1 推理；GPU 计时前后同步，不含数据读取和评测指标。报告另给包含 I/O 和评测的整段墙钟时间。两种时间口径均不包括权重加载。

如果 VTMOT 上每帧 A1 本身弱于零 flow，关键帧传播只能检验实时性与误差传递，不能声称解决跨数据集配准；应先定位域差异与失败序列，再决定是否只用 VTMOT 训练序列做适配。
