# fresh_mutual_fix 同粗权重的细配准最小消融

固定服务器从头训练选出的 best_coarse.pth（粗阶段第 61 轮），只用 RoadScene train176/val23。粗网络、VGG、DNS、SA/CA、Global Correlation、粗 flow 解码器、BN 统计和预训练 flow 上采样层冻结。MutualMatching 仍使用非负相关输入。保留原局部相关与解码器；local_only 通过 local_dcn_steps=0 旁路 32 网格 DCN，不改 flow 的 VI 目标→IR 源方向或单位。

四组是冻结粗权重、DCN-only、旧局部层-only、旧局部层+DCN。后三组各从同一粗权重开始，采用相同种子 2026、每轮相同图像顺序、batch2、六轮 528 次更新、原最终 flow Charbonnier 损失及有效掩码、相同调度器，并按验证最终 512px EPE 与 min_delta=0.01 选权。local 学习率 1e-5，DCN 学习率 1e-4。六轮预算对应既有第 6 轮最佳权重的累计更新数。

原 best_fine.pth 另列为 prior_5plus1_local_dcn：前五轮 DCN-only，第六轮旧局部层+DCN。这是不同的训练日程，不能充当“从第一轮同时训练”的严格对照。报告并列呈现它，防止把日程差异误当作 DCN 作用。

所有四级 flow 统一插值到 512×512，再与同一 GT/掩码计算 EPE；16 和 32 网格 flow 从 256px 单位乘 2，64 和 128 网格已是 512px 单位。另报告 GT 位移 ≥64px、首层真实局部窗口外区域、23 对逐图、五次同步前向的配对时间及峰值推理显存。首层窗口分类来自固定粗 flow，因此不因训练细化器而改变。22 对独立测试图像不得用于选方案。

若局部层-only 的最终 EPE 与严格匹配的局部层+DCN 相差不超过 0.1px，逐图不呈多数退化，大位移与窗口外误差无明显损害，并且推理更快，则优先选旁路 DCN 的简单版本，转向 VTMOT 时序实验。仅当加入 DCN 的最终 EPE 至少好 0.2px、至少 15/23 图像改善，并且大位移和窗口外区域不退化，才继续多尺度迭代分支；其他情形保持现有最佳权重，先检查逐图原因。这些是验证集决策门槛，不是测试集调参。

已在本地用真实 fresh_mutual_fix 粗、细权重做短程检查：初始 DCN 开/关的四级 flow 最大差均为零；每组一次更新均可反传；粗阶段冻结参数和 BN 缓冲区变化为零。代码检查只用两对训练、一对验证，不能作为 23 对结果。

已收到的 23 对验证集参考结果：冻结粗权重最终 EPE 6.2970、大位移 7.9776、首层窗口外 8.5667；既有 5+1 细阶段最佳最终 EPE 5.7464、大位移 7.1064、窗口外 7.8167，逐图 12/23 改善、11/23 退化。这说明现有整体改善并非逐图普遍，尚不能把它归因于 DCN。完整同预算三组的数值仍待服务器训练。

服务器更新代码后，在 GLU-Net 目录运行下列完整命令；输出必须是新目录：

    python roadscene_local_ablation.py --data-root "G:\cxj\RoadScence" --pretrained "pre_trained_models\GLUNet_DPED_CityScape_ADE.pth" --coarse-checkpoint "roadscene_runs\fresh_mutual_fix\best_coarse.pth" --reference-fine-checkpoint "roadscene_runs\fresh_mutual_fix\best_fine.pth" --output "roadscene_runs\fresh_local_ablation_528"

会写出 report_val.json、per_image_val.csv、三个 arm 子目录各自的 history.json 和 best.pth。该脚本不覆盖原 best_coarse.pth/best_fine.pth，不启动联合训练。
