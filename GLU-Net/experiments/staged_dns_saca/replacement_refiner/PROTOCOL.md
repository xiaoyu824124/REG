# 替代 DCN 的两尺度轻量残差细化器

先完成 local_ablation 的同预算细化消融，确认旧局部层-only 与 DCN 的相对作用。替代臂固定同一个 fresh_mutual_fix/best_coarse.pth，冻结 VGG、DNS、SA/CA、全局匹配、粗 flow 解码器、BN 统计和预训练 flow 上采样层。旧 GLU-Net 局部层仍可训练；32 网格 DCN 始终旁路且冻结，新分支替代它。

真实 512 输入中，局部特征网格为 64 和 128（对应输入的 1/8 和 1/4），VGG 通道分别为 256 和 128。两侧共享投影到 32 通道，再跨网格融合。每轮在 VI 目标网格按当前 512px 单位的 VI→IR flow 扭曲 IR 特征，计算有效区域的绝对特征差异，共享更新器预测不超过每轴一格的残差；先 64 网格更新一次，传给 128 网格再更新一次。残差头零初始化，所以开始训练前，四级 flow 与冻结粗权重的原局部路径完全一致。

正式第一轮实验只使用每尺度一次更新。数据为 train176/val23，batch2，六轮 528 次更新；图像顺序哈希逐轮与现有 fresh_mutual_fix 细阶段前六轮比对。旧局部层学习率 1e-5，新细化器 1e-4，和此前 DCN 的 1e-4 相同；最终 512px masked Charbonnier 损失、验证最终 EPE 与 min_delta=0.01 选权保持一致，不另加 MIND 或辅助损失。首层 DCN 不执行。运行后按原统一评测器报告四级与每轮 EPE、23 对逐图、≥64px 位移、首层窗口外、同步推理时间、显存和参数量。

只有一轮版比 local_ablation 中验证集最好的旧细化器最终 EPE 至少低 0.1px、至少 15/23 对逐图改善，且大位移及首层窗口外误差均不退化，才允许正式运行每尺度两轮版。否则停在一轮，不继续堆模块。若两轮版损害最终 EPE、速度或逐图稳定性，保留一轮或旧细化器。测试集 22 对始终不参与选型。

本地短程检查使用真实 fresh_mutual_fix 粗权重，仅两对训练图像和一对验证图像：零初始化四级 flow 最大差全为 0；一步更新可反传；冻结参数及 BN 变化为 0。它不构成完整验证集实验。

服务器需先完成 roadscene_local_ablation.py，产生 fresh_local_ablation_528/report_val.json。随后运行下面的一轮版正式实验，输出须是新目录：

    python roadscene_replacement_refiner.py --data-root "G:\cxj\RoadScence" --pretrained "pre_trained_models\GLUNet_DPED_CityScape_ADE.pth" --coarse-checkpoint "roadscene_runs\fresh_mutual_fix\best_coarse.pth" --reference-fine-checkpoint "roadscene_runs\fresh_mutual_fix\best_fine.pth" --local-ablation-report "roadscene_runs\fresh_local_ablation_528\report_val.json" --rounds 1 --output "roadscene_runs\fresh_replacement_round1"

若一轮报告中的 decision.allow_two_round=true，再加 --rounds 2、--one-round-report 指向上一轮 report_val.json，并换新输出目录。不要直接跳到两轮。
