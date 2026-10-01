# 实际结果与实现记录（2026-10-01）

本地已有 vfbench 环境；没有安装依赖。原有测试结果锁定，新分析仅 train/val。
CRFT 缺训练权重；MIND/DNS 正式训练尚未运行。随机权重、2train+1val 的 smoke 只验证代码，不代表效果。

## 23对验证集：固定历史最佳权重，统一评测

完整 flow 和 AEPE 的单位均为512px；粗 EPE 为256px。CMR按图像对的AEPE严格小于阈值统计。

| 方法 | 粗EPE | Top-1 | 最终flow EPE | 平均逐对AEPE | CMR@5 | CMR@3 | CMR@1 / @0.7 | ms/对 | 峰值MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GLU-Net原始架构（decoder4微调） | 13.3044 | 14.60% | 17.4181 | 18.1452 | 17.39% | 4.35% | 0.00% / 0.00% | 76.27 | 670.40 |
| GLU-Net+SA/CA | 10.9398 | 16.13% | 13.4016 | 13.9648 | 30.43% | 0.00% | 0.00% / 0.00% | 77.57 | 735.80 |
| CRFT官方权重 | 未运行：缺权重 | — | — | — | — | — | — | — | — |

共同有效像素 5,413,286，共同coarse像素 5,103。
GPU为RTX3060 Laptop 6GB，batch1、3次预热、每对5次同步前向中位数；包括GPU模型预处理和最终插值。
这是新的统一计时协议，不能和旧A4000时间直接比较。硬件/batch差异也造成与服务器val记录的轻微数值差别；没有覆盖历史结果。

权重SHA256：

- 基础：`2eac424ee4c9998cafed7ca60e7181d9f54adcee7e8e71250c5f2c7c0c9e50b1`
- baseline ep18：`04b11ad4688054f0b462c643efd70647638dbe796e098015819ee7404066178d`
- attention ep20：`91f7c0cfa41e9f83bd47ee10e92c8e129d0e4f5ccfa4d5b7625dfaa65e22f0ac`

## 逐图与失败案例

验证集18/23对最终AEPE改善；5对退化。完整逐图记录见[val_per_image.csv](val_per_image.csv)。
复现预览：`roadscene_runs/unified_val/previews/{baseline,attention}/{000013,000002,000004}.png`。

| 图像ID | 基线最终AEPE | SA/CA最终AEPE | 差值 |
|---|---:|---:|---:|
| 000013 | 2.4245 | 4.2198 | +1.7952 |
| 000001 | 4.0729 | 4.4083 | +0.3354 |
| 000004 | 5.9403 | 6.0566 | +0.1163 |
| 000006 | 5.5504 | 5.5894 | +0.0390 |
| 000020 | 5.2126 | 5.2355 | +0.0229 |

000013的coarse误差降低，但完整flow从约2.42退化到4.22；这对也使CMR@3从1/23变为0/23。
最大的最终改善包括000002、000019、000007。不能把平均提升解释成每对都变好或已实现亚像素配准。

## 有效位移范围（GT模长，512px）

| 位移桶 | 有效像素数 | 基线最终EPE | SA/CA最终EPE |
|---|---:|---:|---:|
| 0-8 | 445,228 | 3.9247 | 3.9887 |
| 8-16 | 466,996 | 5.4000 | 4.8938 |
| 16-32 | 482,599 | 8.2829 | 7.0555 |
| 32-64 | 1,065,618 | 9.2751 | 8.1166 |
| 64+ | 2,952,845 | 25.7850 | 19.1109 |

证据：大位移(64px+)像素改善较明显；0–8px桶略退化。大位移桶约占54.6%的有效像素，贡献约90%的总误差下降。
训练集176对中105对改善、71对退化，像素加权最终EPE 4.8552→4.7018；见[train_per_image.csv](train_per_image.csv)。
训练/验证的变形分布不同会影响均值，不能只看train平均loss。

## Top-1与最终EPE：证据及假设

证据：val相关图Top-5从32.33%→36.21%，GT平均rank从40.82→31.77；见[sa_ca_val_diagnostics.json](sa_ca_val_diagnostics.json)。
Top-1只看离散16×16 bin的argmax，flow decoder则使用整张相关分布，完整GLU还经过冻结的local阶段。
因此Top-1的小幅变化不限制连续flow误差的改善幅度；不同像素权重和错误的位移大小也会影响EPE。
假设：相关分布改善使粗flow进入更合适的局部细化范围。现有统计与此相容，但没有记录每级误差/warp特征来单独证明这个机制。
没有据此修改模型、超参数或测试协议。

## MIND与CRFT代码验证

- MIND两模态固定8ch描述子计算、[0,1]归一化、平坦区、强度仿射与反转、已知平移、256与16尺度对齐通过。
- B的零MIND gate相对已训练baseline/SA：coarse flow、correlation、完整最终flow最大差异全部0。
- A/B与A/B+SA零attention gate初始coarse输出完全一致；MIND初始参数也逐值一致。
- A输入适配、B门控都有有限非零梯度；编码器参数保持冻结。
- 六分支2train+1val、1epoch smoke全部前向/反向、选模保存、权重重载、完整CuPy推理和统一指标通过；图像顺序SHA一致。
- 修复了旧GLU全局Conv初始化覆盖A均值适配的问题，正式实现使用单独固定MIND初始化。
- CRFT源文件核对官方固定提交；target→source及native64 flow(4,-4)→512 flow(32,-32)通过。
- CRFT resize的两侧GPU结果与官方OpenCV在首对val逐值相同；64px分块/原版flow差异0。
- 额外512px原生随机前向通过：输出1×2×512×512有限，首次约138.82s、峰值3404.09MiB；不是正式速度或准确率。
- 未授权的test入口在读取数据之前被拒绝；新MIND训练不存在test选项。

审计原始记录：[mind_audit.json](mind_audit.json)、[crft_official_source.json](crft_official_source.json)、[512随机检查](crft_native512_random_code_check.json)。
正式MIND表现、其逐图失败/真实训练开销，和训练CRFT的准确率/速度/显存均尚无结果。

## 当前建议

当前保留SA/CA作为已验证控制。A/B已经能运行，但不能根据代码smoke选择加入MIND或转向DNS。
先在A4000执行六分支匹配20epoch；根据最终EPE≥5%改善、至少16/23对改善、位移桶/CMR退化限制、时间和显存≤1.25倍的val门槛决定。
完整门槛、单因素对比、CRFT64输入/512输出协议、预训练分布风险、服务器命令见[ROADSCENE_CRFT_MIND.md](../../ROADSCENE_CRFT_MIND.md)。

## 已锁定测试，仅留存

22对历史测试：coarse256 EPE 15.0437→11.5245，final512 EPE 19.3023→11.8212，Top-1 13.53%→14.97%。
原始文件只复制留存：[frozen_test_original.json](frozen_test_original.json)，未重新推理，也未用于上述模型/门槛选择。

## 实现改动

- `models/our_models/mind.py`：固定二维MIND、A输入投影、B门控投影。
- `models/our_models/GLUNet.py`：可选coarse MIND、正确梯度及独立初始化；local层未改变。
- `roadscene_coarse.py`：复用MIND coarse特征、加载新增参数、禁止训练时test选模。
- `crft_adapter.py`：官方外置模型严格加载、配置/源码SHA、64→512flow换算、可选512分块检查。
- `roadscene_metrics.py`、`roadscene_compare.py`：共同mask AEPE/CMR/原EPE、计时、逐图及失败记录、测试锁。
- `roadscene_mind.py`、`audit_mind.py`：六分支训练协议、初始化/梯度/坐标验证。
- `roadscene_diagnostics.py`：train/val逐图、Top-5/GT rank、位移分桶；此前本地实现本次一并发布。
- `coarse_dns.py`、`roadscene_dns.py`：此前二维DNS对照工具一并发布，默认20epoch对齐控制，未运行真实训练。
- `requirements.txt`、两份RoadScene说明、根README、实验JSON/CSV。

## 本地实际命令（工作目录E:\REG\GLU-Net）

```powershell
$env:CUPY_CACHE_DIR='E:\REG\cupy_cache'
& D:\Anaconda3\envs\vfbench\python.exe audit_mind.py --data-root E:\datasets\RoadScence --pretrained pre_trained_models\GLUNet_DPED_CityScape_ADE.pth --baseline-checkpoint roadscene_runs\sa_ca\best_baseline.pth --attention-checkpoint roadscene_runs\sa_ca\best_attention.pth --full-gate-check --crft-root E:\REG\CRFT-main --output roadscene_runs\mind_audit\final_audit.json
& D:\Anaconda3\envs\vfbench\python.exe roadscene_mind.py --data-root E:\datasets\RoadScence --pretrained pre_trained_models\GLUNet_DPED_CityScape_ADE.pth --output roadscene_runs\mind_codecheck_v2 --code-check
& D:\Anaconda3\envs\vfbench\python.exe roadscene_compare.py --data-root E:\datasets\RoadScence --pretrained pre_trained_models\GLUNet_DPED_CityScape_ADE.pth --baseline-checkpoint roadscene_runs\sa_ca\best_baseline.pth --attention-checkpoint roadscene_runs\sa_ca\best_attention.pth --split val --output roadscene_runs\unified_val
```

在固定MIND初始化修复前的smoke不作为实验结果。额外CRFT512随机检查见其审计JSON；本次没有正式MIND/DNS/CRFT训练命令的运行结果。

补充代码检查：CRFT原生64输入、共同512输出的完整适配链已通过；模拟matcher前缀权重严格保存重载后输出差异0，仍为随机权重、无准确率。见crft_native64_loader_check.json。
