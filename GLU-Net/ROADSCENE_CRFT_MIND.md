# RoadScene：CRFT 独立对照与固定 MIND 消融

## 状态与边界

保留原有 22 对测试结果。这轮开发、失败分析、模型选择只访问 train/val。
原始架构基线指 **GLU-Net 架构 + RoadScene decoder4 微调**，不是未微调的基础权重。
原有基础权重和两份最佳 SA/CA 实验权重都不覆盖。

本地使用已有 vfbench：PyTorch 2.4.0+cu121、torchvision 0.19.0+cu121，
RTX 3060 Laptop 6GB。没有重新安装依赖。CRFT 核心适配器额外只需 einops；
requirements.txt 已加入。没有导入官方 Lightning/yacs 训练入口。
正式 MIND/DNS 训练按用户要求交给 A4000；本地只运行已有权重评测及小规模代码检查。

实际数值、逐图 CSV、运行代价见 `experiments/crft_mind/RESULTS.md` 和同目录 JSON。
本地少量配准预览保存在 `roadscene_runs/unified_val/previews/`；列顺序为
可见光、原 IR、预测 flow 后的 IR、GT flow EPE 热图（40px 封顶，无效区域黑色）。

## 一、来源审计及统一评测

来源：[CRFT 论文 §4.1](https://arxiv.org/html/2604.05689)、
[官方代码](https://github.com/NEU-Liuxuecong/CRFT)。CVF 网页本次访问返回 403，
论文内容依据作者链接的 arXiv 全文；没有用论文表格数值替代本地实验。

### 指标的确切含义

设某对图像 i 的有效像素集合为 V_i，e_i(p)=||F_pred(p)-F_gt(p)||_2。

- 原有最终 flow EPE：`sum_i sum_V e_i / sum_i |V_i|`，按有效像素加权。
- 每对 AEPE_i：`sum_V e_i / |V_i|`；新增 AEPE：所有图像对 AEPE_i 的算术平均。
- **CMR(t)=100 × count(AEPE_i < t) / 图像对总数**。单位是图像对，严格小于，
  不是像素命中率，也不是相关图 Top-1。
- 官方 `test_epoch_end` 阈值完整列表：5,4,3,2,1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1。
  论文表格使用 3、1、0.7px。本工程所有这些阈值均为 **512×512 像素单位**。
- `pixel_success_percent_auxiliary` 单独记录像素 EPE<t 的比例，不能称为论文 CMR。
- 粗尺度 EPE 保留 256px 单位；相关图 Top-1 是 GLU 16×16 查询对应的最近 source bin。

### 官方实现与本文工程的差别

`src/lightning/lightning_crft.py` 的 AEPE 在所有像素上计算，没有 GT 有效掩码；
若没有任何像素满足 sr 阈值，会将该图像 AEPE 写为 inf，聚合时又过滤 inf。
这会改变 CMR 的分母。`sr` 是像素命中比例，最后打印的多阈值统计才基于每对 AEPE。
`configs/data/roadscence_512.py` 尽管文件名有 512，实际 `RS_IMG_RESIZE=64`，
`SR_THRESHOLD=10000.0`。不能把该 sr 当成 CMR@3px。
官方 loader 对找不到 GT 的情形还会生成零 flow；本工程严格拒绝缺 GT。
本地16份核心/评测源文件已核对，与官方提交
`5ce912f55fea89cb3159205f38c24fd0e6a1fa21` 内容一致（忽略换行符）。

统一评测使用相同 GT 有效掩码，保留所有图像对，不排除失败图像。
因此这是 **按论文 AEPE/CMR 数学定义、加入共同有效掩码的统一协议**，
不是对官方旧评测入口的逐字复现。论文采用的变换、patch、尺度及失败过滤也未与本数据逐一证实相同，
禁止将作者表格 AEPE/CMR 与本工程结果放入同一比较列。

### 图像、flow 和坐标

所有模型直接读取 `RoadScenePairs` 的同一 512×512 uint8 RGB 图像对和 `.flo`。
visible 为 target/image0，warped IR 为 source/image1。
GT/pred flow 都是 target→source，warp 使用 `source(x+u,y+v)`。
共同 mask 为有限 GT 且 GT 映射落在 source 图像内；不会根据模型预测删除困难像素。
GLU 最终 flow1 插值到 512，不再乘尺度因子；其数值已是原图像素。
主对照固定使用CRFT官方64×64输入；`flow_f_full` 是64px图像单位，
官方模块已从8×8coarse网格插值并乘8。
随后完整flow双线性插值到512，并按横纵尺寸比各乘8，才得到512px单位；
这两个8分别对应8→64和64→512，不能重复或漏乘。
CRFT coarse flow 为8×8、以coarse网格像素计；统一coarse EPE时area到16×16再乘32，
转为256px单位，同GLU coarse GT/mask比较。512原生代码检查中的coarse则为64×64。
CRFT native correlation是8×8匹配，不等同GLU 16×16 bin；统一表中其Top-1为缺项。

模型输入内容相同，内部预处理保留各自权重所需的规范：GLU 为 ImageNet 标准化、
256 area resize 后 byte 量化；CRFT为bilinear resize到64、uint8舍入，
再做官方RGB每通道mean/std标准化。两侧512→64的GPU准备与OpenCV INTER_LINEAR
已在首对val上验证逐值相同。CRFT输出升到512使用half-pixel双线性坐标与向量尺度换算。
没有向 CRFT 输入已经过 GLU 标准化的图像，也没有将 RGB 改成灰度。

### 计时及复核

batch=1、float32、不开 autocast，关闭 cuDNN benchmark、开启 deterministic。
每个模型先预热3次，每对同步计时5次取中位数，再报告逐对中位数的平均。
计时包括 GPU 上模型预处理、完整前向、最终 flow 插值；排除磁盘读取、CPU→GPU、GT 与指标。
峰值为 `max_memory_allocated`，包含模型和 GPU 图像，不是显卡总占用或 reserved memory。
每个模型单独驻留 GPU；不同 GPU 的速度不能直接比较。
保存图像/GT 内容 SHA256 manifest、基础及模块权重 SHA256、评测与模型源文件哈希。

### CRFT 的现有阻塞与计算代价

本地没有 CRFT_RoadScene.ckpt。官方 README 列出权重名，未找到可用下载链接，
用户也确认没有。因此 **没有训练 CRFT 的准确率结果**。
需要获得作者权重及训练划分/选模来源，或另行执行 train-only/val-selection 的 CRFT 训练。
不能以随机权重替代正式对照，也不能将 CRFT 的论文训练预算宣称为和 GLU 微调预算相同。

主对照的原生64输入在权重缺失时只进行了随机前向代码检查，没有准确率。
额外尝试512原生输入时，官方 `fine_process.WindowSelfAttention` 将全部滑动窗口摊成一个长序列做全局 attention。
512 输入的第一次 score matrix 即尝试分配87.5GiB，之后序列更长。
针对该额外512检查，`crft_adapter.py`在运行时对四个fine attention模块按query行分块64行，
每行仍对 **所有 key** 做 softmax；权重、尺度、位置编码、输出定义不变，float32 不变。
没有改成局部窗口，也没有添加任何配准模块；官方源码保持外置。
主对照64输入保留官方未分块attention，不额外引入分块运行开销。
该适配与官方未分块版本在64×64随机权重前向下 flow 最大差异0，
512随机前向输出有限且为1×2×512×512。它只降低显存，不降低二次计算量。
此数值验证尚未覆盖缺失的真实 CRFT 权重。
实际512随机前向约138.82秒、峰值3404.09MiB，属于首次代码检查，
不能当作主对照64输入的速度或预热后正式速度，更不能与论文33ms并列。
主对照输入64依据官方配置预先固定，不依据val/test误差挑输入尺寸。

`roadscene_compare.py` 默认只允许 train/val。CRFT 的最终 test 入口要求一份成功的
23对验证报告，并检查同一权重、官方源码、运行适配器、协议和评测器哈希。
该入口只评 CRFT，不重新运行已有 GLU/SA 测试结果。MIND/DNS 当前没有 test 入口。

## 二、固定 MIND：定义与两种接入

参考：[原始 MIND 论文](https://pubmed.ncbi.nlm.nih.gov/22722056/)、
[作者公开 MIND-SSC 实现](https://gist.github.com/mattiaspaul/d314c22ac97d37c2cf05e99780bd54c4)。
此处是二维中心到邻域的 MIND，**不是**三维 MIND-SSC 或学习型 DNS 的复现。

两侧使用完全相同的固定计算：RGB→luma (0.299,0.587,0.114)，8个中心到邻居偏移，
顺序 N,S,W,E,NW,NE,SW,SE、半径1、固定3×3 sigma=1 Gaussian patch SSD。
V为四个轴向 patch SSD 的均值；每图独立做相对数值稳定 clamp。
`M=exp(-(D-min_r D)/V)`，每像素最大通道值1，平坦区输出全1，范围[0,1]。
描述子没有学习参数、没有训练集均值、两模态不共享跨图像统计。

实际模型在同一个256×256、byte量化后的 RGB 上重算 MIND，再用 area 对齐16×16。
MIND通道不是flow，缩放时不乘2或16。
“在512计算再池化”与“缩到256再计算”不等价：合成检查平均绝对差约0.04496，
所以本工程固定使用后者，两侧一致，不混用两条路径。

| 接入 | 最粗尺度路径 | 训练参数 | 初始化 |
|---|---|---|---|
| A | MIND(8ch)→共享1×1输入适配(3ch)→预训练VGG→可选SA/CA→global corr | 输入适配+decoder4+可选SA/CA | 适配为8通道均值，各SA门控0 |
| B | 原图→预训练VGG；MIND→area16→1×1轻投影→gate残差→可选SA/CA→global corr | 投影/gate+decoder4+可选SA/CA | MIND gate0，各SA门控0 |

**A仅替换最粗特征的编码器输入。** 256 branch的32×32局部特征和原图local pyramid
继续由原图提取；避免同时改变粗匹配和所有local层。完整前向中的A会多做一次粗编码，计时代价包含它。
编码器参数冻结，但A保留输入适配层经冻结编码器的梯度；不能用no_grad包住A整条编码。
B的投影为8→64→512，只在16×16融入，后面的local阶段不修改。
MIND在GLU全局Conv初始化之后构造，独立固定初始化seed2026；A的均值适配不会被旧初始化覆盖，
有无SA/CA的两个分支使用同一MIND初始参数。

### A 的预训练分布风险

MIND的通道是局部自相似，不是自然图像颜色或亮度。
A把8通道压成3通道后，虽沿用同一VGG基础权重，输入分布和通道语义已改变。
ImageNet标准化本身不能消除此差异；冻结VGG时只能由很小的输入适配学习调整。
这可能降低区分性或损害 coarse decoder 已学的对应分布。
B保留原始预训练特征且从零残差开始，分布冲击较小；这是结构上的理由，尚不是优于A的实验结论。

### 单因素与预算

`roadscene_mind.py` 重跑6个匹配分支：baseline、attention、mind_a、mind_a_attention、
mind_b、mind_b_attention。均从同一基础GLU权重开始，不接着最佳SA模型训练。
默认20epochs、batch2、AdamW lr1e-4、seed2026、loss=coarse EPE+2×corr CE。
每个分支同一训练图像顺序（逐epoch SHA验证），同一验证集、同一最佳 coarse EPE 选模规则。
训练预算按图像/更新/epoch定义，模块带来的 FLOPs 和显存差别另行记录。
原有历史 SA/CA 权重仅作已固定的参考，不冒充这次重新匹配训练的控制。

- baseline→A/B：在无attention结构中检查 MIND 的作用。
- attention→A+attention/B+attention：在同一SA/CA结构中只改变MIND接入。
- A→A+attention、B→B+attention：同一MIND路径中只检查SA/CA。
- 按请求也输出 A/B vs SA/CA 的直接差值，但这同时改变输入和attention两个因素，
  不能作为单因素因果证据。

不堆叠MIND与DNS；没有加入新细化网络。验证图像逐图差异、位移分桶及
“coarse改善而final退化”的case自动写到`paired_differences.json`。

## 三、在新训练前确定的继续判据

以下是工程决策门槛，不是统计显著性或论文定论，只看23对val：

1. 与**相同预算重新训练的SA/CA控制**相比，最终有效像素加权EPE至少降低5%，
   按图像对平均AEPE也下降，至少16/23对最终AEPE改善。
2. 固定GT位移桶 `[0,8),[8,16),[16,32),[32,64),[64,+inf)`，单位512px。
   不接受某桶最终EPE超过控制5%的退化；CMR@5px下降超过1对也不继续。
   完整CMR曲线照实保留，不能因均值改善而声称精确匹配率全部提升。
3. 同GPU相同计时协议下，完整推理时间及峰值allocated显存均不超过SA/CA的1.25倍。
   训练显存和耗时另行报告。
4. 如果只改善coarse、最终flow变差，或改善仅来自少数图像，停该路径，不继续加模块。
   A/B都未满足门槛时先保留SA/CA；如有固定自相似不足以区分对应点的验证证据，再研究DNS。

DNS下一步必须先做 baseline→DNS、DNS→DNS+对应点对比损失的独立比较，
也用20epoch匹配控制和同一评测器。现有`roadscene_dns.py`为二维DSIR启发实现，
其GT对应点对比不是论文同图强度增强目标；没有真实DNS训练结果。
不能凭MIND尚未训练就宣布应转向DNS。

## 四、实际运行与服务器命令

服务器Windows CMD，数据在G:\cxj\RoadScence，仓库在G:\cxj\REG。

```bat
cd /d G:\cxj\REG
git pull --ff-only origin main
cd GLU-Net
conda activate glunet
```

先使用已有环境，只有缺少einops才安装，不重复安装Torch/CUDA等：

```bat
python -c "import torch, torchvision, cupy, einops; print(torch.__version__,torch.version.cuda,einops.__version__)"
```

历史权重在服务器原来的`roadscene_runs\sa_ca`；基础GLU权重已经随仓库提供。
数据无需复制到GLU-Net内部。代码检查及已有对照验证：

```bat
python audit_mind.py --data-root "G:\cxj\RoadScence" --pretrained "pre_trained_models\GLUNet_DPED_CityScape_ADE.pth" --baseline-checkpoint "roadscene_runs\sa_ca\best_baseline.pth" --attention-checkpoint "roadscene_runs\sa_ca\best_attention.pth" --full-gate-check --output "roadscene_runs\mind_audit\audit.json"
python roadscene_compare.py --data-root "G:\cxj\RoadScence" --pretrained "pre_trained_models\GLUNet_DPED_CityScape_ADE.pth" --baseline-checkpoint "roadscene_runs\sa_ca\best_baseline.pth" --attention-checkpoint "roadscene_runs\sa_ca\best_attention.pth" --split val --output "roadscene_runs\unified_val"
```

正式六分支训练，再用统一评测器复核匹配控制（会输出逐图差异、CMR、位移桶及预览）：

```bat
python roadscene_mind.py --data-root "G:\cxj\RoadScence" --pretrained "pre_trained_models\GLUNet_DPED_CityScape_ADE.pth" --epochs 20 --batch-size 2 --lr 0.0001 --output "roadscene_runs\mind_matched_20"
python roadscene_compare.py --data-root "G:\cxj\RoadScence" --pretrained "pre_trained_models\GLUNet_DPED_CityScape_ADE.pth" --experiment-dir "roadscene_runs\mind_matched_20" --split val --output "roadscene_runs\mind_matched_20\val_review"
```

训练输出目录必须全新，脚本拒绝覆盖已有最佳权重。
`--code-check`只用2个train+1个val跑1epoch；输出明示不是实验结果，统一评测拒绝将其当作正式权重。

需要CRFT时单独拉官方引用源码（仍外置，不往GLU模型堆CRFT模块）：

```bat
cd /d G:\cxj\REG
git clone https://github.com/NEU-Liuxuecong/CRFT.git CRFT-main
git -C CRFT-main checkout 5ce912f55fea89cb3159205f38c24fd0e6a1fa21
cd GLU-Net
python audit_mind.py --data-root "G:\cxj\RoadScence" --pretrained "pre_trained_models\GLUNet_DPED_CityScape_ADE.pth" --crft-root "..\CRFT-main" --output "roadscene_runs\crft_code_audit.json"
```

如需重复慢速512随机前向，上一命令另加`--crft-forward-check`。这不产出CRFT准确率。
取得真实权重后，先验证（路径`CRFT_RoadScene.ckpt`当前尚不存在）：

```bat
python roadscene_compare.py --data-root "G:\cxj\RoadScence" --pretrained "pre_trained_models\GLUNet_DPED_CityScape_ADE.pth" --baseline-checkpoint "roadscene_runs\sa_ca\best_baseline.pth" --attention-checkpoint "roadscene_runs\sa_ca\best_attention.pth" --crft-root "..\CRFT-main" --crft-checkpoint "..\CRFT-main\checkpoints\CRFT_RoadScene.ckpt" --split val --output "roadscene_runs\crft_val"
```

确认权重的训练/验证来源和上面固定协议后，只对CRFT最终test：

```bat
python roadscene_compare.py --data-root "G:\cxj\RoadScence" --pretrained "pre_trained_models\GLUNet_DPED_CityScape_ADE.pth" --crft-root "..\CRFT-main" --crft-checkpoint "..\CRFT-main\checkpoints\CRFT_RoadScene.ckpt" --split test --final-crft-test --validated-report "roadscene_runs\crft_val\comparison.json" --output "roadscene_runs\crft_test_final"
```

新MIND验证不满足门槛时不运行新test。已有22对测试只作为已锁定历史结果保存。
