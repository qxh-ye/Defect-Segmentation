# 工业裂缝缺陷智能分割系统

基于 PyTorch + U-Net 的工业裂缝缺陷语义分割项目

## 项目简介

本项目基于 PyTorch 实现 U-Net 语义分割模型，用于工业裂缝缺陷的像素级检测。项目围绕工业视觉缺陷检测场景，完成从数据读取、模型训练、验证评估到推理可视化的完整流程。

当前系统已实现：

* 数据加载
* 模型训练
* 验证评估
* 最优模型保存
* 推理预测
* 结果可视化

项目形成了较完整的计算机视觉工程闭环，可用于工业裂缝分割、缺陷检测算法验证以及后续部署扩展。

## 项目亮点

* 自定义 CrackDataset
* U-Net 分割模型
* BCE + DiceLoss 联合训练
* GPU 训练
* Validation 验证流程
* Best Model 自动保存
* IoU 与 Dice 评估
* 推理可视化
* 工程化目录结构
* 支持后续 ONNX 部署扩展

## 技术栈

| 模块 | 技术 |
| --- | --- |
| 开发语言 | Python |
| 深度学习框架 | PyTorch |
| 图像处理 | OpenCV |
| 数值计算 | NumPy |
| 分割模型 | U-Net |
| 分类损失 | BCEWithLogitsLoss |
| 区域损失 | DiceLoss |
| 评估指标 | IoU |
| 评估指标 | Dice |

## 项目结构

```text
Defect_Segmentation/
├── configs/
│   └── 项目配置文件目录
├── data/
│   └── Crack Segmentation Dataset 数据目录
├── outputs/
│   ├── preds/
│   │   └── 模型预测 Mask 输出
│   ├── comparisons/
│   │   └── 原图 / 标注 / 预测结果可视化对比图
│   └── best_unet_crack_bce_dice.pth
│       └── 验证集表现最优的 U-Net 权重文件
├── src/
│   ├── datasets/
│   │   └── crack_dataset.py
│   │       └── 自定义裂缝分割数据集加载模块
│   ├── losses/
│   │   ├── bce_loss.py
│   │   └── dice_loss.py
│   │       └── BCE Loss 与 Dice Loss 实现
│   ├── metrics/
│   │   └── segmentation_metrics.py
│   │       └── IoU 与 Dice 分割指标计算
│   ├── models/
│   │   └── unet.py
│   │       └── U-Net 语义分割模型
│   └── utils/
│       └── 图像处理、路径管理与可视化工具函数
├── train.py
│   └── 模型训练入口
├── evaluate.py
│   └── 模型评估入口
├── predict.py
│   └── 模型推理与可视化入口
├── requirements.txt
│   └── Python 依赖环境
└── README.md
```

## 模型训练流程

训练流程围绕裂缝图像与像素级 Mask 标注展开，通过 U-Net 输出单通道预测结果，并使用 BCE + DiceLoss 进行联合优化。

```text
Crack Segmentation Dataset
        ↓
DataLoader
        ↓
U-Net
        ↓
BCE + DiceLoss
        ↓
Validation
        ↓
Best Model
```

训练阶段主要步骤：

1. 通过 `CrackDataset` 读取裂缝图像与对应 Mask。
2. 使用 `DataLoader` 构建批量训练数据。
3. 将图像输入 U-Net，输出裂缝区域预测 logits。
4. 使用 `BCEWithLogitsLoss` 进行像素级二分类优化。
5. 使用 `DiceLoss` 强化预测区域与真实 Mask 的重叠约束。
6. 在 Validation 集上计算 IoU 与 Dice。
7. 根据验证集表现自动保存 Best Model。

## 实验结果

| Model | Loss | Mean IoU | Mean Dice |
| --- | --- | ---: | ---: |
| U-Net | BCE + DiceLoss | 0.4652 | 0.5992 |

实验结果基于测试集评估获得，当前结果说明模型已经具备对裂缝区域进行像素级分割的能力，能够在细长裂缝、局部断裂和复杂背景场景下输出较稳定的预测 Mask。后续可结合更强数据增强、更深层特征提取网络和部署推理优化进一步提升性能。

## 可视化结果

### 预测结果展示

![result](outputs/comparisons/comparison_CFD_014_t03_bce_dice_final.png)

可视化结果展示了原始图像、真实 Mask 与模型预测 Mask 的对比效果，便于快速分析模型在裂缝边界、细小缺陷和复杂纹理背景下的分割表现。

## 工程实现亮点

1. 自定义 Dataset：基于 `CrackDataset` 封装图像与 Mask 的读取、预处理和张量转换逻辑，使训练、验证和测试流程保持统一的数据接口。
2. Train / Validation / Test 完整流程：项目拆分训练、评估和推理入口，覆盖模型开发中常见的训练验证、离线评估和单独预测场景。
3. Best Model 机制：训练过程中根据验证集指标保存最优权重，避免只依赖最后一个 epoch 的模型结果。
4. GPU 训练：支持 CUDA 加速训练，适配常见深度学习开发环境。
5. IoU 与 Dice 评估：实现语义分割任务常用指标，能够从区域重叠和像素分割质量两个角度评估模型效果。
6. 推理可视化：预测阶段自动生成 Mask 与对比图，便于定位模型误检、漏检和边界分割问题。
7. 工程化目录结构：按照 datasets、models、losses、metrics、utils 等模块组织代码，便于维护、扩展和部署集成。

## 后续优化方向

### 部署优化

* ONNX 导出
* ONNXRuntime 推理
* TensorRT 部署

### 模型优化

* 数据增强策略
* Attention U-Net
* U-Net++
* DeepLabV3+

### 工程优化

* Flask Web Demo
* Docker 部署
* 
## 简历项目描述

### 简历版项目描述

* 基于 PyTorch 从零实现 U-Net 语义分割模型，用于工业裂缝缺陷的像素级检测与 Mask 预测。
* 设计并实现自定义 `CrackDataset`，完成裂缝图像与标注 Mask 的读取、预处理和训练数据加载流程。
* 使用 `BCEWithLogitsLoss + DiceLoss` 联合优化分割模型，提升小目标裂缝区域和不平衡前景场景下的分割效果。
* 构建 Train / Validation / Test 工程流程，支持 GPU 训练、验证集评估、Best Model 自动保存和离线推理。
* 实现 IoU、Dice 分割指标评估与预测结果可视化，在 Crack Segmentation Dataset 上取得 Mean IoU 0.4652、Mean Dice 0.5992。

## 项目价值

本项目实现了工业裂缝缺陷分割任务中的关键工程环节，包括数据管理、模型训练、验证评估、模型保存、推理预测和部署准备，形成完整的计算机视觉工程闭环。

项目适合作为以下方向的工程实践展示：

* 计算机视觉
* 工业质检
* AI 应用工程
* 深度学习模型训练与部署
