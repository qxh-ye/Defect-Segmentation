# Defect Segmentation Project

基于 PyTorch 的工业缺陷图像分割项目，计划实现数据集读取、U-Net 模型训练、Dice/IoU 指标评估、预测结果可视化及 GPU 训练流程。

## 技术栈

- Python
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- CUDA

## 项目结构

```text
Defect_Segmentation/
├── src/
│   ├── datasets/
│   ├── models/
│   ├── losses/
│   ├── metrics/
│   └── utils/
├── train.py
├── predict.py
├── evaluate.py
└── README.md