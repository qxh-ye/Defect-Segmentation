from src.metrics.segmentation_metrics import calculate_iou
import torch

logits = torch.randn(2, 1, 448, 448)
masks = torch.randint(0, 2, (2, 1, 448, 448)).float()

iou = calculate_iou(logits, masks)

print(iou)