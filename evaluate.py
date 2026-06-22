from src.metrics.segmentation_metrics import calculate_iou, calculate_dice
import torch

logits = torch.randn(2, 1, 448, 448)
masks = torch.randint(0, 2, (2, 1, 448, 448)).float()

iou = calculate_iou(logits, masks)
dice = calculate_dice(logits, masks)

print(iou)
print(dice)