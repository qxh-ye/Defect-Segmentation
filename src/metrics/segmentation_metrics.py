import torch

def calculate_iou(logits, masks, threshold=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)

    preds = (probs > threshold).float()
    masks = masks.float()

    intersection = (preds * masks).sum()
    union = preds.sum() + masks.sum() - intersection

    iou = (intersection + eps) / (union + eps)

    return iou