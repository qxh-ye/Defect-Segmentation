import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, masks):
        probs = torch.sigmoid(logits)

        probs = probs.view(-1)  # [B,1,H,W] -> [B*H*W]
        masks = masks.view(-1)

        intersection = (probs * masks).sum()

        dice = (2 * intersection + self.eps) / (probs.sum() + masks.sum() + self.eps)

        loss = 1 - dice

        return loss