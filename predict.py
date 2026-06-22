import os
import cv2
import numpy as np
import torch
from torchvision import transforms

from src.models.unet import UNet

def predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    image_path = "data/crack_segmentation_dataset/test/images/CFD_014.jpg"
    model_path = "outputs/best_unet_crack.pth"
    save_path = "outputs/preds/pred_CFD_014.png"

    model = UNet(in_channels=3, num_classes=1)
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    model.to(device)
    model.eval()

    original = cv2.imread(image_path)

    if original is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    original = cv2.resize(original, (448, 448))
    image_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    transform = transforms.ToTensor()

    image_tensor = transform(image_rgb)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits)
        pred_mask = (probs > 0.4).float()

    pred_mask = pred_mask.squeeze().cpu().numpy()
    pred_mask = (pred_mask * 255).astype("uint8")

    os.makedirs("outputs/preds", exist_ok=True)
    cv2.imwrite(save_path, pred_mask)

    print(f"Prediction saved to: {save_path}")

    mask_path = "data/crack_segmentation_dataset/test/masks/CFD_014.jpg"
    gt_mask = cv2.imread(
        mask_path,
        cv2.IMREAD_GRAYSCALE
    )
    if gt_mask is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    gt_mask = cv2.resize(
        gt_mask,
        (448, 448)
    )

    gt_mask = cv2.cvtColor(
        gt_mask,
        cv2.COLOR_GRAY2BGR
    )

    pred_mask = cv2.cvtColor(
        pred_mask,
        cv2.COLOR_GRAY2BGR
    )

    comparison = np.hstack(
        [
            original,
            gt_mask,
            pred_mask
        ]
    )
    comparison_save_path = "outputs/comparisons/comparison_CFD_014_t04.png"

    os.makedirs("outputs/comparisons", exist_ok=True)

    cv2.imwrite(
        comparison_save_path,
        comparison
    )
    print(f"Comparison saved to: {comparison_save_path}")

if __name__ == "__main__":
    predict()