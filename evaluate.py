import torch
from torch.utils.data import DataLoader

from src.datasets.crack_dataset import CrackDataset
from src.models.unet import UNet
from src.metrics.segmentation_metrics import calculate_iou, calculate_dice

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    test_dataset = CrackDataset(
        image_dir="data/crack_segmentation_dataset/test/images",
        mask_dir="data/crack_segmentation_dataset/test/masks"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2
    )

    model = UNet(in_channels=3, num_classes=1)

    model.load_state_dict(
        torch.load(
            "outputs/best_unet_crack_bce_dice.pth",
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    total_iou = 0.0
    total_dice = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            outputs = model(images)

            iou = calculate_iou(outputs, masks)
            dice = calculate_dice(outputs, masks)

            total_iou += iou.item()
            total_dice += dice.item()
            num_batches += 1

            if batch_idx % 20 == 0:
                print(
                    f"Batch [{batch_idx}/{len(test_loader)}] "
                    f"IoU: {iou.item():.4f} "
                    f"Dice: {dice.item():.4f} "
                )

    mean_iou = total_iou / num_batches
    mean_dice = total_dice / num_batches

    print("="*40)
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Mean Dice: {mean_dice:.4f}")
    print("="*40)

if __name__ == "__main__":
    evaluate()
