import torch
from torch import nn
from torch.utils.data import DataLoader

from src.datasets.crack_dataset import CrackDataset
from src.models.unet import UNet

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_dataset = CrackDataset(
        image_dir="data/crack_segmentation_dataset/train/images",
        mask_dir="data/crack_segmentation_dataset/train/masks"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2
    )

    model = UNet(in_channels=3, num_classes=1)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    epochs = 1
    max_batches = 200

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            outputs = model(images)

            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}] "
                    f"Batch [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )
            if batch_idx >= max_batches:
                break

        avg_loss = total_loss / len(train_loader)

        print(f"Epoch [{epoch+1}/{epochs}] Avg Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "outputs/unet_crack.pth")
    print("Model saved to outputs/unet_crack.pth")

if __name__ == "__main__":
    train()
