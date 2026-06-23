import torch
from torch import nn
from torch.utils.data import DataLoader

from src.datasets.crack_dataset import CrackDataset
from src.models.unet import UNet
from src.losses.dice_loss import DiceLoss

def train_one_epoch(model, train_loader, bce_loss, dice_loss, optimizer, device, epoch, epochs, max_batches=None):
    model.train()

    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        outputs = model(images)
        bce = bce_loss(outputs, masks)
        dice = dice_loss(outputs, masks)
        loss = bce + dice

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"Batch [{batch_idx}/{len(train_loader)}] "
                f"BCE: {bce.item():.4f} "
                f"Dice: {dice.item():.4f} "
                f"Loss: {loss.item():.4f}"
            )
        if max_batches is not None and batch_idx >= max_batches:
            break

    avg_loss = total_loss / num_batches
    return avg_loss

def validate(model, val_loader, bce_loss, dice_loss, device, max_batches=None):
    model.eval()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            outputs = model(images)
            loss = bce_loss(outputs, masks) + dice_loss(outputs, masks)

            total_loss += loss.item()
            num_batches += 1

            if max_batches is not None and batch_idx >= max_batches:
                break

    avg_loss = total_loss / num_batches
    return avg_loss



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

    val_dataset = CrackDataset(
        image_dir="data/crack_segmentation_dataset/test/images",
        mask_dir="data/crack_segmentation_dataset/test/masks"
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2
    )

    model = UNet(in_channels=3, num_classes=1)
    model = model.to(device)

    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    epochs = 5
    max_batches = 20

    best_val_loss = float("inf")

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            bce_loss=bce_loss,
            dice_loss=dice_loss,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            epochs=epochs,
            max_batches=None
        )

        val_loss = validate(
            model=model,
            val_loader=val_loader,
            bce_loss=bce_loss,
            dice_loss=dice_loss,
            device=device,
            max_batches=None
        )

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f}"
        )


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"outputs/best_unet_crack_bce_dice.pth")
            print(f"Best model saved. Val Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()
