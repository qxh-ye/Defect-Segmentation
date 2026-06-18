from src.datasets.crack_dataset import CrackDataset
from torch.utils.data import DataLoader

dataset = CrackDataset(
    "data/crack_segmentation_dataset/train/images",
    "data/crack_segmentation_dataset/train/masks"
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

for images, masks in loader:
    print(images.shape)
    print(masks.shape)
    break