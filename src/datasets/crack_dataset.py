import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class CrackDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.image_names = sorted(os.listdir(image_dir))

        self.image_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        self.mask_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]

        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, image_name)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.image_transform(image)
        mask = self.mask_transform(mask)

        return {
            "image": image,
            "mask": mask
        }

