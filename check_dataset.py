from src.models.unet import Down
import torch

x = torch.randn(4, 64, 448, 448)

down = Down(64, 128)

out = down(x)

print(out.shape)