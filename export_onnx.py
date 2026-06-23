import os
import torch

from src.models.unet import UNet

def export_onnx():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model_path = "outputs/best_unet_crack_bce_dice.pth"
    onnx_save_path = "outputs/onnxs/unet_crack.onnx"

    os.makedirs("outputs/onnxs", exist_ok=True)

    model = UNet(in_channels=3, num_classes=1)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    dummy_input = torch.randn(
        1,
        3,
        448,
        448,
        device=device
    )

    torch.onnx.export(
        model,
        dummy_input,
        onnx_save_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {
                0: "batch_size"
            },
            "output": {
                0: "batch_size"
            }
        }
    )

    print(f"ONNX model exported to: {onnx_save_path}")

if __name__ == "__main__":
    export_onnx()













