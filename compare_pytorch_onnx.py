from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch

from src.models.unet import UNet


PYTORCH_MODEL_PATH = Path("outputs/best_unet_crack_bce_dice.pth")
ONNX_MODEL_PATH = Path("outputs/onnxs/unet_crack.onnx")
IMAGE_PATH = Path("data/crack_segmentation_dataset/test/images/CFD_014.jpg")
INPUT_SIZE = (448, 448)


def preprocess_image(image_path: Path) -> np.ndarray:
    """使用统一的 OpenCV 流程生成 float32 NCHW 输入。"""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"无法读取测试图片: {image_path}")

    image = cv2.resize(image, INPUT_SIZE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)

    return np.ascontiguousarray(image)


def run_pytorch(input_array: np.ndarray) -> np.ndarray:
    """加载 PyTorch U-Net，并返回未经激活函数处理的 logits。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=3, num_classes=1)
    state_dict = torch.load(PYTORCH_MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    input_tensor = torch.from_numpy(input_array).to(device)

    with torch.inference_mode():
        logits = model(input_tensor)

    return logits.cpu().numpy()


def run_onnx(input_array: np.ndarray) -> np.ndarray:
    """使用 ONNXRuntime 执行推理，并返回模型原始 logits。"""
    session = ort.InferenceSession(
        str(ONNX_MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: input_array})[0]

    return logits


def main() -> None:
    # 两个推理后端共享同一份预处理结果，避免输入差异影响精度对比。
    input_array = preprocess_image(IMAGE_PATH)

    pytorch_logits = run_pytorch(input_array)
    onnx_logits = run_onnx(input_array)

    if pytorch_logits.shape != onnx_logits.shape:
        raise ValueError(
            "PyTorch 与 ONNX 输出形状不一致: "
            f"{pytorch_logits.shape} != {onnx_logits.shape}"
        )

    absolute_difference = np.abs(pytorch_logits - onnx_logits)
    mean_abs_diff = float(np.mean(absolute_difference))
    max_abs_diff = float(np.max(absolute_difference))

    print(f"PyTorch output shape: {pytorch_logits.shape}")
    print(f"ONNX output shape: {onnx_logits.shape}")
    print(f"Mean Abs Diff: {mean_abs_diff:.10f}")
    print(f"Max Abs Diff: {max_abs_diff:.10f}")


if __name__ == "__main__":
    main()
