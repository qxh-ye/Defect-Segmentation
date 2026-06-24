from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import onnxruntime as ort
import torch

from src.models.unet import UNet


PYTORCH_MODEL_PATH = Path("outputs/best_unet_crack_bce_dice.pth")
ONNX_MODEL_PATH = Path("outputs/onnxs/unet_crack.onnx")
IMAGE_PATH = Path("data/crack_segmentation_dataset/test/images/CFD_014.jpg")
INPUT_SIZE = (448, 448)
WARMUP_TIMES = 10
BENCHMARK_TIMES = 50


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


def benchmark_pytorch(input_array: np.ndarray) -> float:
    """测试 PyTorch U-Net 的平均推理耗时，单位为毫秒。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=3, num_classes=1)
    state_dict = torch.load(PYTORCH_MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    input_tensor = torch.from_numpy(input_array).to(device)

    # 预热模型，排除首次执行时的初始化开销。
    with torch.no_grad():
        for _ in range(WARMUP_TIMES):
            model(input_tensor)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = perf_counter()
        for _ in range(BENCHMARK_TIMES):
            model(input_tensor)

        # CUDA 运算是异步执行的，停止计时前必须等待全部任务完成。
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_time = perf_counter() - start_time

    return elapsed_time * 1000.0 / BENCHMARK_TIMES


def benchmark_onnxruntime(input_array: np.ndarray) -> float:
    """使用 CPUExecutionProvider 测试 ONNXRuntime 平均推理耗时。"""
    session = ort.InferenceSession(
        str(ONNX_MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    model_input = {input_name: input_array}

    # 先执行固定次数预热，再进行正式计时。
    for _ in range(WARMUP_TIMES):
        session.run(None, model_input)

    start_time = perf_counter()
    for _ in range(BENCHMARK_TIMES):
        session.run(None, model_input)
    elapsed_time = perf_counter() - start_time

    return elapsed_time * 1000.0 / BENCHMARK_TIMES


def main() -> None:
    # PyTorch 与 ONNXRuntime 使用完全相同的预处理结果。
    input_array = preprocess_image(IMAGE_PATH)

    pytorch_avg_time = benchmark_pytorch(input_array)
    onnx_avg_time = benchmark_onnxruntime(input_array)
    speedup = pytorch_avg_time / onnx_avg_time

    print(f"PyTorch Avg Inference Time: {pytorch_avg_time:.3f} ms")
    print(f"ONNXRuntime Avg Inference Time: {onnx_avg_time:.3f} ms")
    print(f"Speedup: {speedup:.3f}x")


if __name__ == "__main__":
    main()
