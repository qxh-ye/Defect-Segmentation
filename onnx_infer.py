import os
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

ONNX_PATH = Path("outputs/onnxs/unet_crack.onnx")

session = ort.InferenceSession(
    str(ONNX_PATH),
    providers=["CPUExecutionProvider"]
)

INPUT_NAME = session.get_inputs()[0].name
OUTPUT_NAME = session.get_outputs()[0].name

def preprocess(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    original = cv2.resize(image, (448, 448))

    image_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    image_rgb = image_rgb.astype(np.float32) / 255.0

    image_tensor = np.transpose(image_rgb, (2, 0, 1))
    image_tensor = np.expand_dims(image_tensor, axis=0)

    return original, image_tensor

def postprocess(output, threshold=0.5):
    probs = 1 / (1 + np.exp(-output))

    pred_mask = (probs > threshold).astype(np.uint8)

    pred_mask = np.squeeze(pred_mask)

    pred_mask = (pred_mask * 255).astype(np.uint8)

    return pred_mask

def infer_image(image_path, threshold=0.5):
    """
        对单张图片进行 ONNX 推理，返回原图和预测Mask
        这个函数给 Flask 调用
    """
    original, image_tensor = preprocess(image_path)

    outputs = session.run(
        [OUTPUT_NAME],
        {
            INPUT_NAME: image_tensor
        }
    )

    pred_mask = postprocess(outputs[0], threshold=threshold)

    return original, pred_mask

def main():
    image_path = "data/crack_segmentation_dataset/test/images/CFD_014.jpg"
    mask_path = "data/crack_segmentation_dataset/test/masks/CFD_014.jpg"
    onnx_path = "outputs/onnxs/unet_crack.onnx"

    save_path = "outputs/onnx_preds/pred_CFD_014_onnx.png"
    comparison_path = "outputs/onnx_comparisons/comparison_CFD_014_onnx.png"

    os.makedirs("outputs/onnx_preds", exist_ok=True)
    os.makedirs("outputs/onnx_comparisons", exist_ok=True)

    original, image_tensor = preprocess(image_path)

    session = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    outputs = session.run(
        [output_name],
        {
            input_name: image_tensor
        }
    )

    pred_mask = postprocess(outputs[0], threshold=0.5)

    cv2.imwrite(save_path, pred_mask)

    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if gt_mask is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    gt_mask = cv2.resize(gt_mask, (448, 448))
    gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGR)

    pred_mask_color = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)

    comparison = np.hstack(
        [
            original,
            gt_mask,
            pred_mask_color
        ]
    )

    cv2.imwrite(comparison_path, comparison)

    print(f"ONNX prediction saved to: {save_path}")
    print(f"ONNX comparison saved to: {comparison_path}")

if __name__ == "__main__":
    main()


