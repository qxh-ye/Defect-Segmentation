import onnx

onnx_path = "outputs/onnxs/unet_crack.onnx"

model = onnx.load(onnx_path)

onnx.checker.check_model(model)

print("ONNX model is valid")