from .build_engine import EngineCalibrator, EngineBuilder
import argparse

parser = argparse.ArgumentParser(description="trt model inference")

parser.add_argument("--onnx_path", type=str, default=None, help="onnx model path")
parser.add_argument("--trt_path", type=str, default=None, help="trt model path")

builder = EngineBuilder(True, 8)

args = parser.parse_args()
onnx_model_path = args.onnx_path
tensorrt_model_path = args.trt_path

builder.create_network(onnx_model_path, 8, False)

builder.create_engine(
    tensorrt_model_path,
    precision="fp16"
)
