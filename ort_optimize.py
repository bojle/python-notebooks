import sys
import onnxruntime as rt

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} model.onnx model_opt.onnx")

sess_options = rt.SessionOptions()
# Set graph optimization level
sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
# To enable model serialization after graph optimization set this
sess_options.optimized_model_filepath = sys.argv[2]
session = rt.InferenceSession(sys.argv[1], sess_options, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])

