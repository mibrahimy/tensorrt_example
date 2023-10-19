import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_path, trt_path, max_batch_size=1):
    # Initialize TensorRT builder and network.
    builder = trt.Builder(TRT_LOGGER)
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Load the ONNX model and parse it in order to populate the TensorRT network.
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # Create builder config
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    # config.flags |= 1 << (int)(trt.BuilderFlag.FP16) if builder.platform_has_fast_fp16 else 0
    
    profile = builder.create_optimization_profile()
    profile.set_shape("input", min=(1, 3, 224, 224), opt=(max_batch_size, 3, 224, 224), max=(max_batch_size, 3, 224, 224))  # Assuming the input name is "input"
    config.add_optimization_profile(profile)

    engine = builder.build_engine(network, config)
    
    # Save the TensorRT engine to file.
    with open(trt_path, 'wb') as f:
        f.write(engine.serialize())

    return engine

if __name__ == "__main__":
    onnx_file_path = "/home/mi/mi_workspace/upwork/jasim/onnx_models/efficientnet_epoch_10.onnx"
    trt_file_path = "/home/mi/mi_workspace/upwork/jasim/trt_models/efficientnet_epoch_10_fp32.engine"
    max_batch_size = 1

    engine = build_engine(onnx_file_path, trt_file_path, max_batch_size)
    if engine:
        print(f"Successfully converted {onnx_file_path} to {trt_file_path}")
    else:
        print(f"Conversion failed!")
