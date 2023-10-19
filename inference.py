import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import time

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(trt_path):
    with open(trt_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine):
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    return h_input, d_input, h_output, d_output

def do_inference(context, h_input, d_input, h_output, d_output):
    # Transfer input data to the GPU.
    cuda.memcpy_htod(d_input, h_input)
    # Set the input and output bindings for the engine.
    bindings = [int(d_input), int(d_output)]
    # Set the input shape for the engine.
    context.set_binding_shape(0, (1,3,224,224))
    # Run inference.
    context.execute_v2(bindings)
    # Transfer predictions back from the GPU.
    cuda.memcpy_dtoh(h_output, d_output)
    # Return the host output.
    return h_output

if __name__ == "__main__":
    trt_file_path = "/home/mi/mi_workspace/upwork/jasim/trt_models/efficientnet_epoch_10.engine"
    engine = load_engine(trt_file_path)

    h_input, d_input, h_output, d_output = allocate_buffers(engine)

    # Assuming input image data is stored in a NumPy array named input_image_data.
    # Ensure the input image data is the correct shape and datatype for your model.
    input_image_data = np.random.random_sample(engine.get_binding_shape(0)).astype(np.float32)
    np.copyto(h_input, input_image_data.ravel())

    with engine.create_execution_context() as context:
        # Timing the inference step
        start_time = time.time()
        output_data = do_inference(context, h_input, d_input, h_output, d_output)
        end_time = time.time()

        print(f"Inference took {end_time - start_time:.6f} seconds")
        print(f"Output data: {output_data}")
