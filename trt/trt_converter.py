# Converter for converting pytorch model to tensorrt engine. Use as following:
# trt_converter.convert_2_trt_engine(
#         torch_model=torch_model,
#         output_dir='/tmp/trt',
#         min_inp_shape=(1, 1, 1, 3),
#         opt_inp_shape=(1, OPT_HEIGHT, OPT_WIDTH, 3),
#         max_inp_shape=(1, MAX_HEIGHT, MAX_WIDTH, 3),
#         inp_dynamic_axes=(1, 2),
#         out_dynamic_axes=(1, 2),
#         opset_version=9
#     )

import os
import torch
import pycuda.driver as cuda
import tensorrt as trt

ONNX_SAVER_NAME = 'model.onnx'
TRT_SAVER_NAME = 'model.trt'


def convert_2_trt_engine(torch_model, output_dir, min_inp_shape, opt_inp_shape, max_inp_shape, inp_dynamic_axes,
                         out_dynamic_axes, opset_version=11):
    """Convert torch model to tensorrt engine.

    Args:
        torch_model: Torch model.
        output_dir: Dir to save tensorrt engine.
        min_inp_shape: Minimum input shape. Such as [1, 3, 1, 1].
        opt_inp_shape: Optimized input shape. Such as [1, 3, 256, 256].
        max_inp_shape: Maximum input shape. Such as [1, 3, 2048, 2048].
        inp_dynamic_axes: Dynamic axes for input. Such as [2, 3] mean height and width are dynamic if input is picture.
        out_dynamic_axes: Dynamic axes for output. Such as [2, 3] mean height and width are dynamic if output is picture.
        opset_version: Opset version for onnx.

    Returns:
        If success.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convert torch model to onnx model firstly and save.
    convert_2_onnx_model(torch_model, opt_inp_shape, inp_dynamic_axes, out_dynamic_axes, output_dir, opset_version)
    # Convert onnx model to tensorrt engine and save.
    onnx_2_trt_engine(min_inp_shape, opt_inp_shape, max_inp_shape, output_dir)


def convert_2_onnx_model(torch_model, dummy_shape, inp_dynamic_axes, out_dynamic_axes, output_dir, opset_version=11):
    """Convert to onnx model.

    Args:
        torch_model: Torch model.
        output_dir: Dir to save tensorrt engine.
        dummy_shape: Dummy input shape. Such as (1, 3, 256, 256).
        inp_dynamic_axes: Dynamic axes for input. Such as (2, 3) mean height and width are dynamic if input is picture.
        out_dynamic_axes: Dynamic axes for output. Such as (2, 3) mean height and width are dynamic if output is picture.
        opset_version: Opset version for onnx.

    Returns:

    """
    print('Transfer torch model to onnx model...')
    onnx_model_path = os.path.join(output_dir, ONNX_SAVER_NAME)
    torch.onnx.export(torch_model,
                      torch.randn(*dummy_shape, device='cuda'),
                      onnx_model_path,
                      verbose=True,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': list(inp_dynamic_axes), 'output': list(out_dynamic_axes)},
                      opset_version=opset_version)
    return True


class MyInt8Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, inp_max_shape, inp_datas, output_dir):
        """Constructor for MyInt8Calibrator.

        Args:
            inp_max_shape: Input max shape.
            inp_datas: A iterator of some typical input data for current model.
            output_dir: Output dir to save cache.
        """
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.device_input = cuda.mem_alloc(trt.volume(inp_max_shape) * 4)
        self.batches = inp_datas
        self.cache_file = os.path.join(output_dir, '/calibrator_cache')

    def get_algorithm(self):
        return trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2

    def get_batch(self, names):
        try:
            # Assume self.batches is a generator that provides batch data.
            data = next(self.batches)
            # Assume that self.device_input is a device buffer allocated by the constructor.
            cuda.memcpy_htod(self.device_input, data)
            return [int(self.device_input)]
        except StopIteration:
            # When we're out of batches, we return either [] or None.
            # This signals to TensorRT that there is no calibration data remaining.
            return None

    def get_batch_size(self):
        return 1

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def onnx_2_trt_engine(min_inp_shape, opt_inp_shape, max_inp_shape, output_dir):
    """Convert onnx model to tensorrt engine.

    Args:
        min_inp_shape: Minimum input shape. Such as (1, 3, 1, 1).
        opt_inp_shape: Optimized input shape. Such as (1, 3, 256, 256).
        max_inp_shape: Maximum input shape. Such as (1, 3, 2048, 2048).
        output_dir: Dir to save tensorrt engine.

    Returns:
        If success.
    """
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    print('platform_has_tf32: {}, platform_has_fast_fp16: {}, platform_has_fast_int8: {}'
          .format(builder.platform_has_tf32, builder.platform_has_fast_fp16, builder.platform_has_fast_int8))

    # trt parse onnx model
    print('Parsing onnx model...')
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    success = parser.parse_from_file(os.path.join(output_dir, ONNX_SAVER_NAME))
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))
    if not success:
        print('parse onnx model failed.')
        return False
    print('Parse onnx model successfully.')

    print('Creating trt serialized engining...')
    # build trt config
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    config.set_flag(trt.BuilderFlag.TF32)

    # 一些量化手段
    # config.set_flag(trt.BuilderFlag.FP16)  # fp16量化
    # config.set_flag(trt.BuilderFlag.INT8)  # int8量化
    # config.set_quantization_flag(trt.QuantizationFlag.CALIBRATE_BEFORE_FUSION)
    # int8量化的校准器，提供一些代表性输入，方便trt量化时计算激活函数范围用于量化
    # config.int8_calibrator = MyInt8Calibrator(max_inp_shape, iter([]), output_dir)

    # dynamic shape
    op_profile = builder.create_optimization_profile()
    op_profile.set_shape(network.get_input(0).name, min=trt.Dims(list(min_inp_shape)),
                         opt=trt.Dims(list(opt_inp_shape)),
                         max=trt.Dims(list(max_inp_shape)))
    config.add_optimization_profile(op_profile)

    # save serialized engine
    serialized_engine = builder.build_serialized_network(network, config)
    trt_model_path = os.path.join(output_dir, TRT_SAVER_NAME)
    with open(trt_model_path, 'wb') as f:
        f.write(serialized_engine)

    print('Trt engine had been saved into {}'.format(trt_model_path))
    return True
