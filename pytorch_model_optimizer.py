# Pytorch模型做pure-trt优化并比较对应的性能
# pure-trt指的是使用pure tensorrt模型，使用的推理引擎是纯tensorrt引擎
#
# 环境：
# gpu: Nvidia GeForce 3060
# gpu driver: 470.57.02
# cuda: cuda_11.3
# cudnn: v8.1
# pytorch: torch-1.10.0+cu113 torchvision-0.11.1+cu113
# tensorrt: TensorRT-8.2.0.0

import os
import timeit
import urllib.request

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import torch
import torchvision
import onnxruntime as onnx_rt
from PIL import Image
from matplotlib import pyplot as plt

MODEL_PATH = '/tmp/resnet-50/'
ONNX_MODEL_PATH = MODEL_PATH + 'model.onnx'
TRT_MODEL_PATH = MODEL_PATH + 'model.trt'
MAX_BATCH_SIZE = 128

# resnet输出分类映射关系
SYNSET_URL = "".join(
    [
        "https://gist.githubusercontent.com/zhreshold/",
        "4d0b62f3d01426887599d4f7ede23ee5/raw/",
        "596b27d23537e5a1b5751d2b0481ef172f58b539/",
        "imagenet1000_clsid_to_human.txt",
    ]
)
SYNSET_NAME = "imagenet1000_clsid_to_human.txt"


def preprocess_input(img):
    """预处理图片数据"""
    image = np.float32(img) / 255.0
    image[:, :, ] -= (np.float32(0.485), np.float32(0.456), np.float32(0.406))
    image[:, :, ] /= (np.float32(0.229), np.float32(0.224), np.float32(0.225))
    image = image.transpose((0, 3, 1, 2))
    print('image shape: {}'.format(image.shape))
    return image


def load_data():
    """加载测试数据"""
    print('Loading image data...')
    # img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
    # urllib.request.urlretrieve(img_url, './cat.png')
    img = Image.open('./cat.png').resize((224, 224))
    # plt.imshow(img)
    # plt.show()
    img = np.array(img)[np.newaxis, :].astype("float32")
    data = preprocess_input(img)
    return data


def load_torch_model():
    """加载pytorch resnet-50模型"""
    print('Loading pytorch model...')
    model = torchvision.models.resnet50(pretrained=True).eval().cuda()
    print('Loaded pytorch resnet-50 model.')
    return model


def load_onnx_model(torch_model):
    """加载onnx模型"""
    # torch模型转换成onnx
    print('Transfer torch model to onnx model...')
    dummy_input = torch.randn(1, 3, 224, 224, device="cuda")
    input_names = ["input"]
    output_names = ["output"]
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    torch.onnx.export(torch_model, dummy_input, ONNX_MODEL_PATH, verbose=True,
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes={'input': [0], 'output': [0]})

    onnx_sess = onnx_rt.InferenceSession(ONNX_MODEL_PATH)
    return onnx_sess


def onnx_2_trt_engine_by_trtexec():
    """通过trtexec命令转换onnx模型到pure-tensorrt engine文件并保存"""
    # onnx执行tensorrt优化并保存引擎文件
    print('Apply tensorrt optimizing...')
    save_trt_engine_cmd = 'trtexec --onnx={} --saveEngine={} --minShapes=input:1x3x224x224' \
                          ' --maxShapes=input:{}x3x224x224 --optShapes=input:64x3x224x224 --best' \
                          ' --exportLayerInfo={}' \
        .format(ONNX_MODEL_PATH, TRT_MODEL_PATH, MAX_BATCH_SIZE, MODEL_PATH + 'layer_info.json')
    os.system(save_trt_engine_cmd)
    return True


class MyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.device_input = cuda.mem_alloc(1 * 3 * 224 * 224 * 4)
        self.batches = iter([np.ascontiguousarray(load_data())])  # 这里应该是一批典型输入的迭代器
        self.cache_file = MODEL_PATH + '/calibrator_cache'

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


def onnx_2_trt_engine_by_api():
    """通过trt api转换onnx模型到pure-tensorrt engine并保存"""
    logger = trt.Logger(trt.Logger.INFO)

    # trt解析onnx模型
    builder = trt.Builder(logger)
    print('platform_has_tf32: {}, platform_has_fast_fp16: {}, platform_has_fast_int8: {}'
          .format(builder.platform_has_tf32, builder.platform_has_fast_fp16, builder.platform_has_fast_int8))
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    success = parser.parse_from_file(ONNX_MODEL_PATH)
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))
    if not success:
        print('parse onnx model failed.')
        return False
    print('parse onnx model successfully.')

    print('Creating trt serialized engining...')
    # build trt config
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    # 混合精度
    config.set_flag(trt.BuilderFlag.TF32)
    config.set_flag(trt.BuilderFlag.FP16)
    # config.set_flag(trt.BuilderFlag.INT8)  # int8量化
    # config.set_quantization_flag(trt.QuantizationFlag.CALIBRATE_BEFORE_FUSION)
    # config.int8_calibrator = MyCalibrator()  # int8量化的校准器，提供一些代表性输入，方便trt量化时计算激活函数范围用于量化
    # 动态输入shape
    op_profile = builder.create_optimization_profile()
    op_profile.set_shape(network.get_input(0).name, min=trt.Dims([1, 3, 224, 224]), opt=trt.Dims([64, 3, 224, 224]),
                         max=trt.Dims([MAX_BATCH_SIZE, 3, 224, 224]))
    config.add_optimization_profile(op_profile)

    # 保存serialized engine到文件
    serialized_engine = builder.build_serialized_network(network, config)
    with open(TRT_MODEL_PATH, 'wb') as f:
        f.write(serialized_engine)

    return True


def load_trt_engine():
    """从trt serialized engine文件加载trt engine"""
    print('Loading tensorrt engine...')
    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    with open(TRT_MODEL_PATH, "rb") as f:
        serialized_engine = f.read()
    engine = runtime.deserialize_cuda_engine(serialized_engine)  # 创建trt engine
    inspector = engine.create_engine_inspector()
    print('engine layer_info:\n{}'.format(
        inspector.get_engine_information(trt.LayerInformationFormat(1))))  # 打印engine layer描述
    context = engine.create_execution_context()  # 创建trt执行上下文
    return engine, context


def trt_malloc(inp_data):
    """malloc tensorrt的input和output的cpu和gpu内存"""
    # h_input = cuda.pagelocked_empty(trt.volume(input_shape), dtype=np.float32)
    h_input = np.array(inp_data)
    # Allocate device memory for inputs and outputs.
    # print('h_input.nbytes: {}, h_output.nbytes: {}'.format(h_input.nbytes, h_output.nbytes))
    d_input = cuda.mem_alloc(MAX_BATCH_SIZE * 3 * 224 * 224 * 4)  # 可以申请比较大的batch size方便不同输入重复利用这块显存
    d_output = cuda.mem_alloc(MAX_BATCH_SIZE * 1000 * 4)  # 可以申请比较大的batch size方便不同输入重复利用这块显存
    return h_input, d_input, d_output


def trt_infer(input_idx, h_input, d_input, d_output, trt_ctx, stream):
    """pure-trt模型推理"""
    batch_size = h_input.shape[0]
    # 设置真实的input shape
    trt_ctx.set_binding_shape(input_idx, h_input.shape)
    # 拷贝输入数据从cpu到gpu
    cuda.memcpy_htod_async(d_input, h_input, stream)
    # 执行推理
    trt_ctx.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    # 拷贝输出数据从gpu到cpu
    h_output = cuda.pagelocked_empty((batch_size, 1000), dtype=np.float32)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    # 同步等待cuda stream执行完毕
    stream.synchronize()
    return h_output


def confirm_output(data, torch_model, onnx_session, trt_engine, trt_ctx):
    """确认不同模型输出是否一致"""
    urllib.request.urlretrieve(SYNSET_URL, SYNSET_NAME)
    with open(SYNSET_NAME) as f:
        synset = eval(f.read())

    input_x = torch.from_numpy(data)
    torch_output = torch_model(input_x.cuda())
    torch_top5 = torch.argsort(torch_output, 1).cpu().detach().numpy()[0][-1:-6:-1]
    print("Torch output top-5 id: {}, predict class name: {}".format(torch_top5, synset[torch_top5[0]]))

    input_name = onnx_sess.get_inputs()[0].name
    output_name = onnx_sess.get_outputs()[0].name
    onnx_output = onnx_sess.run([output_name], {input_name: data})
    top5_onnx = np.argsort(onnx_output[0][0])[-1:-6:-1]
    print("Onnx output top-5 id: {}, predict class name: {}".format(top5_onnx, synset[top5_onnx[0]]))

    input_idx = trt_engine['input']
    output_idx = trt_engine['output']
    # print('input shape: {}, output shape: {}'.format(trt_ctx.get_binding_shape(input_idx),
    #                                                  trt_ctx.get_binding_shape(output_idx)))
    h_input, d_input, d_output = trt_malloc(np.ascontiguousarray(data))
    stream = cuda.Stream()  # 创建cuda stream，一个stream对应一系列cuda操作，譬如拷贝内存与执行cuda核函数
    trt_out = trt_infer(input_idx, h_input, d_input, d_output, trt_ctx, stream)
    top5_trt = np.argsort(trt_out[0])[-1:-6:-1]
    print("Pure-trt output top-5 id: {}, predict class name: {}".format(top5_trt, synset[top5_trt[0]]))


def compare_infer_speed(data, torch_model, onnx_sess, trt_engine, trt_ctx):
    """比较不同模型的推理速度"""
    timing_number = 10
    timing_repeat = 10
    input_x = torch.from_numpy(data)
    torch_speed = (
            np.array(timeit.Timer(lambda: torch_model(input_x.cuda()))
                     .repeat(repeat=timing_repeat, number=timing_number))
            * 1000 / timing_number
    )
    torch_speed = {
        "mean": np.mean(torch_speed),
        "median": np.median(torch_speed),
        "std": np.std(torch_speed),
    }

    input_name = onnx_sess.get_inputs()[0].name
    output_name = onnx_sess.get_outputs()[0].name
    onnx_speed = (
            np.array(timeit.Timer(lambda: onnx_sess.run([output_name], {input_name: data}))
                     .repeat(repeat=timing_repeat, number=timing_number))
            * 1000 / timing_number
    )
    onnx_speed = {
        "mean": np.mean(onnx_speed),
        "median": np.median(onnx_speed),
        "std": np.std(onnx_speed),
    }

    input_idx = trt_engine['input']
    output_idx = trt_engine['output']
    # print('input shape: {}, output shape: {}'.format(trt_ctx.get_binding_shape(input_idx),
    #                                                  trt_ctx.get_binding_shape(output_idx)))
    h_input, d_input, d_output = trt_malloc(np.ascontiguousarray(data))
    stream = cuda.Stream()  # 创建cuda stream，一个stream对应一系列cuda操作，譬如拷贝内存与执行cuda核函数
    trt_speed = (
            np.array(timeit.Timer(lambda: trt_infer(input_idx, h_input, d_input, d_output, trt_ctx, stream))
                     .repeat(repeat=timing_repeat, number=timing_number))
            * 1000 / timing_number
    )
    trt_speed = {
        "mean": np.mean(trt_speed),
        "median": np.median(trt_speed),
        "std": np.std(trt_speed),
    }

    print('torch_speed: {}\nonnx_speed:{}\ntrt_speed:{}'.format(torch_speed, onnx_speed, trt_speed))


if __name__ == '__main__':
    data = load_data()  # 加载测试数据

    torch_model = load_torch_model()  # 加载torch resnet-50模型
    onnx_sess = load_onnx_model(torch_model)  # 转换并加载onnx模型
    # onnx_2_trt_engine_by_trtexec()  # 通过trtexec命令将onnx模型转换为trt engine
    onnx_2_trt_engine_by_api()  # 通过api将onnx模型转换为trt engine
    trt_engine, trt_ctx = load_trt_engine()  # 加载tensorrt engine

    # Torch output top-5 id: [282 281 287 285 283], predict class name: tiger cat
    # Onnx output top-5 id: [282 281 287 285 283], predict class name: tiger cat
    # Pure-trt output top-5 id: [282 281 287 285 283], predict class name: tiger cat
    confirm_output(data, torch_model, onnx_sess, trt_engine, trt_ctx)  # 比较模型的输出是否一致

    # torch_speed: {'mean': 6.773437949977961, 'median': 6.643094050014042, 'std': 0.32557030622709593}
    # onnx_speed:{'mean': 18.87734026999169, 'median': 18.866386999980023, 'std': 0.09169522564388215}
    # trt_speed:{'mean': 2.316488880042016, 'median': 2.312389200051257, 'std': 0.015701868470197833}
    compare_infer_speed(data, torch_model, onnx_sess, trt_engine, trt_ctx)  # 比较模型的推理速度
