# Pytorch模型做pure-trt优化并比较对应的性能
# pure-trt指的是使用pure tensorrt模型，使用的推理引擎是纯tensorrt引擎
#
# 环境：
# gpu: Nvidia GeForce 3060
# gpu driver: 470.57.02
# cuda: cuda_11.3
# cudnn: v8.1
# pytorch: torch-1.10.0+cu113 torchvision-0.11.1+cu113
# tensorrt: TensorRT-8.0.0.3

import os
import timeit
import urllib.request

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt

MODEL_PATH = '/tmp/resnet-50/'
ONNX_MODEL_PATH = MODEL_PATH + 'model.onnx'
TRT_MODEL_PATH = MODEL_PATH + 'model.trt'

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
    img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
    urllib.request.urlretrieve(img_url, './cat.png')
    img = Image.open('./cat.png').resize((224, 224))
    plt.imshow(img)
    plt.show()
    img = np.array(img)[np.newaxis, :].astype("float32")
    data = preprocess_input(img)
    return data


def load_torch_model():
    """加载pytorch resnet-50模型"""
    print('Loading pytorch model...')
    model = torchvision.models.resnet50(pretrained=True).eval().cuda()
    print('Loaded pytorch resnet-50 model.')
    return model


def load_trt_model(torch_model):
    """转换并加载pure-tensorrt模型"""
    # 先把torch模型转换成onnx
    print('Transfer torch model to onnx model...')
    dummy_input = torch.randn(1, 3, 224, 224, device="cuda")
    input_names = ["input"]
    output_names = ["output"]
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    torch.onnx.export(torch_model, dummy_input, ONNX_MODEL_PATH, verbose=True, input_names=input_names,
                      output_names=output_names)

    # onnx执行tensorrt优化并保存引擎文件
    print('Apply tensorrt optimizing...')
    save_trt_engine_cmd = 'trtexec --onnx={} --explicitBatch --saveEngine={}' \
        .format(ONNX_MODEL_PATH, TRT_MODEL_PATH)
    os.system(save_trt_engine_cmd)

    # 加载tensorrt引擎文件
    print('Creating tensorrt engine...')
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    with open(TRT_MODEL_PATH, "rb") as f:
        serialized_engine = f.read()
    engine = runtime.deserialize_cuda_engine(serialized_engine)  # 创建trt engine
    context = engine.create_execution_context()  # 创建trt执行上下文
    return engine, context


def trt_malloc(trt_engine, inp_data, input_idx, output_idx):
    """malloc tensorrt的input和output的cpu和gpu内存"""
    # h_input = cuda.pagelocked_empty(trt.volume(trt_engine.get_binding_shape(input_idx)), dtype=np.float32)
    h_input = np.array(inp_data)
    h_output = cuda.pagelocked_empty(trt.volume(trt_engine.get_binding_shape(output_idx)), dtype=np.float32)
    # Allocate device memory for inputs and outputs.
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    return h_input, h_output, d_input, d_output


def trt_infer(h_input, h_output, d_input, d_output, trt_ctx, stream):
    """pure-trt模型推理"""
    # 拷贝输入数据从cpu到gpu
    cuda.memcpy_htod_async(d_input, h_input, stream)
    # 执行推理
    trt_ctx.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    # 拷贝输出数据从gpu到cpu
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    # 同步等待cuda stream执行完毕
    stream.synchronize()
    return h_output


def confirm_output(data, torch_model, trt_engine, trt_ctx):
    """确认不同模型输出是否一致"""
    urllib.request.urlretrieve(SYNSET_URL, SYNSET_NAME)
    with open(SYNSET_NAME) as f:
        synset = eval(f.read())

    input_x = torch.from_numpy(data)
    torch_output = torch_model(input_x.cuda())
    torch_top5 = torch.argsort(torch_output, 1).cpu().detach().numpy()[0][-1:-6:-1]
    print("Torch output top-5 id: {}, predict class name: {}".format(torch_top5, synset[torch_top5[0]]))

    input_idx = trt_engine['input']
    output_idx = trt_engine['output']
    # print('input shape: {}, output shape: {}'.format(trt_engine.get_binding_shape(input_idx),
    #                                                  trt_engine.get_binding_shape(output_idx)))
    h_input, h_output, d_input, d_output = trt_malloc(trt_engine,
                                                      np.ascontiguousarray(data),
                                                      input_idx,
                                                      output_idx)
    stream = cuda.Stream()  # 创建cuda stream，一个stream对应一系列cuda操作，譬如拷贝内存与执行cuda核函数
    trt_out = trt_infer(h_input, h_output, d_input, d_output, trt_ctx, stream)
    top5_trt = np.argsort(trt_out)[-1:-6:-1]
    print("Pure-trt output top-5 id: {}, predict class name: {}".format(top5_trt, synset[top5_trt[0]]))


def compare_infer_speed(data, torch_model, trt_engine, trt_ctx):
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

    input_idx = trt_engine['input']
    output_idx = trt_engine['output']
    # print('input shape: {}, output shape: {}'.format(trt_engine.get_binding_shape(input_idx),
    #                                                  trt_engine.get_binding_shape(output_idx)))
    h_input, h_output, d_input, d_output = trt_malloc(trt_engine,
                                                      np.ascontiguousarray(data),
                                                      input_idx,
                                                      output_idx)
    stream = cuda.Stream()  # 创建cuda stream，一个stream对应一系列cuda操作，譬如拷贝内存与执行cuda核函数
    trt_speed = (
            np.array(timeit.Timer(lambda: trt_infer(h_input, h_output, d_input, d_output, trt_ctx, stream))
                     .repeat(repeat=timing_repeat, number=timing_number))
            * 1000 / timing_number
    )
    trt_speed = {
        "mean": np.mean(trt_speed),
        "median": np.median(trt_speed),
        "std": np.std(trt_speed),
    }

    print('torch_speed: {}\ntrt_speed:{}'.format(torch_speed, trt_speed))


if __name__ == '__main__':
    data = load_data()  # 加载测试数据

    torch_model = load_torch_model()  # 加载torch resnet-50模型
    trt_engine, trt_ctx = load_trt_model(torch_model)  # 转换并加载tensorrt模型

    # Torch output top-5 id: [282 281 287 285 283], predict class name: tiger cat
    # Pure-trt output top-5 id: [282 281 287 285 283], predict class name: tiger cat
    confirm_output(data, torch_model, trt_engine, trt_ctx)  # 比较模型的输出是否一致

    # torch_speed: {'mean': 6.454062859993428, 'median': 6.378020749980351, 'std': 0.3652696340397562}
    # trt_speed: {'mean': 2.8215294500023447, 'median': 2.8216907999649266, 'std': 0.008183779458733451}
    compare_infer_speed(data, torch_model, trt_engine, trt_ctx)  # 比较模型的推理速度
