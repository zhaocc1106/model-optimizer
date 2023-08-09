# keras模型做tvm, tf-trt, pure-trt优化并比较对应的性能
# tf-trt指的是直接使用tensorflow的tensorrt convert接口转换成的模型，使用的推理引擎是tensorflow嵌入tensorrt引擎
# pure-trt指的是使用pure tensorrt模型，使用的推理引擎是纯tensorrt引擎
#
# 环境：
# gpu: Nvidia GeForce 3060
# gpu driver: 470.57.02
# cuda: cuda_11.2
# cudnn: v8.1
# tensorflow: tensorflow-gpu==2.4.0
# tensorrt: TensorRT-8.0.0.3, TensorRT-7.2.1.6(tf-trt需要的版本，tf-trt需要单独跑)
# tvm: 0.8.dev1949+gf4c146ca3
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tvm
import tvm.relay as relay
import timeit
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.python.compiler.tensorrt import trt_convert
from tvm.contrib.download import download_testdata
import tvm.auto_scheduler as auto_scheduler
from tvm.autotvm.tuner import XGBTuner
from tvm import autotvm
from PIL import Image
from matplotlib import pyplot as plt

print(tf.__version__)
print(keras.__version__)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

MODEL_PATH = '/tmp/resnet50/'
SAVER_PATH = MODEL_PATH + 'tf_model_saver'
TF_TRT_PATH = MODEL_PATH + 'tf_trt_model'
ONNX_PATH = MODEL_PATH + 'model.onnx'
TRT_PATH = MODEL_PATH + 'model.trt'

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


def load_keras_model():
    """加载一个keras模型"""
    if tuple(keras.__version__.split(".")) < ("2", "4", "0"):
        weights_url = "".join(
            [
                "https://github.com/fchollet/deep-learning-models/releases/",
                "download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5",
            ]
        )
        weights_file = "resnet50_keras_old.h5"
    else:
        weights_url = "".join(
            [
                " https://storage.googleapis.com/tensorflow/keras-applications/",
                "resnet/resnet50_weights_tf_dim_ordering_tf_kernels.h5",
            ]
        )
        weights_file = "resnet50_keras_new.h5"

    weights_path = download_testdata(weights_url, weights_file, module="keras")
    keras_resnet50 = keras.applications.resnet50.ResNet50(
        include_top=True, weights=None, input_shape=(224, 224, 3), classes=1000
    )
    keras_resnet50.load_weights(weights_path)

    if not os.path.exists(SAVER_PATH):
        os.makedirs(SAVER_PATH)
    keras_resnet50.save(filepath=SAVER_PATH, save_format='tf')

    # keras_resnet50.summary()
    return keras_resnet50


def tvm_compile(keras_model, shape_dict, auto_tune=True):
    """将keras模型编译为tvm模型"""
    mod, params = relay.frontend.from_keras(keras_model, shape_dict)
    target = "cuda"
    dev = tvm.cuda(0)
    # tvm.autotvm.measure.measure_methods.set_cuda_target_arch("sm_80")

    if auto_tune:
        # 执行auto tvm tuning优化
        tuning_option = {
            "tuning_records": "resnet-50-v2-autotuning.json",
            "tuner": "xgb",
            # "n_trial": 2000,
            "trials": 100,
            "early_stopping": 600,
            "measure_option": autotvm.measure_option(
                builder=autotvm.LocalBuilder(timeout=10),
                runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
            ),
        }
        tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)

        # Tune the extracted tasks sequentially.
        for i, task in enumerate(tasks):
            prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
            tuner_obj = XGBTuner(task, loss_type="rank")
            tuner_obj.tune(
                n_trial=min(tuning_option["trials"], len(task.config_space)),
                early_stopping=tuning_option["early_stopping"],
                measure_option=tuning_option["measure_option"],
                callbacks=[
                    autotvm.callback.progress_bar(tuning_option["trials"], prefix=prefix),
                    autotvm.callback.log_to_file(tuning_option["tuning_records"]),
                ],
            )

    # compile the model
    # TODO(mbs): opt_level=3 causes nn.contrib_conv2d_winograd_weight_transform
    # to end up in the module which fails memory validation on cuda most likely
    # due to a latent bug. Note that the pass context only has an effect within
    # evaluate() and is not captured by create_executor().
    if auto_tune:
        with autotvm.apply_history_best(tuning_option["tuning_records"]):
            with tvm.transform.PassContext(opt_level=3):
                model = relay.build_module.create_executor("graph", mod, dev, target, params).evaluate()
    else:
        with tvm.transform.PassContext(opt_level=3):
            model = relay.build_module.create_executor("graph", mod, dev, target, params).evaluate()
    return model


def load_data():
    """加载测试数据"""
    # img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
    # img_path = download_testdata(img_url, "cat.png", module="data")
    img = Image.open('./cat.png').resize((224, 224))
    # plt.imshow(img)
    # plt.show()
    data = np.array(img)[np.newaxis, :].astype("float32")
    print("input_1: ", data.shape)
    data = preprocess_input(data)
    # print(data)
    return data


def convert_2_tf_trt():
    """将模型转换为tf-tensorrt模型"""
    conversion_params = trt_convert.DEFAULT_TRT_CONVERSION_PARAMS
    conversion_params = conversion_params._replace(max_workspace_size_bytes=(1 << 32))
    conversion_params = conversion_params._replace(precision_mode="FP32")
    conversion_params = conversion_params._replace(maximum_cached_engines=100)
    # conversion_params = conversion_params._replace(minimum_segment_size=100)
    converter = trt_convert.TrtGraphConverterV2(input_saved_model_dir=SAVER_PATH,
                                                input_saved_model_tags=['serve'],
                                                input_saved_model_signature_key='serving_default',
                                                conversion_params=conversion_params)
    converter.convert()

    def my_input_fn():
        inp = np.zeros(shape=(1, 244, 244, 3)).astype(np.float32)  # 这里的batch要比预估时的batch要大，否则在预估时会重新构建engine，很耗时
        yield [inp]

    converter.build(input_fn=my_input_fn)
    converter.save(TF_TRT_PATH)
    print('Convert tf_trt completely!')
    model = tf.saved_model.load(TF_TRT_PATH)
    return model


def load_trt():
    """加载pure tensorrt模型引擎"""
    convert_2_onnx_cmd = 'python -m tf2onnx.convert --saved-model {} --output {} --tag serve' \
                         ' --signature_def serving_default --target tensorrt --opset 11' \
        .format(SAVER_PATH, ONNX_PATH)
    save_trt_engine_cmd = 'trtexec --onnx={} --explicitBatch --shapes=input_1:1x224x224x3 --saveEngine={}' \
        .format(ONNX_PATH, TRT_PATH)
    os.system(convert_2_onnx_cmd)  # 将tf模型转换成onnx模型
    os.system(save_trt_engine_cmd)  # onnx模型转换成pure tensorrt模型

    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    with open(TRT_PATH, "rb") as f:
        serialized_engine = f.read()
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    context = engine.create_execution_context()
    return engine, context


def trt_malloc(trt_engine, inp_data, input_idx, output_idx):
    """malloc tensorrt的input和output的cpu和gpu内存"""
    h_input = np.array(inp_data)
    h_output = cuda.pagelocked_empty(trt_engine.get_binding_shape(output_idx), dtype=np.float32)
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


def confirm_output(data, keras_model, tvm_model=None, tf_trt_model=None, trt_engine=None, trt_ctx=None):
    """确认模型输出是否一致并且正确"""
    synset_path = download_testdata(SYNSET_URL, SYNSET_NAME, module="data")
    with open(synset_path) as f:
        synset = eval(f.read())

    keras_out = keras_model.predict(data)
    top5_keras = np.argsort(keras_out[0])[-1:-6:-1]
    print("Keras output top-5 id: {}, predict class name: {}".format(top5_keras, synset[top5_keras[0]]))

    if tvm_model:
        tvm_out = tvm_model(tvm.nd.array(data.transpose([0, 3, 1, 2]).astype("float32")))
        top5_tvm = np.argsort(tvm_out.numpy()[0])[-1:-6:-1]
        print("Tvm output top-5 id: {}, predict class name: {}".format(top5_tvm, synset[top5_tvm[0]]))

    if tf_trt_model:
        tf_trt_out = tf_trt_model(data)
        top5_tf_trt = np.argsort(tf_trt_out[0])[-1:-6:-1]
        print("Tf-trt output top-5 id: {}, predict class name: {}".format(top5_tf_trt, synset[top5_tf_trt[0]]))

    if trt_engine and trt_ctx:
        input_idx = trt_engine['input_1']
        output_idx = trt_engine['predictions']
        # print('input shape: {}, output shape: {}'.format(trt_engine.get_binding_shape(input_idx),
        #                                                  trt_engine.get_binding_shape(output_idx)))
        h_input, h_output, d_input, d_output = trt_malloc(trt_engine, data, input_idx, output_idx)
        stream = cuda.Stream()  # 创建cuda stream，一个stream对应一系列cuda操作，譬如拷贝内存与执行cuda核函数
        trt_out = trt_infer(h_input, h_output, d_input, d_output, trt_ctx, stream)
        top5_trt = np.argsort(trt_out)[-1:-6:-1]
        print("Pure-trt output top-5 id: {}, predict class name: {}".format(top5_trt, synset[top5_trt[0]]))


def compare_infer_speed(data, keras_model, tvm_model=None, tf_trt_model=None, trt_engine=None, trt_ctx=None):
    """比较不同模型的推理速度"""
    timing_number = 10
    timing_repeat = 10
    keras_speed = (
            np.array(timeit.Timer(lambda: keras_model.predict(data))
                     .repeat(repeat=timing_repeat, number=timing_number))
            * 1000 / timing_number
    )
    keras_speed = {
        "mean": np.mean(keras_speed),
        "median": np.median(keras_speed),
        "std": np.std(keras_speed),
    }
    print('keras_speed: {}'.format(keras_speed))

    if tvm_model:
        tvm_data = tvm.nd.array(data.transpose([0, 3, 1, 2]).astype("float32"))
        tvm_speed = (
                np.array(timeit.Timer(lambda: tvm_model(tvm_data))
                         .repeat(repeat=timing_repeat, number=timing_number))
                * 1000 / timing_number
        )
        tvm_speed = {
            "mean": np.mean(tvm_speed),
            "median": np.median(tvm_speed),
            "std": np.std(tvm_speed),
        }
        print('tvm_speed: {}'.format(tvm_speed))

    if tf_trt_model:
        tf_trt_speed = (
                np.array(timeit.Timer(lambda: tf_trt_model(data))
                         .repeat(repeat=timing_repeat, number=timing_number))
                * 1000 / timing_number
        )
        tf_trt_speed = {
            "mean": np.mean(tf_trt_speed),
            "median": np.median(tf_trt_speed),
            "std": np.std(tf_trt_speed),
        }
        print('tf_trt_speed: {}'.format(tf_trt_speed))

    if trt_engine and trt_ctx:
        input_idx = trt_engine['input_1']
        output_idx = trt_engine['predictions']
        # print('input shape: {}, output shape: {}'.format(trt_engine.get_binding_shape(input_idx),
        #                                                  trt_engine.get_binding_shape(output_idx)))
        h_input, h_output, d_input, d_output = trt_malloc(trt_engine, data, input_idx, output_idx)
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
        print('trt_speed: {}'.format(trt_speed))


if __name__ == '__main__':
    data = load_data()  # 加载测试数据
    keras_model = load_keras_model()  # 加载一个keras模型
    tvm_model = tvm_compile(
        keras_model,  # 将keras模型编译转换为tvm模型
        {"input_1": data.transpose([0, 3, 1, 2]).shape},  # tvm input layout是NCHW格式，tensorflow默认为NHWC格式
        auto_tune=True)  # 执行auto tvm tuning优化
    # 转换成tf-tensorrt(tensorflow嵌入tensorrt引擎)模型，tf-trt需要单独跑，因为它使用的tensorrt版本是TensorRT-7.2.1.6
    # tf_trt_model = convert_2_tf_trt()
    trt_engine, trt_ctx = load_trt()  # 转换成pure-tensorrt(tensorrt引擎推理)模型

    # Keras output top-5 id: [285 282 263 278 281], predict class name: Egyptian cat
    # Tvm output top-5 id: [285 282 263 278 281], predict class name: Egyptian cat
    # Tf-trt output top-5 id: [285 282 263 278 281], predict class name: Egyptian cat
    # Pure-trt output top-5 id: [285 282 263 278 281], predict class name: Egyptian cat
    confirm_output(data, keras_model, tvm_model, None, trt_engine, trt_ctx)  # 确认模型的输出是否一致

    # keras_speed: {'mean': 38.15342679999958, 'median': 38.13050270000531, 'std': 0.739319260929989}
    # tvm_speed: {'mean': 4.332577469986063, 'median': 4.116869249992305, 'std': 0.42891433991018335}
    # tf_trt_speed: {'mean': 9.36842440001783, 'median': 9.254672099996242, 'std': 1.0029966871852858}
    # trt_speed: {'mean': 2.6983253600064927, 'median': 2.682709500004421, 'std': 0.07963485829412356}
    compare_infer_speed(data, keras_model, tvm_model, None, trt_engine, trt_ctx)  # 比较模型的推理速度
