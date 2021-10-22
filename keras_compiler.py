# keras模型做tvm优化
#
# 环境：
# gpu: Nvidia GeForce 3060
# gpu driver: 470.57.02
# cuda: cuda_11.0
# cudnn: v8.0
# tensorflow: tensorflow-gpu==2.4.0
# tensorrt: TensorRT-7.2.1.6
# tvm: 0.8.dev1949+gf4c146ca3

import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tvm
import tvm.relay as relay
import timeit

from tensorflow.keras.applications.resnet50 import preprocess_input
from tvm.contrib.download import download_testdata
from PIL import Image
from matplotlib import pyplot as plt

print(tf.__version__)
print(keras.__version__)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

SAVER_PATH = '/tmp/resnet50/tf_model_saver'
TRT_PATH = '/tmp/resnet50/trt_model'

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


def tvm_compile(keras_model, shape_dict):
    """将keras模型编译为tvm模型"""
    mod, params = relay.frontend.from_keras(keras_model, shape_dict)
    # compile the model
    target = "cuda"
    dev = tvm.cuda(0)
    tvm.autotvm.measure.measure_methods.set_cuda_target_arch("sm_80")

    # TODO(mbs): opt_level=3 causes nn.contrib_conv2d_winograd_weight_transform
    # to end up in the module which fails memory validation on cuda most likely
    # due to a latent bug. Note that the pass context only has an effect within
    # evaluate() and is not captured by create_executor().
    with tvm.transform.PassContext(opt_level=0):
        model = relay.build_module.create_executor("graph", mod, dev, target, params).evaluate()
    return model


def load_data():
    """加载测试数据"""
    img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
    img_path = download_testdata(img_url, "cat.png", module="data")
    img = Image.open(img_path).resize((224, 224))
    # plt.imshow(img)
    # plt.show()
    data = np.array(img)[np.newaxis, :].astype("float32")
    print("input_1: ", data.shape)
    data = preprocess_input(data)
    return data


def confirm_output(keras_model, tvm_model, trt_model, data):
    """确认两个模型输出是否一致并且正确"""
    keras_out = keras_model.predict(data)
    top1_keras = np.argmax(keras_out)

    tvm_out = tvm_model(tvm.nd.array(data.transpose([0, 3, 1, 2]).astype("float32")))
    top1_tvm = np.argmax(tvm_out.numpy()[0])

    trt_out = trt_model(data)
    top1_trt = np.argmax(trt_out)

    synset_path = download_testdata(SYNSET_URL, SYNSET_NAME, module="data")
    with open(synset_path) as f:
        synset = eval(f.read())
    print("Tvm output top-1 id: {}, class name: {}\nKeras output top-1 id: {}, class name: {}\nTrt output top-1 id: {}"
          ", class name: {}".format(top1_tvm, synset[top1_tvm], top1_keras, synset[top1_keras], top1_trt,
                                    synset[top1_trt]))

    assert (top1_keras == top1_tvm)
    assert (top1_keras == top1_trt)


def compare_infer_speed(keras_model, tvm_model, trt_model, data):
    """比较keras和tvm模型的推理速度"""
    timing_number = 10
    timing_repeat = 10
    tvm_speed = (
            np.array(timeit.Timer(lambda: tvm_model(tvm.nd.array(data.transpose([0, 3, 1, 2]).astype("float32"))))
                     .repeat(repeat=timing_repeat, number=timing_number))
            * 1000 / timing_number
    )
    tvm_speed = {
        "mean": np.mean(tvm_speed),
        "median": np.median(tvm_speed),
        "std": np.std(tvm_speed),
    }

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

    trt_speed = (
            np.array(timeit.Timer(lambda: trt_model(data))
                     .repeat(repeat=timing_repeat, number=timing_number))
            * 1000 / timing_number
    )
    trt_speed = {
        "mean": np.mean(trt_speed),
        "median": np.median(trt_speed),
        "std": np.std(trt_speed),
    }

    print('tvm_speed: {}\nkeras_speed: {}\ntrt_speed: {}'.format(tvm_speed, keras_speed, trt_speed))


def convert_2_trt():
    """将模型转换为tf-tensorrt模型"""
    from tensorflow.python.compiler.tensorrt import trt_convert as trt
    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
    conversion_params = conversion_params._replace(max_workspace_size_bytes=(1 << 32))
    conversion_params = conversion_params._replace(precision_mode="FP32")
    conversion_params = conversion_params._replace(maximum_cached_engines=100)
    # conversion_params = conversion_params._replace(minimum_segment_size=100)
    converter = trt.TrtGraphConverterV2(input_saved_model_dir=SAVER_PATH,
                                        input_saved_model_tags=['serve'],
                                        input_saved_model_signature_key='serving_default',
                                        conversion_params=conversion_params)
    converter.convert()

    def my_input_fn():
        inp = np.zeros(shape=(1, 244, 244, 3)).astype(np.float32)  # 这里的batch要比预估时的batch要大，否则在预估时会重新构建engine，很耗时
        yield [inp]

    converter.build(input_fn=my_input_fn)
    converter.save(TRT_PATH)
    print('Convert trt completely!')
    model = tf.saved_model.load(TRT_PATH)
    return model


if __name__ == '__main__':
    data = load_data()  # 加载测试数据
    keras_model = load_keras_model()  # 加载一个keras模型
    tvm_model = tvm_compile(keras_model,  # 将keras模型编译转换为tvm模型
                            {"input_1": data.transpose(
                                [0, 3, 1, 2]).shape})  # tvm input layout是NCHW格式，tensorflow默认为NHWC格式
    trt_model = convert_2_trt()  # 转换成tf-tensorrt(tensorflow嵌入tensorrt引擎)模型

    # Tvm output top-1 id: 285, class name: Egyptian cat
    # Keras output top-1 id: 285, class name: Egyptian cat
    # Trt output top-1 id: 285, class name: Egyptian cat
    confirm_output(keras_model, tvm_model, trt_model, data)  # 确认模型的输出是否一致

    # tvm_speed: {'mean': 6.871772010017594, 'median': 6.730263200006448, 'std': 0.36091768164278515}
    # keras_speed: {'mean': 38.15342679999958, 'median': 38.13050270000531, 'std': 0.739319260929989}
    # trt_speed: {'mean': 9.36842440001783, 'median': 9.254672099996242, 'std': 1.0029966871852858}
    compare_infer_speed(keras_model, tvm_model, trt_model, data)  # 比较模型的推理速度
