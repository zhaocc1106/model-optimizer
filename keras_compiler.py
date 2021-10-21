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


def confirm_output(keras_model, tvm_model, data):
    """确认两个模型输出是否一致并且正确"""
    keras_out = keras_model.predict(data)
    top1_keras = np.argmax(keras_out)

    tvm_out = tvm_model(tvm.nd.array(data.transpose([0, 3, 1, 2]).astype("float32")))
    top1_tvm = np.argmax(tvm_out.numpy()[0])

    synset_url = "".join(
        [
            "https://gist.githubusercontent.com/zhreshold/",
            "4d0b62f3d01426887599d4f7ede23ee5/raw/",
            "596b27d23537e5a1b5751d2b0481ef172f58b539/",
            "imagenet1000_clsid_to_human.txt",
        ]
    )
    synset_name = "imagenet1000_clsid_to_human.txt"
    synset_path = download_testdata(synset_url, synset_name, module="data")
    with open(synset_path) as f:
        synset = eval(f.read())
    print("Tvm output top-1 id: {}, class name: {}\nKeras output top-1 id: {}, class name: {}"
          .format(top1_tvm, synset[top1_tvm], top1_keras, synset[top1_keras]))

    assert (top1_keras == top1_tvm)


def compare_infer_speed(keras_model, tvm_model, data):
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

    print('tvm_speed: {}\nkeras_speed: {}'.format(tvm_speed, keras_speed))


if __name__ == '__main__':
    data = load_data()  # 加载测试数据
    keras_model = load_keras_model()  # 加载一个keras模型
    tvm_model = tvm_compile(keras_model,  # 将keras模型编译转换为tvm模型
                            {"input_1": data.transpose(
                                [0, 3, 1, 2]).shape})  # tvm input layout是NCHW格式，tensorflow默认为NHWC格式

    # Tvm output top-1 id: 285, class name: Egyptian cat
    # Keras output top-1 id: 285, class name: Egyptian cat
    confirm_output(keras_model, tvm_model, data)  # 确认两个模型的输出是否一致

    # tvm_speed: {'mean': 7.203045489950455, 'median': 6.938162950063997, 'std': 0.6353804612247146}
    # keras_speed: {'mean': 39.83375730000262, 'median': 40.232931900027324, 'std': 1.3520431233972965}
    compare_infer_speed(keras_model, tvm_model, data)  # 比较两个模型的推理速度
