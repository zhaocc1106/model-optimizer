# model-optimizer
各类深度模型的优化器的应用和性能对比，包括tvm、tf-tensorrt、tensorrt等。

## 目录
[keras_model_optimizer.py](https://github.com/zhaocc1106/model-optimizer/blob/master/keras_model_optimizer.py): 将keras版resnet-50模型分别进行tvm、tf-tensorrt、pure-tensorrt优化并比较推理速度。<br>
[pytorch_model_optimizer.py](https://github.com/zhaocc1106/model-optimizer/blob/master/pytorch_model_optimizer.py): 将pytorch版resnet-50模型进行pure-tensorrt优化并比较推理速度，包括tensorrt int8校准量化实现。

## 相关仓库
[c++ api infer pure tensorrt model](https://github.com/zhaocc1106/tensorrt-infer)
