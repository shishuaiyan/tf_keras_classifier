### 功能：
- API: tensorflow.keras
- 数据增强(keras.preprocessing.image.ImageDataGenerator)
- 使用mobilenetV2预训练模型
- 继续训练（会覆盖之前的模型）
- 权重保存、冻结为pb(keras保存的pb模型无法使用tf.gfile.FastGFile读取)

### update
- 新增冻结为一整个pb模型代码(可使用tf.gfile.FastGFile读取)(参考至[博客](https://blog.csdn.net/qq_25109263/article/details/81285952))操作流程如下：
    - 调用evaluate.py中的`Evaluate.freeze2h5()`将指定ckpt冻结为`.h5`模型
    - 调用h5_to_pb.py中`freeze_h5_to_pb()`方法将`.h5`转为`.pb`
    - TODO: 暂时无法改变input和output的名字，后续可考虑load_model后添加层并指定名字来实现
- 为训练清晰度分类模型，新增手动模糊图片的操作`gen_blur_img.py`