### 功能：
- API: tensorflow.keras
- 数据增强(keras.preprocessing.image.ImageDataGenerator)
- 使用mobilenetV2预训练模型
- 继续训练（会覆盖之前的模型）
- 权重保存、冻结为pb(keras保存的pb模型无法使用tf.gfile.FastGFile读取)