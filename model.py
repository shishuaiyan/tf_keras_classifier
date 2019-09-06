import tensorflow as tf
from tensorflow import keras

class Model:
    def __init__(self, img_size, epoch):
        self.epoch = epoch
        self.img_size = img_size
        self.model = self.__create_model()

    def __create_model(self):
        MobileNetV2 = keras.applications.mobilenet_v2.MobileNetV2
        pre_trained_model = MobileNetV2(input_shape=(self.img_size,self.img_size,3),
                                        include_top=True,
                                        weights='imagenet')
        # 锁住部分层，清晰度分类时效果不好,需要全部放开训练
        # pre_trained_model.summary()
        # for i in range(len(pre_trained_model.layers) - 13): # 除最后11层(一个8层Bottleneck Residual block + 最后的卷积池化)外，其他层不参与训练
        #     # print(pre_trained_model.layers[i])
        #     pre_trained_model.layers[i].trainable = False
        # for layer in pre_trained_model.layers:
        #     if layer.trainable == False:
        #         print(layer)
        last_layer = pre_trained_model.get_layer('global_average_pooling2d')
        last_output = last_layer.output
        # print('last layer ouput shape: ', last_layer.output_shape)
        # print('pre_trained_model input: ', pre_trained_model.input)
        # print('pre_trained_model output', pre_trained_model.output)
        print(last_output)
        x = keras.layers.Dropout(0.3)(last_output)
        x = keras.layers.Dense(1, activation='sigmoid',
                                  kernel_regularizer=keras.regularizers.l1_l2(),
                                  name='output_dense')(x)

        model = keras.Model(pre_trained_model.input, x)

        model.compile(optimizer=keras.optimizers.Adam(lr=0.0001),
                      loss='binary_crossentropy',
                      metrics=['acc'])
        return model

    def get_model(self):
        return self.model

    def get_in_out_name(self):
        return self.model.layers[0].name, self.model.layers[-1].name

if __name__ == '__main__':
    import global_variables as gl
    model = Model(gl.img_size, gl.epoch)
    model.model.summary()
    input_name, output_name = model.get_in_out_name()
    print(input_name, output_name)
