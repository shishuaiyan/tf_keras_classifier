import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import global_variables as gl
from model import Model
from generate_data import GenerateData
import os, time

class Trainer:
    def __init__(self):
        self.model = Model(gl.img_size, gl.epoch).get_model()
        self.epoch = gl.epoch
        self.checkpoint_path = gl.checkpoint_path
        self.checkpoint_dir = gl.checkpoint_dir
        self.train_generator, self.valid_generator = GenerateData(gl.train_dir, gl.valid_dir, gl.batch_size, gl.img_size).get_generator()
        self.train_step_per_epoch = int(self.train_generator.samples / gl.batch_size) + 1
        self.valid_step_per_epoch = int(self.valid_generator.samples / gl.batch_size) + 1

    def __load_weights(self):
        if os.path.isdir(self.checkpoint_dir):
            ckpt_path = tf.train.latest_checkpoint(self.checkpoint_dir)
            if ckpt_path:
                print('Restored model from {}'.format(ckpt_path))
                self.model.load_weights(ckpt_path)
                # model加载权重之后需要compile编译，否则无法继续训练
                self.model.compile(optimizer=keras.optimizers.Adam(lr=0.0001),
                                  loss='binary_crossentropy',
                                  metrics=['acc'])
                return
        print('Init model weight....')

    def train(self):
        self.model.summary()
        self.__load_weights()

        save_callback = keras.callbacks.ModelCheckpoint(
            self.checkpoint_path,
            save_weights_only=True,
            verbose=1,
            period=2        # 每5个epoch保存一次
        )

        tensorboard = keras.callbacks.TensorBoard(log_dir="logs/{}".format(time.time()))

        self.model.save_weights(self.checkpoint_path.format(epoch=0))
        history = self.model.fit_generator(
            self.train_generator,
            validation_data=self.valid_generator,
            steps_per_epoch=self.train_step_per_epoch,
            epochs=self.epoch,
            validation_steps=self.valid_step_per_epoch,
            verbose=2,      # verbose为1时显示各batch的训练误差，以及一个epoch后各batch的valid误差
                            # 为2时只显示训练一个epoch后各batch的valid误差
            callbacks=[save_callback, tensorboard]
        )

        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(len(acc))

        plt.figure()
        plt.plot(epochs, acc, 'r', label='Training accuracy')
        plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.legend(loc=0)

        plt.figure()
        plt.plot(epochs, loss, 'r', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend(loc=0)

        plt.show()


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()

