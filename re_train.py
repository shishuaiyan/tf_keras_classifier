import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import global_variables as gl
from model import Model
from generate_data import GenerateData
import os

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
            ckpt_path = tf.train.latest_checkpoint(gl.checkpoint_dir)
            if ckpt_path:
                print('Restored model from {}'.format(ckpt_path))
                self.model.load_weights(ckpt_path)
                return
        print('Init model weight....')

    def train(self):
        self.model.summary()
        self.__load_weights()

        save_callback = keras.callbacks.ModelCheckpoint(
            self.checkpoint_path,
            save_weights_only=True,
            verbose=1,
            period=5        # 每5个epoch保存一次
        )

        self.model.save_weights(self.checkpoint_path.format(epoch=0))
        history = self.model.fit_generator(
            self.train_generator,
            validation_data=self.valid_generator,
            steps_per_epoch=self.train_step_per_epoch,
            epochs=self.epoch,
            validation_steps=self.valid_step_per_epoch,
            verbose=2,   # verbose=1时只显示loss和acc，为2时加上定义的valid acc
            callbacks=[save_callback]
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

