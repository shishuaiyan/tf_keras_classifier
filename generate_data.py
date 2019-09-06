import tensorflow as tf
import os
from tensorflow import keras

import global_variables as gl

class GenerateData():
    def __init__(self, train_dir, valid_dir, batch_size, img_size):
        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.train_generator, self.valid_generator = self.__data_generator()

    def __data_generator(self):
        train_data_gen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255.,
            rotation_range=20,
            width_shift_range=0.05,
            height_shift_range=0.05,
            shear_range=0.2,
            zoom_range=0.,
            horizontal_flip=True)
        valid_data_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)
        train_generator = train_data_gen.flow_from_directory(self.train_dir,
                                                             batch_size=self.batch_size,
                                                             class_mode='binary',
                                                             target_size=(self.img_size,self.img_size))
        valid_generator = valid_data_gen.flow_from_directory(self.valid_dir,
                                                             batch_size=self.batch_size,
                                                             class_mode='binary',
                                                             target_size=(self.img_size,self.img_size))
        return train_generator, valid_generator

    def get_generator(self):
        return self.train_generator, self.valid_generator

    def show_data_in_gen(self):
        import matplotlib.pyplot as plt
        batch_image, batch_label = self.train_generator.next()   # 返回下一个batch
        plt.imshow(batch_image[0])
        print(batch_image[0], batch_label[0])
        plt.show()


def main():
    gd = GenerateData(gl.train_dir, gl.valid_dir, gl.batch_size, gl.img_size)
    train_generator, valid_generator = gd.get_generator()
    print(train_generator.samples)  # 图片个数
    # gd.show_data_in_gen()

if __name__ == '__main__':
    main()