# reference: https://www.tensorflow.org/tutorials/keras/save_and_restore_models
import tensorflow as tf
import os, cv2
import numpy as np
from tensorflow import keras
from model import Model
import global_variables as gl

class Evaluate:
    def __init__(self, checkpoint_dir):
        self.img_size = gl.img_size
        self.epoch = gl.epoch
        self.checkpoint_dir = checkpoint_dir
        self.pb_save_base_dir = gl.pb_save_base_dir
        self.h5_save_base_dir = gl.h5_save_base_dir

    def restore_from_ckpt(self, ckpt_name=None):
        self.model = Model(self.img_size, self.epoch).get_model()
        self.checkpoint_path = os.path.join(self.checkpoint_dir, ckpt_name) if ckpt_name else tf.train.latest_checkpoint(gl.checkpoint_dir)
        print('Restored model from {}'.format(self.checkpoint_path))
        self.model.load_weights(self.checkpoint_path)

    def restore_from_pb(self, pb_path):
        print('Restored model from {}'.format(pb_path))
        self.model = tf.contrib.saved_model.load_keras_model(os.path.dirname(pb_path))
        # The model has to be compiled before evaluating.
        # This step is not required if the saved model is only being deployed.
        # self.model.compile(optimizer=tf.keras.optimizers.Adam(),
        #       loss=tf.keras.losses.sparse_categorical_crossentropy,
        #       metrics=['acc'])

    def eval_from_np(self, test_images, test_labels):
        loss, acc = self.model.evaluate(test_images, test_labels)
        print('    loss: {:<.5f}  acc: {:<.5f}'.format(loss, acc))

    def eval_from_generator(self, test_dir):
        test_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)
        test_generator = test_gen.flow_from_directory(
            test_dir,
            target_size=(self.img_size, self.img_size),
            class_mode='binary'
        )
        loss, acc = self.model.evaluate_generator(test_generator, verbose=0)
        print('    loss: {:<.5f}  acc: {:<.5f}'.format(loss, acc))

    def get_score(self, img_path):
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = np.expand_dims(image, axis=0)
        image = image / 255.
        score = self.model.predict(image)[0][0]
        return score

    def freeze2h5(self):
        if not os.path.isdir(self.h5_save_base_dir):
            os.mkdir(self.h5_save_base_dir)
        self.h5_model_path = os.path.join(self.h5_save_base_dir, 'saved_model.h5')
        self.model.save(self.h5_model_path)

    def freeze2pb(self):
        import time
        print(int(time.time()))
        pb_model_dir = os.path.join(self.pb_save_base_dir, str(int(time.time())))
        tf.contrib.saved_model.save_keras_model(self.model, self.pb_save_base_dir)
        for i in range(50):  # 自动创建的名字和time()有差距，一般是1，故加此判断
            pb_model_dir = os.path.join(self.pb_save_base_dir, str(int(os.path.basename(pb_model_dir))+i))
            if os.path.isdir(pb_model_dir):
                self.pb_model_path = os.path.join(pb_model_dir, 'saved_model.pb')
                print('pb model saved in {}'.format(self.pb_model_path))
                return self.pb_model_path


def test_model():
    checkpoint_dir = r'D:\Desktop\shishuai.yan\Desktop\git_code\tf_keras_classifier\output\training_12'
    evaluate = Evaluate(checkpoint_dir)
    evaluate.restore_from_ckpt(ckpt_name='model-0200.ckpt')
    evaluate.freeze2h5()
    # evaluate.freeze2pb()
    score = evaluate.get_score(r'D:\Desktop\shishuai.yan\Desktop\0.jpg')
    print(score)
test_model()

# if __name__ == '__main__':
#     evaluate = Evaluate(gl.checkpoint_dir)
#     evaluate.restore_from_ckpt()
#     # evaluate.eval_from_generator(gl.valid_dir)
#     # score = evaluate.get_score(r'D:\Desktop\shishuai.yan\Desktop\git_code\tf_keras_classifier\imgs\clear_2\valid\1\0_2_100.jpg')
#     score = evaluate.get_score(r'D:\Desktop\shishuai.yan\Desktop\0.jpg')
#     print(score)        # 分数越低，越清晰
#     # pb_path = evaluate.freeze2pb()
#     # print(pb_path)
#
#     # pb_path = r'D:\Desktop\shishuai.yan\Desktop\git_code\tf_keras_classifier\output\saved_pb_model\1567665978\saved_model.pb'
#     # evaluate.restore_from_pb(pb_path)
#     # score = evaluate.get_score(r'D:\Desktop\shishuai.yan\Desktop\git_code\tf_keras_classifier\imgs\clear_2\valid\1\0_2_100.jpg')
#     # print(score)

