import os

__base_dir = r'D:\Desktop\shishuai.yan\Desktop\git_code\tf_keras_classifier\imgs\clear_2'
train_dir = os.path.join(__base_dir, 'train')
valid_dir = os.path.join(__base_dir, 'valid')
checkpoint_path = os.path.abspath('./output/training_2/model-{epoch:04d}.ckpt')
checkpoint_dir = os.path.dirname(checkpoint_path)
checkpoint_name = None      # 指定使用某一个ckpt的权重，默认为最新的权重文件
pb_save_base_dir = os.path.abspath('./output/saved_pb_model')

img_size = 96
batch_size = 64
epoch = 20
