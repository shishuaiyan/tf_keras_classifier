from tensorflow import keras
import tensorflow as tf
from tensorflow.python.framework import graph_io
import os, cv2
import numpy as np

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

def freeze_h5_to_pb(h5_model_path, pb_output_dir, pb_model_name):
    # """----------------------------------导入keras模型------------------------------"""
    keras.backend.set_learning_phase(0)
    net_model = keras.models.load_model(h5_model_path)
    print('input is :', net_model.input.name)
    print ('output is:', net_model.output.name)
    # """----------------------------------保存为.pb格式------------------------------"""
    frozen_graph = freeze_session(keras.backend.get_session(), output_names=[net_model.output.op.name])
    graph_io.write_graph(frozen_graph, pb_output_dir, pb_model_name, as_text=False)


def test_pb(pb_output_dir, pb_model_name, img_path):
    pb_model_path = os.path.join(pb_output_dir, pb_model_name)
    with tf.gfile.FastGFile(pb_model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
    sess = tf.Session()
    softmax_tensor = sess.graph.get_tensor_by_name('output_dense/Sigmoid:0')    # output name
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image/255.
    image = cv2.resize(image, (96, 96))
    image = np.expand_dims(image, axis=0)
    predictions = sess.run(softmax_tensor, feed_dict={'input_1:0': image})      # input name
    # 与h5模型对比
    new_model = keras.models.load_model('./output/saved_h5_model/saved_model.h5')
    y = new_model.predict(image)
    print('pb_model: {}     h5_model: {}'.format(predictions[0][0], y[0][0]))


if __name__ == '__main__':
    h5_model_path='./output/saved_h5_model/saved_model.h5'
    pb_output_dir= './output/saved_h5_model'
    pb_model_name='my_model_pb.pb'
    # freeze_h5_to_pb()和test_pb()不能一起运行，会报错，并不知道为什么。。
    # 先freeze，后test，分开运行
    # freeze_h5_to_pb(h5_model_path, pb_output_dir, pb_model_name)
    test_pb(pb_output_dir, pb_model_name, r'D:\Desktop\shishuai.yan\Desktop\1.jpg')
