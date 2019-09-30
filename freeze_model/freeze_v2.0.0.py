# coding=utf-8
from __future__ import print_function, division, unicode_literals

import sys
import os
import argparse

import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile


def graph_def(src, output_node_names, ispb=False):
    """
    冻结并产生图的pb模型
    :param src: 训练模型的路径表
    :param output_node_names: 产生的pb模型输出节点名列表,以','分割的字符串表
    :param ispb: 源模型是否为pb模型
    """
    if ispb:
        if src is not None:
            _graph_ = tf.Graph()
            with _graph_.as_default():
                with gfile.FastGFile(src, 'rb') as f:
                    _graph_def = tf.GraphDef()
                    _graph_def.ParseFromString(f.read())
                    tf.import_graph_def(_graph_def, name='')
                    print("[INFO] model %s load done!" % src)
        return _graph_def
    else:
        with tf.Graph().as_default():
            sess = tf.Session()
            if not tf.gfile.Exists(src):
                raise AssertionError("src dir %s dosen't exists." % src)
            checkpoint = tf.train.get_checkpoint_state(src)
            input_checkpoint = checkpoint.model_checkpoint_path
            print("[INFO] input_checkpoint:", input_checkpoint)
            saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
            saver.restore(sess, input_checkpoint)
            del tf.get_collection_ref(tf.GraphKeys.TRAIN_OP)[:]  # save之前移除训练相关节点
            graph_def = sess.graph.as_graph_def()
            # RefSwitch -> Switch + add '/read' to the input names
            # AssignSub -> Sub + remove use_locking attributes
            # AssignAdd -> Add + remove use_locking attributes
            ref_switch_count = 0
            assign_sub_count = 0
            assign_add_count = 0
            for node in graph_def.node:
                if node.op == 'RefSwitch':
                    ref_switch_count += 1
                    node.op = 'Switch'
                    for index in range(len(node.input)):
                        if 'moving_' in node.input[index]:
                            node.input[index] = node.input[index] + '/read'
                elif node.op == 'AssignSub':
                    assign_sub_count += 1
                    node.op = 'Sub'
                    if 'use_locking' in node.attr:
                        del node.attr['use_locking']
                elif node.op == 'AssignAdd':
                    assign_add_count += 1
                    node.op = 'Add'
                    if 'use_locking' in node.attr:
                        del node.attr['use_locking']
            print("RefSwitch(%d) AssignSub(%d) AssignAdd(%d)" % (ref_switch_count, assign_sub_count, assign_add_count))
            output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, output_node_names.split(','))
            sess.close()
        return output_graph_def


def main(args=None):
    '''
    将多个子图合并为一个大图，各子图之间无直接联系，使用时通过不同的输入输出节点名来获得不同子图的输出
    '''
    networks = {1: {'name': 'YoloV3', 'src': args.ckpt_file_yolov3, 'output_names': 'YoloV3/detections', 'ispb': False},
                2: {'name': 'PNet', 'src': args.ckpt_file_mtcnn+'/1', 'output_names': 'PNet/detections', 'ispb': False},
                3: {'name': 'RNet', 'src': args.ckpt_file_mtcnn+'/2', 'output_names': 'RNet/detections', 'ispb': False},
                4: {'name': 'ONet', 'src': args.ckpt_file_mtcnn+'/3', 'output_names': 'ONet/detections', 'ispb': False},
                5: {'name': 'HAFaceNet', 'src': args.insightface_pb_file, 'output_names': 'FaceNet/detections', 'ispb': True},
                # 6: {'name': 'HCNet', 'src': args.ckpt_path_hcnet, 'output_names': 'HCNet/head_clear', 'ispb': False},
                6: {'name': 'HCNet', 'src': args.pb_path_hcnet, 'output_names': 'output_dense/Sigmoid', 'ispb': True},
                7: {'name': 'FaceNet', 'src': args.ckpt_file_facenet, 'output_names': 'embeddings_ID' if args.facenet_new else 'embeddings', 'ispb': False},
                8: {'name': 'TraceNet', 'src': args.tracenet_pb_file, 'output_names': 'output', 'ispb': True}}
    model_file = '../'+args.model_name+'.pb'
    with tf.Graph().as_default() as graph:
        with tf.Session(graph=graph) as sess:
            for idx in networks:
                name = networks[idx]['name']
                output_names = networks[idx]['output_names']
                ispb = networks[idx]['ispb']
                net_graph_def = graph_def(args.model_dir+networks[idx]['src'], output_names, ispb)
                if name == 'YoloV3':
                    inputs_yolov3 = tf.placeholder(tf.float32, [None, None, None, 3], name=name+'/inputs')
                    image_size_yolov3 = tf.placeholder(tf.float32, [None, 1, 2], name=name+'/image_sizes')
                    output_yolov3 = tf.import_graph_def(net_graph_def, input_map={name+'/input:0': inputs_yolov3, name+'/image_size:0': image_size_yolov3}, return_elements=[output_names+':0'])
                    tf.identity(output_yolov3[0], output_names)
                elif name == 'PNet':
                    inputs_pnet = tf.placeholder(tf.float32, [None, None, None, 3], name=name+'/inputs')
                    scale_pnet = tf.placeholder(tf.float32, name=name+'/scale')
                    output_pnet = tf.import_graph_def(net_graph_def, input_map={name+'/input:0': inputs_pnet, name+'/scale:0': scale_pnet}, return_elements=[output_names + ':0'])
                    tf.identity(output_pnet[0], output_names)
                elif name == 'RNet':
                    inputs_rnet = tf.placeholder(tf.float32, [None, 24, 24, 3], name=name+'/inputs')
                    input_boxes_rnet = tf.placeholder(tf.int32, [None, 4], name=name+'/input_boxes')
                    output_rnet = tf.import_graph_def(net_graph_def, input_map={name+'/input:0': inputs_rnet, name+'/input_boxes:0': input_boxes_rnet}, return_elements=[output_names + ':0'])
                    tf.identity(output_rnet[0], output_names)
                elif name == 'ONet':
                    inputs_onet = tf.placeholder(tf.float32, [None, 48, 48, 3], name=name+'/inputs')
                    input_boxes_onet = tf.placeholder(tf.int32, [None, 4], name=name+'/input_boxes')
                    output_onet = tf.import_graph_def(net_graph_def, input_map={name+'/input:0': inputs_onet, name+'/input_boxes:0': input_boxes_onet}, return_elements=[output_names + ':0'])
                    tf.identity(output_onet[0], output_names)
                elif name == 'HAFaceNet':
                    inputs_haface = tf.placeholder(tf.uint8, [None, None, None, 3], name=name+'/inputs')
                    output_haface = tf.import_graph_def(net_graph_def, input_map={'FaceNet/input:0': inputs_haface}, return_elements=[output_names + ':0'])
                    tf.identity(output_haface[0], 'HAFaceNet/outputs')
                elif name == 'HCNet':
                    # inputs_hcnet = tf.placeholder(tf.float32, [None, 48, 48, 3], name=name + '/inputs')
                    # output_hcnet = tf.import_graph_def(net_graph_def, input_map={'HCNet/inputs:0': inputs_hcnet},return_elements=[output_names + ':0'])
                    # tf.identity(output_hcnet[0], 'HCNet/outputs')

                    inputs_hcnet = tf.placeholder(tf.float32, [None, 96, 96, 3], name=name + '/inputs')
                    output_hcnet = tf.import_graph_def(net_graph_def, input_map={'input_1:0': inputs_hcnet},return_elements=[output_names + ':0'])
                    tf.identity(output_hcnet[0], 'HCNet/outputs')
                elif name == 'FaceNet':
                    inputs_facenet = tf.placeholder(tf.float32, [None, None, None, 3], name=name+'/inputs')
                    if args.facenet_new:
                        output_facenet = tf.import_graph_def(net_graph_def, input_map={'input_ID:0': inputs_facenet, 'phase_train:0': False}, return_elements=[output_names + ':0'])
                    else:
                        output_facenet = tf.import_graph_def(net_graph_def, input_map={'input:0': inputs_facenet, 'phase_train:0': False}, return_elements=[output_names + ':0'])
                    tf.identity(output_facenet[0], "FaceNet/outputs")
                elif name == 'TraceNet':
                    inputs_trace = tf.placeholder(tf.uint8, [None, args.tracenet_size, args.tracenet_size, 3], name=name + '/inputs')
                    output_trace = tf.import_graph_def(net_graph_def, input_map={'data:0': inputs_trace}, return_elements=[output_names + ':0'])
                    tf.identity(output_trace[0], 'TraceNet/outputs')
                else:
                    pass

            # 保存pb图
            with tf.gfile.GFile(args.model_dir+model_file, 'wb') as f:
                f.write(sess.graph_def.SerializeToString())
                print("%d ops in the final graph." % len(sess.graph_def.node))
                print("[INFO] output_graph:", args.model_dir+model_file)
                print("[INFO] all done")
            for node in sess.graph_def.node:
                print(node.name)
                pass
            pass


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dir', type=str, help='Model root dir.', default='../../models/training/')

    parser.add_argument('--model_name', type=str, help='Model name.', default='freezed_v2.0.1')

    # yolov3_head-20181225
    parser.add_argument('--ckpt_file_yolov3', type=str, help='Checkpoint file.', default='yolov3/head_416/20190815')
    # mtcnn_official
    parser.add_argument('--ckpt_file_mtcnn', type=str, help='Checkpoint file.', default='mtcnn/official')
    # facenet-180613
    parser.add_argument('--ckpt_file_facenet', type=str, help='Checkpoint file.', default='facenet/20180613-102209')
    parser.add_argument('--facenet_new', type=bool, help='New Facenet Id.', default=False)
    # insightface_pb
    parser.add_argument('--insightface_pb_file', type=str, help='Checkpoint file.', default='facenet/tf_facenet_r100_0620.pb')
    # hcnet
    parser.add_argument('--ckpt_path_hcnet', type=str, help='Checkpoint file.', default='clear/20190727')
    parser.add_argument('--pb_path_hcnet', type=str, help='Checkpoint file.', default='clear/clear_mobilenet.pb')
    # tracenet_pb_file
    parser.add_argument('--tracenet_pb_file', type=str, help='Checkpoint file.', default='tracknet/tf_track_y1_80x80.pb')
    parser.add_argument('--tracenet_size', type=int, help='input size.', default=80)

    return parser.parse_args(argv)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    main(parse_arguments(sys.argv[1:]))
