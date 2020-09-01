# import onnx
# from onnx_tf.backend import prepare
import tensorflow as tf

import numpy as np
import os
import pdb

from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

export_dir = './models/sm_4s_ep_70000_from_torch'
graph_pb = './models/model-70000.pb'

builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
sigs = {}
tf.disable_eager_execution()
def load_pb(path_to_pb):
    with tf.gfile.GFile(path_to_pb, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph

tf_graph = load_pb('./models/model-70000.pb')
sess = tf.Session(graph=tf_graph)

# # Show tensor names in graph
# for op in tf_graph.get_operations():
#     print(op.values())

output_tensor = tf_graph.get_tensor_by_name('140:0')
input_tensor = tf_graph.get_tensor_by_name('observation:0')

# output = sess.run(output_tensor, feed_dict={input_tensor: dummy_input})
# print(output)

sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
        tf.saved_model.signature_def_utils.predict_signature_def({"observation:0": input_tensor}, {"140": output_tensor})

builder.add_meta_graph_and_variables(sess,
                                     [tag_constants.SERVING],
                                     signature_def_map=sigs)

builder.save()

