import tensorflow as tf


def fuse(variables):
    return tf.concat([tf.reshape(v, [-1]) for v in variables], 0)


def extract(variables, fused):
    assign_params = []
    bias = 0
    for v in variables:
        size = v.shape.num_elements()
        assign_params.append(tf.reshape(fused[bias:bias + size], v.shape))
        bias += size
    return assign_params
