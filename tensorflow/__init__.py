import os
import sys
import tensorflow as tf
import numpy as np


_libpath = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), './preduce.so')
_lib = tf.load_op_library(_libpath)


preduce = _lib.p_reduce


def _test():
    rank = int(os.environ['PMI_RANK'])

    group = []
    if rank < 2:
        group.append([1, 1, 0, 0])
    else:
        group.append([0, 0, 1, 1])

    if rank & 1:
        group.append([0, 1, 0, 1])
    else:
        group.append([1, 0, 1, 0])

    group.append([1, 1, 1, 1])

    a = tf.Variable([float(rank)] * 8)

    groupf = tf.placeholder(tf.int32, (4,))
    b = preduce(a, groupf) / tf.cast(tf.reduce_sum(groupf), tf.float32)

    c = tf.assign(a, b)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.visible_device_list = str(rank)
    with tf.train.MonitoredTrainingSession(config=sess_config) as sess:
        for i in range(3):
            sess.run([c], feed_dict={
                groupf: np.array(group[i], dtype=np.int32)
            })
            av = sess.run([a])
            print(rank, i, av)


if __name__ == '__main__':
    _test()
