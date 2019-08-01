import os
import sys
import tensorflow as tf


_libpath = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), './preduce.so')
_lib = tf.load_op_library(_libpath)


preduce = _lib.p_reduce
psync = _lib.p_reduce_sync


def _test():
    rank = int(os.environ['PMI_RANK'])
    if rank < 2:
        group = tf.constant([1, 1, 0, 0], dtype=tf.int32)
    else:
        group = tf.constant([0, 0, 1, 1], dtype=tf.int32)

    sess_config = tf.ConfigProto(
              allow_soft_placement=True)
    sess_config.gpu_options.visible_device_list = str(rank)

    with tf.device('/GPU'):
        a = tf.Variable([1.])
        b = preduce(a, group)
    with tf.control_dependencies([b]):
        _sync = psync()

    with tf.train.MonitoredTrainingSession(config=sess_config) as sess:
        bv, _ = sess.run([b, _sync])
        print(rank, bv)


if __name__ == '__main__':
    _test()
