import tensorflow as tf
import numpy as np


def create_scheduler(scheduler, comm_rank, n_workers, gg=None):
    def _get_group_static(step):
        group = np.zeros(n_workers, dtype=np.int32)
        lowbit = max(step & (-step), 1)
        for i in range(n_workers):
            if comm_rank // lowbit == i // lowbit:
                group[i] = 1
        return group

    def _get_group_dynamic(step):
        group = np.zeros(n_workers, dtype=np.int32)
        if step % FLAGS.section_length == 0:
            members = gg.get_group(comm_rank)
            for i in range(n_workers):
                if members[i] == '1':
                    group[i] = 1
        else:
            group[comm_rank] = 1
        return group

    get_group_fn = _get_group_static if scheduler == 'static' \
            else _get_group_dynamic
    return lambda flag: tf.py_func(get_group_fn, [flag], tf.int32)
