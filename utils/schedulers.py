import tensorflow as tf
import numpy as np


def create_scheduler(scheduler, comm_rank, n_workers, gg=None):
    comm_local_rank = comm_rank % 4
    def _get_group_static(step):
        group = np.zeros(n_workers, dtype=np.int32)
        if step % 4 == 0:
            if comm_local_rank == 0:
                for i in range(0, n_workers, 4):
                    group[i] = 1
            elif comm_local_rank > 1:
                group[comm_rank ^ 1] = 1
        elif step % 4 == 1:
            for i in range(4):
                group[comm_rank // 4 * 4 + i] = 1
        elif step % 4 == 2:
            if comm_local_rank == 1:
                for i in range(1, n_workers, 4):
                    group[i] = 1
            elif comm_local_rank != 2:
                group[comm_rank // 4 * 4] = 1
                group[comm_rank // 4 * 4 + 3] = 1
        elif step % 4 == 3:
            for i in range(4):
                group[comm_rank // 4 * 4 + i] = 1
        group[comm_rank] = 1
        return group

    def _get_group_dynamic(step):
        group = np.zeros(n_workers, dtype=np.int32)
        members = gg.get_group(comm_rank)
        for i in range(n_workers):
            if members[i] == '1':
                group[i] = 1
        return group

    get_group_fn = _get_group_static if scheduler == 'static' \
            else _get_group_dynamic
    return lambda flag: tf.py_func(get_group_fn, [flag], tf.int32)
