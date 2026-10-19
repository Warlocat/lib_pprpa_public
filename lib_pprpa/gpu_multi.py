'''Opt-in dispatch of independent GPU tasks over several CUDA devices.

The kernels that use it are strip loops whose tasks are independent: ao2mo pair
strips, ket rows of the low-rank exchange, row ranges of the resident ERIs.
Each device runs one thread that pulls tasks from a shared list; the per-rank
state (MO grids, buffers) is built on the device by a ``setup`` callback, so
nothing is copied between devices.  Two threads on one device are not safe
(cuFFT and the MO-grid GEMMs gave wrong results under contention), so ranks
that share a device run one after another in that device's thread; naming a
device twice still exercises the split bookkeeping, which the tests use on one
GPU.  The default everywhere is the current device alone.
'''

import threading
from concurrent.futures import ThreadPoolExecutor

import cupy as cp

__all__ = ['default_devices', 'each', 'run']


def default_devices(devices=None):
    '''``devices`` as a list of CUDA device ids; None means the current device.'''
    if devices is None:
        return [cp.cuda.Device().id]
    return [int(d) for d in devices]


def _by_device(devices):
    groups = {}
    for rank, dev in enumerate(devices):
        groups.setdefault(dev, []).append(rank)
    return list(groups.items())


def each(devices, fn):
    '''Call ``fn(rank)`` once per rank, on its device; return the results in rank order.'''
    results = [None] * len(devices)

    def worker(item):
        dev, ranks = item
        with cp.cuda.Device(dev):
            for rank in ranks:
                results[rank] = fn(rank)
            cp.cuda.Stream.null.synchronize()

    groups = _by_device(devices)
    with ThreadPoolExecutor(len(groups)) as ex:
        list(ex.map(worker, groups))
    return results


def run(devices, setup, work, tasks):
    '''Run ``work(rank, state, task)`` for every task over the devices.

    One thread per distinct device pulls tasks from the shared list; ``state``
    is what ``setup(rank)`` returned on that device.  Returns
    ``(results, states)`` with the results in task order.
    '''
    queue = list(enumerate(tasks))
    lock = threading.Lock()
    results = [None] * len(queue)
    states = [None] * len(devices)

    def worker(item):
        dev, ranks = item
        with cp.cuda.Device(dev):
            for rank in ranks:
                states[rank] = state = setup(rank)
                while True:
                    with lock:
                        if not queue:
                            break
                        i, task = queue.pop(0)
                    results[i] = work(rank, state, task)
            cp.cuda.Stream.null.synchronize()

    groups = _by_device(devices)
    with ThreadPoolExecutor(len(groups)) as ex:
        list(ex.map(worker, groups))
    return results, states
