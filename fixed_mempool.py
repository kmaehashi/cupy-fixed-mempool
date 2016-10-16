# -*- coding: utf-8 -*-

import collections
import weakref

import six
from cupy.cuda import device, runtime, set_allocator
from cupy.cuda.memory import Memory, MemoryPointer, PooledMemory, _malloc


class FixedSizeSingleDeviceMemoryPool(object):
    def __init__(self, max_size, allocator=_malloc):
        self._in_use = {}
        self._free = collections.defaultdict(list)
        self._alloc = allocator
        self._weakref = weakref.ref(self)
        self._allocation_unit_size = 256
        self._max_size = max_size

    def malloc(self, size):
        if size == 0:
            return MemoryPointer(Memory(0), 0)

        # Round up the memory size to fit memory alignment of cudaMalloc
        unit = self._allocation_unit_size
        size = (((size + unit - 1) // unit) * unit)
        free = self._free[size]
        if free:
            mem = free.pop()
        else:
            try:
                mem = self._alloc(size).mem
            except runtime.CUDARuntimeError as e:
                if e.status != 2:
                    raise
                self.free_all_free()
                mem = self._alloc(size).mem

        self._in_use[mem.ptr] = mem
        pmem = PooledMemory(mem, self._weakref)
        return MemoryPointer(pmem, 0)

    def free(self, ptr, size):
        mem = self._in_use.pop(ptr, None)
        if mem is None:
            raise RuntimeError('Cannot free out-of-pool memory')

        total = sum(map(lambda l: sum(map(lambda m: m.size, l)), self._free.values()))
        if total + size < self._max_size:
            free = self._free[size]
            free.append(mem)

    def free_all_free(self):
        self._free = collections.defaultdict(list)

    def n_free_blocks(self):
        n = 0
        for v in six.itervalues(self._free):
            n += len(v)
        return n

class FixedSizeMemoryPool(object):
    def __init__(self, max_size, allocator=_malloc):
        self._pools = collections.defaultdict(
            lambda: FixedSizeSingleDeviceMemoryPool(max_size, allocator))

    def malloc(self, size):
        dev = device.get_device_id()
        return self._pools[dev].malloc(size)

    def free_all_free(self):
        dev = device.get_device_id()
        self._pools[dev].free_all_free()

    def n_free_blocks(self):
        dev = device.get_device_id()
        return self._pools[dev].n_free_blocks()

class FixedSizeMemoryPoolState(object):
    """
    SensorBee UDS to setup fixed-size memory pool for CuPy.
    Note that this state changes the process-wide memory pool, i.e.,
    all topologies are affected by loading this state.
    This state must not be loaded twice.
    """
    @classmethod
    def create(cls, max_size):
        state = cls(FixedSizeMemoryPool(max_size))
        return state

    def __init__(self, memory_pool):
        # The following line is essential to make sure that our memory allocator
        # setup is performed AFTER the default memory allocator setup, which runs
        # when ``chainer.cuda`` module is imported for the first time.
        import chainer.cuda

        self.memory_pool = memory_pool
        set_allocator(self.memory_pool.malloc)

    def terminate(self):
        if self.memory_pool:
            self.memory_pool.free_all_free()
            self.memory_pool = None
