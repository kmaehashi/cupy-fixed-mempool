# cupy-fixed-mempool

## CuPy

```
import chainer.cuda
from cupy.cuda import set_allocator
from fixed_mempool import FixedSizeMemoryPool

mp = FixedSizeMemoryPool(1024*1024*1024)
set_allocator(mp.malloc)
```

## Sensorbee

```
build_sensorbee
./sensorbee runfile example.bql
```
