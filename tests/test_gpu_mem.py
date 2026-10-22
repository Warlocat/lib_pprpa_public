"""gpu_mem.pooled: blocks over gpu4pyscf's cudaMalloc threshold come from cupy's pool
inside the scope (up to _POOL_BLOCK_MAX), free_bytes is capped by the pool's limit there,
and gpu_multi's device threads inherit the scope."""

import pytest


def _gpu4pyscf_available():
    try:
        import cupy as cp
        import gpu4pyscf  # noqa: F401
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _gpu4pyscf_available(), reason="GPU4PySCF CUDA device is unavailable")

BIG = 128 * 1024**2     # bytes, over gpu4pyscf's 100 MiB threshold


def test_pooled_serves_large_blocks_from_the_pool():
    import cupy as cp
    import gpu4pyscf  # noqa: F401  installs the conditional allocator
    from lib_pprpa.gpu_mem import is_pooled, pooled
    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    before = pool.total_bytes()
    a = cp.empty(BIG, dtype=cp.uint8)
    assert pool.total_bytes() == before            # straight to cudaMalloc
    del a
    with pooled():
        with pooled():                              # nested scopes are one scope
            b = cp.empty(BIG, dtype=cp.uint8)
        assert is_pooled()
        assert pool.total_bytes() >= before + BIG
        del b
    assert not is_pooled()
    assert pool.total_bytes() == before            # free blocks released on exit


def test_free_bytes_inside_pooled_respects_the_pool_limit():
    import cupy as cp
    from lib_pprpa.gpu_mem import free_bytes, pooled
    pool = cp.get_default_memory_pool()
    limit = pool.get_limit()
    cap = 1024**3
    try:
        pool.set_limit(size=cap)
        assert free_bytes() > cap                   # outside: the driver's figure
        with pooled():
            keep = cp.empty(BIG, dtype=cp.uint8)
            assert free_bytes() <= cap - BIG
            del keep
    finally:
        pool.set_limit(size=limit)


def test_device_threads_inherit_the_scope():
    import cupy as cp
    from lib_pprpa.gpu_mem import is_pooled, pooled
    from lib_pprpa.gpu_multi import each, run
    two = [cp.cuda.Device().id] * 2
    assert each(two, lambda rank: is_pooled()) == [False, False]
    with pooled():
        assert each(two, lambda rank: is_pooled()) == [True, True]
        results, _ = run(two, lambda rank: None, lambda rank, st, task: is_pooled(), range(3))
        assert results == [True] * 3
    assert not is_pooled()


def test_pooled_leaves_blocks_over_the_cap_to_cudamalloc(monkeypatch):
    import cupy as cp
    import lib_pprpa.gpu_mem as gm
    monkeypatch.setattr(gm, "_POOL_BLOCK_MAX", BIG // 2)
    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    before = pool.total_bytes()
    with gm.pooled():
        a = cp.empty(BIG, dtype=cp.uint8)               # over the cap: not cached
        assert pool.total_bytes() == before
        b = cp.empty(BIG // 4, dtype=cp.uint8)          # under it: from the pool
        assert pool.total_bytes() >= before + BIG // 4
        del a, b
