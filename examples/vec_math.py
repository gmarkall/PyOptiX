import math

from numba import cuda, float32, types
from numba.core.extending import overload
from numba_support import float3, make_float3

__all__ = ["clamp", "dot", "normalize"]


def clamp(x, a, b):
    pass


@overload(clamp, target="cuda", fastmath=True)
def jit_clamp(x, a, b):
    if (
        isinstance(x, types.Float)
        and isinstance(a, types.Float)
        and isinstance(b, types.Float)
    ):

        def clamp_float_impl(x, a, b):
            return max(a, min(x, b))

        return clamp_float_impl
    elif (
        isinstance(x, type(float3))
        and isinstance(a, types.Float)
        and isinstance(b, types.Float)
    ):

        def clamp_float3_impl(x, a, b):
            return make_float3(clamp(x.x, a, b), clamp(x.y, a, b), clamp(x.z, a, b))

        return clamp_float3_impl


def dot(a, b):
    pass


@overload(dot, target="cuda", fastmath=True)
def jit_dot(a, b):
    if isinstance(a, type(float3)) and isinstance(b, type(float3)):

        def dot_float3_impl(a, b):
            return a.x * b.x + a.y * b.y + a.z * b.z

        return dot_float3_impl


@cuda.jit(device=True, fastmath=True)
def normalize(v):
    invLen = float32(1.0) / math.sqrt(dot(v, v))
    return v * invLen
