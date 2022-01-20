# -------------------------------------------------------------------------------
#
# Numba extensions for general CUDA / OptiX support
#
# -------------------------------------------------------------------------------

from operator import add, mul, sub

from llvmlite import ir
from numba import cuda, float32, int32, types, uint8, uint32
from numba.core import cgutils
from numba.core.extending import (
    make_attribute_wrapper,
    models,
    overload,
    register_model,
    type_callable,
    typeof_impl,
)
from numba.core.imputils import lower_constant
from numba.core.typing.templates import (
    AttributeTemplate,
    ConcreteTemplate,
    signature,
)
from numba.cuda.cudadecl import register, register_attr, register_global
from numba.cuda.cudadrv import nvvm
from numba.cuda.cudaimpl import lower
from numba.cuda.types import dim3

import optix

# UChar4
# ------

# Numba presently doesn't implement the UChar4 type (which is fairly standard
# CUDA) so we provide some minimal support for it here.


# Prototype a function to construct a uchar4


def make_uchar4(x, y, z, w):
    pass


# UChar4 typing


class UChar4(types.Type):
    def __init__(self):
        super().__init__(name="UChar4")


uchar4 = UChar4()


@register
class MakeUChar4(ConcreteTemplate):
    key = make_uchar4
    cases = [signature(uchar4, types.uchar, types.uchar, types.uchar, types.uchar)]


register_global(make_uchar4, types.Function(MakeUChar4))


# UChar4 data model


@register_model(UChar4)
class UChar4Model(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("x", types.uchar),
            ("y", types.uchar),
            ("z", types.uchar),
            ("w", types.uchar),
        ]
        super().__init__(dmm, fe_type, members)


make_attribute_wrapper(UChar4, "x", "x")
make_attribute_wrapper(UChar4, "y", "y")
make_attribute_wrapper(UChar4, "z", "z")
make_attribute_wrapper(UChar4, "w", "w")


# UChar4 lowering


@lower(make_uchar4, types.uchar, types.uchar, types.uchar, types.uchar)
def lower_make_uchar4(context, builder, sig, args):
    uc4 = cgutils.create_struct_proxy(uchar4)(context, builder)
    uc4.x = args[0]
    uc4.y = args[1]
    uc4.z = args[2]
    uc4.w = args[3]
    return uc4._getvalue()


# float3
# ------

# Float3 typing


class Float3(types.Type):
    def __init__(self):
        super().__init__(name="Float3")


float3 = Float3()


# Float2 typing (forward declaration)


class Float2(types.Type):
    def __init__(self):
        super().__init__(name="Float2")


float2 = Float2()


# Float3 data model


@register_model(Float3)
class Float3Model(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("x", types.float32),
            ("y", types.float32),
            ("z", types.float32),
        ]
        super().__init__(dmm, fe_type, members)


make_attribute_wrapper(Float3, "x", "x")
make_attribute_wrapper(Float3, "y", "y")
make_attribute_wrapper(Float3, "z", "z")


def lower_float3_ops(op):
    class Float3_op_template(ConcreteTemplate):
        key = op
        cases = [
            signature(float3, float3, float3),
            signature(float3, types.float32, float3),
            signature(float3, float3, types.float32),
        ]

    def float3_op_impl(context, builder, sig, args):
        def op_attr(lhs, rhs, res, attr):
            setattr(
                res,
                attr,
                context.compile_internal(
                    builder,
                    lambda x, y: op(x, y),
                    signature(types.float32, types.float32, types.float32),
                    (getattr(lhs, attr), getattr(rhs, attr)),
                ),
            )

        arg0, arg1 = args

        if isinstance(sig.args[0], types.Float):
            lf3 = cgutils.create_struct_proxy(float3)(context, builder)
            lf3.x = arg0
            lf3.y = arg0
            lf3.z = arg0
        else:
            lf3 = cgutils.create_struct_proxy(float3)(context, builder, value=args[0])

        if isinstance(sig.args[1], types.Float):
            rf3 = cgutils.create_struct_proxy(float3)(context, builder)
            rf3.x = arg1
            rf3.y = arg1
            rf3.z = arg1
        else:
            rf3 = cgutils.create_struct_proxy(float3)(context, builder, value=args[1])

        res = cgutils.create_struct_proxy(float3)(context, builder)
        op_attr(lf3, rf3, res, "x")
        op_attr(lf3, rf3, res, "y")
        op_attr(lf3, rf3, res, "z")
        return res._getvalue()

    register_global(op, types.Function(Float3_op_template))
    lower(op, float3, float3)(float3_op_impl)
    lower(op, types.float32, float3)(float3_op_impl)
    lower(op, float3, types.float32)(float3_op_impl)


lower_float3_ops(mul)
lower_float3_ops(add)


@lower(add, float32, float3)
def add_float32_float3_impl(context, builder, sig, args):
    s = args[0]
    rhs = cgutils.create_struct_proxy(float3)(context, builder, args[1])
    res = cgutils.create_struct_proxy(float3)(context, builder)
    res.x = builder.fadd(s, rhs.x)
    res.y = builder.fadd(s, rhs.y)
    res.z = builder.fadd(s, rhs.z)
    return res._getvalue()


@lower(add, float3, float32)
def add_float3_float32_impl(context, builder, sig, args):
    lhs = cgutils.create_struct_proxy(float3)(context, builder, args[0])
    s = args[1]
    res = cgutils.create_struct_proxy(float3)(context, builder)
    res.x = builder.fadd(lhs.x, s)
    res.y = builder.fadd(lhs.y, s)
    res.z = builder.fadd(lhs.z, s)
    return res._getvalue()


# Prototype a function to construct a float3


def make_float3(x, y, z):
    pass


@register
class MakeFloat3(ConcreteTemplate):
    key = make_float3
    cases = [
        signature(float3, types.float32, types.float32, types.float32),
        signature(float3, float2, types.float32),
    ]


register_global(make_float3, types.Function(MakeFloat3))


# make_float3 lowering


@lower(make_float3, types.float32, types.float32, types.float32)
def lower_make_float3(context, builder, sig, args):
    f3 = cgutils.create_struct_proxy(float3)(context, builder)
    f3.x = args[0]
    f3.y = args[1]
    f3.z = args[2]
    return f3._getvalue()


@lower(make_float3, float2, types.float32)
def lower_make_float3(context, builder, sig, args):
    f2 = cgutils.create_struct_proxy(float2)(context, builder, args[0])
    f3 = cgutils.create_struct_proxy(float3)(context, builder)
    f3.x = f2.x
    f3.y = f2.y
    f3.z = args[1]
    return f3._getvalue()


# float2
# ------


# Float2 data model


@register_model(Float2)
class Float2Model(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("x", types.float32),
            ("y", types.float32),
        ]
        super().__init__(dmm, fe_type, members)


make_attribute_wrapper(Float2, "x", "x")
make_attribute_wrapper(Float2, "y", "y")


def lower_float2_ops(op):
    class Float2_op_template(ConcreteTemplate):
        key = op
        cases = [
            signature(float2, float2, float2),
            signature(float2, types.float32, float2),
            signature(float2, float2, types.float32),
        ]

    def float2_op_impl(context, builder, sig, args):
        def op_attr(lhs, rhs, res, attr):
            setattr(
                res,
                attr,
                context.compile_internal(
                    builder,
                    lambda x, y: op(x, y),
                    signature(types.float32, types.float32, types.float32),
                    (getattr(lhs, attr), getattr(rhs, attr)),
                ),
            )

        arg0, arg1 = args

        if isinstance(sig.args[0], types.Float):
            lf2 = cgutils.create_struct_proxy(float2)(context, builder)
            lf2.x = arg0
            lf2.y = arg0
        else:
            lf2 = cgutils.create_struct_proxy(float2)(context, builder, value=args[0])

        if isinstance(sig.args[1], types.Float):
            rf2 = cgutils.create_struct_proxy(float2)(context, builder)
            rf2.x = arg1
            rf2.y = arg1
        else:
            rf2 = cgutils.create_struct_proxy(float2)(context, builder, value=args[1])

        res = cgutils.create_struct_proxy(float2)(context, builder)
        op_attr(lf2, rf2, res, "x")
        op_attr(lf2, rf2, res, "y")
        return res._getvalue()

    register_global(op, types.Function(Float2_op_template))
    lower(op, float2, float2)(float2_op_impl)
    lower(op, types.Float, float2)(float2_op_impl)
    lower(op, float2, types.Float)(float2_op_impl)


lower_float2_ops(mul)
lower_float2_ops(sub)


# Prototype a function to construct a float2


def make_float2(x, y):
    pass


@register
class MakeFloat2(ConcreteTemplate):
    key = make_float2
    cases = [signature(float2, types.float32, types.float32)]


register_global(make_float2, types.Function(MakeFloat2))


# make_float2 lowering


@lower(make_float2, types.float32, types.float32)
def lower_make_float2(context, builder, sig, args):
    f2 = cgutils.create_struct_proxy(float2)(context, builder)
    f2.x = args[0]
    f2.y = args[1]
    return f2._getvalue()


# uint3
# ------


class UInt3(types.Type):
    def __init__(self):
        super().__init__(name="UInt3")


uint3 = UInt3()


# UInt3 data model


@register_model(UInt3)
class UInt3Model(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("x", types.uint32),
            ("y", types.uint32),
            ("z", types.uint32),
        ]
        super().__init__(dmm, fe_type, members)


make_attribute_wrapper(UInt3, "x", "x")
make_attribute_wrapper(UInt3, "y", "y")
make_attribute_wrapper(UInt3, "z", "z")


# Prototype a function to construct a uint3


def make_uint3(x, y, z):
    pass


@register
class MakeUInt3(ConcreteTemplate):
    key = make_uint3
    cases = [signature(uint3, types.uint32, types.uint32, types.uint32)]


register_global(make_uint3, types.Function(MakeUInt3))


# make_uint3 lowering


@lower(make_uint3, types.uint32, types.uint32, types.uint32)
def lower_make_uint3(context, builder, sig, args):
    # u4 = uint32
    u4_3 = cgutils.create_struct_proxy(uint3)(context, builder)
    u4_3.x = args[0]
    u4_3.y = args[1]
    u4_3.z = args[2]
    return u4_3._getvalue()


# Temporary Payload Parameter Pack
class PayloadPack(types.Type):
    def __init__(self):
        super().__init__(name="PayloadPack")


payload_pack = PayloadPack()


# UInt3 data model


@register_model(PayloadPack)
class PayloadPackModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("p0", types.uint32),
            ("p1", types.uint32),
            ("p2", types.uint32),
        ]
        super().__init__(dmm, fe_type, members)


make_attribute_wrapper(PayloadPack, "p0", "p0")
make_attribute_wrapper(PayloadPack, "p1", "p1")
make_attribute_wrapper(PayloadPack, "p2", "p2")

# OptiX typedefs and enums
# -----------

OptixVisibilityMask = types.Integer("OptixVisibilityMask", bitwidth=32, signed=False)
OptixTraversableHandle = types.Integer(
    "OptixTraversableHandle", bitwidth=64, signed=False
)


OPTIX_RAY_FLAG_NONE = 0
# class OptixRayFlags(Enum):
#     OPTIX_RAY_FLAG_NONE = 0
#     OPTIX_RAY_FLAG_DISABLE_ANYHIT = 1 << 0
#     OPTIX_RAY_FLAG_ENFORCE_ANYHIT = 1 << 1
#     OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT = 1 << 2
#     OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT = 1 << 3,
#     OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES = 1 << 4
#     OPTIX_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES = 1 << 5
#     OPTIX_RAY_FLAG_CULL_DISABLED_ANYHIT = 1 << 6
#     OPTIX_RAY_FLAG_CULL_ENFORCED_ANYHIT = 1 << 7


# OptiX types
# -----------

# Typing for OptiX types


class SbtDataPointer(types.RawPointer):
    def __init__(self):
        super().__init__(name="SbtDataPointer")


sbt_data_pointer = SbtDataPointer()


# Models for OptiX types


@register_model(SbtDataPointer)
class SbtDataPointerModel(models.OpaqueModel):
    pass


# Params
# ------------

# Structures as declared in triangle.h


class ParamsStruct:
    fields = (
        ("image", "uchar4*"),
        ("image_width", "unsigned int"),
        ("image_height", "unsigned int"),
        ("cam_eye", "float3"),
        ("cam_u", "float3"),
        ("cam_v", "float3"),
        ("cam_w", "float3"),
        ("handle", "OptixTraversableHandle"),
    )


# "Declare" a global called params
params = ParamsStruct()


class Params(types.Type):
    def __init__(self):
        super().__init__(name="ParamsType")


params_type = Params()


# ParamsStruct data model


@register_model(Params)
class ParamsModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("image", types.CPointer(uchar4)),
            ("image_width", types.uint32),
            ("image_height", types.uint32),
            ("cam_eye", float3),
            ("cam_u", float3),
            ("cam_v", float3),
            ("cam_w", float3),
            ("handle", OptixTraversableHandle),
        ]
        super().__init__(dmm, fe_type, members)


make_attribute_wrapper(Params, "image", "image")
make_attribute_wrapper(Params, "image_width", "image_width")
make_attribute_wrapper(Params, "image_height", "image_height")
make_attribute_wrapper(Params, "cam_eye", "cam_eye")
make_attribute_wrapper(Params, "cam_u", "cam_u")
make_attribute_wrapper(Params, "cam_v", "cam_v")
make_attribute_wrapper(Params, "cam_w", "cam_w")
make_attribute_wrapper(Params, "handle", "handle")


@typeof_impl.register(ParamsStruct)
def typeof_params(val, c):
    return params_type


# ParamsStruct lowering
# The below makes 'param' a global variable, accessible from any user defined
# kernels.


@lower_constant(Params)
def constant_params(context, builder, ty, pyval):
    try:
        gvar = builder.module.get_global("params")
    except KeyError:
        llty = context.get_value_type(ty)
        gvar = cgutils.add_global_variable(
            builder.module, llty, "params", addrspace=nvvm.ADDRSPACE_CONSTANT
        )
        gvar.linkage = "external"
        gvar.global_constant = True

    return builder.load(gvar)


# MissData
# ------------

# Structures as declared in triangle.h
class MissDataStruct:
    fields = ("bg_color", "float3")


MissData = MissDataStruct()


class MissData(types.Type):
    def __init__(self):
        super().__init__(name="MissDataType")


miss_data_type = MissData()


@register_model(MissData)
class MissDataModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("bg_color", float3),
        ]
        super().__init__(dmm, fe_type, members)


make_attribute_wrapper(MissData, "bg_color", "bg_color")


@typeof_impl.register(MissDataStruct)
def typeof_miss_data(val, c):
    return miss_data_type


# MissData Constructor
@type_callable(MissDataStruct)
def type_miss_data_struct(context):
    def typer(sbt_data_pointer):
        if isinstance(sbt_data_pointer, SbtDataPointer):
            return miss_data_type

    return typer


@lower(MissDataStruct, sbt_data_pointer)
def lower_miss_data_ctor(context, builder, sig, args):
    # Anyway to err if this ctor is not called inside __miss__* program?
    # TODO: Optimize
    ptr = args[0]
    ptr = builder.bitcast(ptr, context.get_value_type(miss_data_type).as_pointer())

    bg_color_ptr = cgutils.gep_inbounds(builder, ptr, 0, 0)

    xptr = cgutils.gep_inbounds(builder, bg_color_ptr, 0, 0)
    yptr = cgutils.gep_inbounds(builder, bg_color_ptr, 0, 1)
    zptr = cgutils.gep_inbounds(builder, bg_color_ptr, 0, 2)

    output_miss_data = cgutils.create_struct_proxy(miss_data_type)(context, builder)
    output_bg_color_ptr = cgutils.gep_inbounds(
        builder, output_miss_data._getpointer(), 0, 0
    )
    output_bg_color_x_ptr = cgutils.gep_inbounds(builder, output_bg_color_ptr, 0, 0)
    output_bg_color_y_ptr = cgutils.gep_inbounds(builder, output_bg_color_ptr, 0, 1)
    output_bg_color_z_ptr = cgutils.gep_inbounds(builder, output_bg_color_ptr, 0, 2)

    x = builder.load(xptr)
    y = builder.load(yptr)
    z = builder.load(zptr)

    builder.store(x, output_bg_color_x_ptr)
    builder.store(y, output_bg_color_y_ptr)
    builder.store(z, output_bg_color_z_ptr)

    # Doesn't seem to do what's expected?
    # miss_data.bg_color.x = builder.load(xptr)
    # miss_data.bg_color.y = builder.load(yptr)
    # miss_data.bg_color.z = builder.load(zptr)
    return output_miss_data._getvalue()


# OptiX functions
# ---------------

# Here we "prototype" the OptiX functions that the user will call in their
# kernels, so that Numba has something to refer to when compiling the kernel.


def _optix_GetLaunchIndex():
    pass


def _optix_GetLaunchDimensions():
    pass


def _optix_GetSbtDataPointer():
    pass


def _optix_SetPayload_0():
    pass


def _optix_SetPayload_1():
    pass


def _optix_SetPayload_2():
    pass


def _optix_GetTriangleBarycentrics():
    pass


def _optix_Trace():
    pass


# Monkey-patch the functions into the optix module, so the user can write
# optix.GetLaunchIndex etc., for symmetry with the rest of the API implemented
# in PyOptiX.

optix.GetLaunchIndex = _optix_GetLaunchIndex
optix.GetLaunchDimensions = _optix_GetLaunchDimensions
optix.GetSbtDataPointer = _optix_GetSbtDataPointer
optix.GetTriangleBarycentrics = _optix_GetTriangleBarycentrics
optix.SetPayload_0 = _optix_SetPayload_0
optix.SetPayload_1 = _optix_SetPayload_1
optix.SetPayload_2 = _optix_SetPayload_2

optix.Trace = _optix_Trace


# OptiX function typing


@register
class OptixGetLaunchIndex(ConcreteTemplate):
    key = optix.GetLaunchIndex
    cases = [signature(dim3)]


@register
class OptixGetLaunchDimensions(ConcreteTemplate):
    key = optix.GetLaunchDimensions
    cases = [signature(dim3)]


@register
class OptixGetSbtDataPointer(ConcreteTemplate):
    key = optix.GetSbtDataPointer
    cases = [signature(sbt_data_pointer)]


def registerSetPayload(reg):
    class OptixSetPayloadReg(ConcreteTemplate):
        key = getattr(optix, "SetPayload_" + str(reg))
        cases = [signature(types.void, uint32)]

    register(OptixSetPayloadReg)
    return OptixSetPayloadReg


OptixSetPayload_0 = registerSetPayload(0)
OptixSetPayload_1 = registerSetPayload(1)
OptixSetPayload_2 = registerSetPayload(2)


@register
class OptixGetTriangleBarycentrics(ConcreteTemplate):
    key = optix.GetTriangleBarycentrics
    cases = [signature(float2)]


@register
class OptixTrace(ConcreteTemplate):
    key = optix.Trace
    cases = [
        signature(
            payload_pack,
            OptixTraversableHandle,
            float3,
            float3,
            float32,
            float32,
            float32,
            OptixVisibilityMask,
            uint32,
            uint32,
            uint32,
            uint32,
        )
    ]


@register_attr
class OptixModuleTemplate(AttributeTemplate):
    key = types.Module(optix)

    def resolve_GetLaunchIndex(self, mod):
        return types.Function(OptixGetLaunchIndex)

    def resolve_GetLaunchDimensions(self, mod):
        return types.Function(OptixGetLaunchDimensions)

    def resolve_GetSbtDataPointer(self, mod):
        return types.Function(OptixGetSbtDataPointer)

    def resolve_SetPayload_0(self, mod):
        return types.Function(OptixSetPayload_0)

    def resolve_SetPayload_1(self, mod):
        return types.Function(OptixSetPayload_1)

    def resolve_SetPayload_2(self, mod):
        return types.Function(OptixSetPayload_2)

    def resolve_GetTriangleBarycentrics(self, mod):
        return types.Function(OptixGetTriangleBarycentrics)

    def resolve_Trace(self, mod):
        return types.Function(OptixTrace)


# OptiX function lowering


@lower(optix.GetLaunchIndex)
def lower_optix_getLaunchIndex(context, builder, sig, args):
    def get_launch_index(axis):
        asm = ir.InlineAsm(
            ir.FunctionType(ir.IntType(32), []),
            f"call ($0), _optix_get_launch_index_{axis}, ();",
            "=r",
        )
        return builder.call(asm, [])

    index = cgutils.create_struct_proxy(dim3)(context, builder)
    index.x = get_launch_index("x")
    index.y = get_launch_index("y")
    index.z = get_launch_index("z")
    return index._getvalue()


@lower(optix.GetLaunchDimensions)
def lower_optix_getLaunchDimensions(context, builder, sig, args):
    def get_launch_dimensions(axis):
        asm = ir.InlineAsm(
            ir.FunctionType(ir.IntType(32), []),
            f"call ($0), _optix_get_launch_dimension_{axis}, ();",
            "=r",
        )
        return builder.call(asm, [])

    index = cgutils.create_struct_proxy(dim3)(context, builder)
    index.x = get_launch_dimensions("x")
    index.y = get_launch_dimensions("y")
    index.z = get_launch_dimensions("z")
    return index._getvalue()


@lower(optix.GetSbtDataPointer)
def lower_optix_getSbtDataPointer(context, builder, sig, args):
    asm = ir.InlineAsm(
        ir.FunctionType(ir.IntType(64), []),
        "call ($0), _optix_get_sbt_data_ptr_64, ();",
        "=l",
    )
    ptr = builder.call(asm, [])
    ptr = builder.inttoptr(ptr, ir.IntType(8).as_pointer())
    return ptr


def lower_optix_SetPayloadReg(reg):
    def lower_optix_SetPayload_impl(context, builder, sig, args):
        asm = ir.InlineAsm(
            ir.FunctionType(ir.VoidType(), [ir.IntType(32), ir.IntType(32)]),
            f"call _optix_set_payload, ($0, $1);",
            "r,r",
        )
        builder.call(asm, [context.get_constant(types.int32, reg), args[0]])

    lower(getattr(optix, f"SetPayload_{reg}"), uint32)(lower_optix_SetPayload_impl)


lower_optix_SetPayloadReg(0)
lower_optix_SetPayloadReg(1)
lower_optix_SetPayloadReg(2)


@lower(optix.GetTriangleBarycentrics)
def lower_optix_getTriangleBarycentrics(context, builder, sig, args):
    f2 = cgutils.create_struct_proxy(float2)(context, builder)
    retty = ir.LiteralStructType([ir.FloatType(), ir.FloatType()])
    asm = ir.InlineAsm(
        ir.FunctionType(retty, []),
        "call ($0, $1), _optix_get_triangle_barycentrics, ();",
        "=f,=f",
    )
    ret = builder.call(asm, [])
    f2.x = builder.extract_value(ret, 0)
    f2.y = builder.extract_value(ret, 1)
    return f2._getvalue()


@lower(
    optix.Trace,
    OptixTraversableHandle,
    float3,
    float3,
    float32,
    float32,
    float32,
    OptixVisibilityMask,
    uint32,
    uint32,
    uint32,
    uint32,
)
def lower_optix_Trace(context, builder, sig, args):
    # Only implements the version that accepts 3 payload registers
    # TODO: Optimize returns, adapt to 0-8 payload registers.

    (
        handle,
        rayOrigin,
        rayDirection,
        tmin,
        tmax,
        rayTime,
        visibilityMask,
        rayFlags,
        SBToffset,
        SBTstride,
        missSBTIndex,
    ) = args

    rayOrigin = cgutils.create_struct_proxy(float3)(context, builder, rayOrigin)
    rayDirection = cgutils.create_struct_proxy(float3)(context, builder, rayDirection)
    output = cgutils.create_struct_proxy(payload_pack)(context, builder)

    ox, oy, oz = rayOrigin.x, rayOrigin.y, rayOrigin.z
    dx, dy, dz = rayDirection.x, rayDirection.y, rayDirection.z

    n_payload_registers = 3
    n_stub_output_operands = 32 - n_payload_registers
    outputs = [output.p0, output.p1, output.p2] + [
        builder.load(builder.alloca(ir.IntType(32)))
        for _ in range(n_stub_output_operands)
    ]

    retty = ir.LiteralStructType([ir.IntType(32)] * 32)
    asm = ir.InlineAsm(
        ir.FunctionType(retty, []),
        "call "
        "($0,$1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22,$23,$24,$25,$26,$27,$28,$29,"
        "$30,$31),"
        "_optix_trace_typed_32,"
        "($32,$33,$34,$35,$36,$37,$38,$39,$40,$41,$42,$43,$44,$45,$46,$47,$48,$49,$50,$51,$52,$53,$54,$55,$56,$57,$58,$59,"
        "$60,$61,$62,$63,$64,$65,$66,$67,$68,$69,$70,$71,$72,$73,$74,$75,$76,$77,$78,$79,$80);",
        "=r," * 32 + "r,l,f,f,f,f,f,f,f,f,f,r,r,r,r,r,r," + "r," * 31 + "r",
        side_effect=True,
    )

    zero = context.get_constant(types.int32, 0)
    c_payload_registers = context.get_constant(types.int32, n_payload_registers)
    args = [
        zero,
        handle,
        ox,
        oy,
        oz,
        dx,
        dy,
        dz,
        tmin,
        tmax,
        rayTime,
        visibilityMask,
        rayFlags,
        SBToffset,
        SBTstride,
        missSBTIndex,
        c_payload_registers,
    ] + outputs
    ret = builder.call(asm, args)
    output.p0 = builder.extract_value(ret, 0)
    output.p1 = builder.extract_value(ret, 1)
    output.p2 = builder.extract_value(ret, 2)
    return output._getvalue()
