#!/usr/bin/env python3


import ctypes  # C interop helpers
import math
from operator import add, mul, sub
from enum import Enum

import cupy as cp  # CUDA bindings
import numpy as np  # Packing of structures in C-compatible format
import optix

from llvmlite import ir

from numba import cuda, float32, types, uint8, uint32
from numba.core import cgutils
from numba.core.extending import (make_attribute_wrapper, models, overload,
                                  register_model, typeof_impl)
from numba.core.imputils import lower_constant
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
                                         signature)

from numba.cuda import get_current_device
from numba.cuda.compiler import compile_cuda as numba_compile_cuda
from numba.cuda.cudadecl import register, register_attr, register_global
from numba.cuda.cudadrv import nvvm
from numba.cuda.cudaimpl import lower
from numba.cuda.types import dim3
from PIL import Image, ImageOps  # Image IO

# -------------------------------------------------------------------------------
#
# Numba extensions for general CUDA / OptiX support
#
# -------------------------------------------------------------------------------

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
    cases = [signature(uchar4, types.uchar, types.uchar, types.uchar,
                       types.uchar)]


register_global(make_uchar4, types.Function(MakeUChar4))


# UChar4 data model

@register_model(UChar4)
class UChar4Model(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('x', types.uchar),
            ('y', types.uchar),
            ('z', types.uchar),
            ('w', types.uchar),
        ]
        super().__init__(dmm, fe_type, members)


make_attribute_wrapper(UChar4, 'x', 'x')
make_attribute_wrapper(UChar4, 'y', 'y')
make_attribute_wrapper(UChar4, 'z', 'z')
make_attribute_wrapper(UChar4, 'w', 'w')


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


# Float3 data model

@register_model(Float3)
class Float3Model(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('x', types.float32),
            ('y', types.float32),
            ('z', types.float32),
        ]
        super().__init__(dmm, fe_type, members)


make_attribute_wrapper(Float3, 'x', 'x')
make_attribute_wrapper(Float3, 'y', 'y')
make_attribute_wrapper(Float3, 'z', 'z')


def lower_float3_ops(op):
    class Float3_op_template(ConcreteTemplate):
        key = op
        cases = [
            signature(float3, float3, float3),
            signature(float3, types.float32, float3),
            signature(float3, float3, types.float32)
        ]

    def float3_op_impl(context, builder, sig, args):
        def op_attr(lhs, rhs, res, attr):
            setattr(res, attr, context.compile_internal(
                builder,
                lambda x, y: op(x, y),
                signature(types.float32, types.float32, types.float32),
                (getattr(lhs, attr), getattr(rhs, attr))
            ))

        arg0, arg1 = args

        if isinstance(sig.args[0], types.Float):
            lf3 = cgutils.create_struct_proxy(float3)(context, builder)
            lf3.x = arg0
            lf3.y = arg0
            lf3.z = arg0
        else:
            lf3 = cgutils.create_struct_proxy(float3)(context, builder,
                                                      value=args[0])

        if isinstance(sig.args[1], types.Float):
            rf3 = cgutils.create_struct_proxy(float3)(context, builder)
            rf3.x = arg1
            rf3.y = arg1
            rf3.z = arg1
        else:
            rf3 = cgutils.create_struct_proxy(float3)(context, builder,
                                                      value=args[1])

        res = cgutils.create_struct_proxy(float3)(context, builder)
        op_attr(lf3, rf3, res, 'x')
        op_attr(lf3, rf3, res, 'y')
        op_attr(lf3, rf3, res, 'z')
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
    cases = [signature(float3, types.float32, types.float32, types.float32)]


register_global(make_float3, types.Function(MakeFloat3))


# make_float3 lowering

@lower(make_float3, types.float32, types.float32, types.float32)
def lower_make_float3(context, builder, sig, args):
    f3 = cgutils.create_struct_proxy(float3)(context, builder)
    f3.x = args[0]
    f3.y = args[1]
    f3.z = args[2]
    return f3._getvalue()


# float2
# ------

# Float2 typing

class Float2(types.Type):
    def __init__(self):
        super().__init__(name="Float2")


float2 = Float2()


# Float2 data model

@register_model(Float2)
class Float2Model(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('x', types.float32),
            ('y', types.float32),
        ]
        super().__init__(dmm, fe_type, members)


make_attribute_wrapper(Float2, 'x', 'x')
make_attribute_wrapper(Float2, 'y', 'y')


def lower_float2_ops(op):
    class Float2_op_template(ConcreteTemplate):
        key = op
        cases = [
            signature(float2, float2, float2),
            signature(float2, types.float32, float2),
            signature(float2, float2, types.float32)
        ]

    def float2_op_impl(context, builder, sig, args):
        def op_attr(lhs, rhs, res, attr):
            setattr(res, attr, context.compile_internal(
                builder,
                lambda x, y: op(x, y),
                signature(types.float32, types.float32, types.float32),
                (getattr(lhs, attr), getattr(rhs, attr))
            ))

        arg0, arg1 = args

        if isinstance(sig.args[0], types.Float):
            lf2 = cgutils.create_struct_proxy(float2)(context, builder)
            lf2.x = arg0
            lf2.y = arg0
        else:
            lf2 = cgutils.create_struct_proxy(float2)(context, builder,
                                                      value=args[0])

        if isinstance(sig.args[1], types.Float):
            rf2 = cgutils.create_struct_proxy(float2)(context, builder)
            rf2.x = arg1
            rf2.y = arg1
        else:
            rf2 = cgutils.create_struct_proxy(float2)(context, builder,
                                                      value=args[1])

        res = cgutils.create_struct_proxy(float2)(context, builder)
        op_attr(lf2, rf2, res, 'x')
        op_attr(lf2, rf2, res, 'y')
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
            ('x', types.uint32),
            ('y', types.uint32),
            ('z', types.uint32),
        ]
        super().__init__(dmm, fe_type, members)


make_attribute_wrapper(UInt3, 'x', 'x')
make_attribute_wrapper(UInt3, 'y', 'y')
make_attribute_wrapper(UInt3, 'z', 'z')


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


# OptiX typedefs and enums
# -----------

OptixVisibilityMask = types.Integer('OptixVisibilityMask', bitwidth=32,
                                    signed=False)
OptixTraversableHandle = types.Integer('OptixTraversableHandle', bitwidth=64,
                                       signed=False)


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


# Params
# ------------

# Structures as declared in triangle.h

class ParamsStruct:
    fields = (
        ('image', 'uchar4*'),
        ('image_width', 'unsigned int'),
        ('image_height', 'unsigned int'),
        ('cam_eye', 'float3'),
        ('cam_u', 'float3'),
        ('cam_v', 'float3'),
        ('cam_w', 'float3'),
        ('handle', 'OptixTraversableHandle'),
    )


class MissDataStruct:
    fields = {
        ('bg_color', 'float3')
    }


# "Declare" a global called params

params = ParamsStruct()
MissData = MissDataStruct()


class Params(types.Type):
    def __init__(self):
        super().__init__(name='ParamsType')


params_type = Params()


# ParamsStruct data model

@register_model(Params)
class ParamsModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('image', types.CPointer(uchar4)),
            ('image_width', types.uint32),
            ('image_height', types.uint32),
            ('cam_eye', float3),
            ('cam_u', float3),
            ('cam_v', float3),
            ('cam_w', float3),
            ('handle', OptixTraversableHandle),
        ]
        super().__init__(dmm, fe_type, members)


make_attribute_wrapper(Params, 'image', 'image')
make_attribute_wrapper(Params, 'image_width', 'image_width')
make_attribute_wrapper(Params, 'image_height', 'image_height')
make_attribute_wrapper(Params, 'cam_eye', 'cam_eye')
make_attribute_wrapper(Params, 'cam_u', 'cam_u')
make_attribute_wrapper(Params, 'cam_v', 'cam_v')
make_attribute_wrapper(Params, 'cam_w', 'cam_w')
make_attribute_wrapper(Params, 'handle', 'handle')


@typeof_impl.register(ParamsStruct)
def typeof_params(val, c):
    return params_type


# ParamsStruct lowering

@lower_constant(Params)
def constant_params(context, builder, ty, pyval):
    try:
        gvar = builder.module.get_global('params')
    except KeyError:
        llty = context.get_value_type(ty)
        gvar = cgutils.add_global_variable(builder.module, llty, 'params',
                                           addrspace=nvvm.ADDRSPACE_CONSTANT)
        gvar.linkage = 'external'
        gvar.global_constant = True

    return builder.load(gvar)


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


def _optix_Trace():
    pass


# Monkey-patch the functions into the optix module, so the user can write
# optix.GetLaunchIndex etc., for symmetry with the rest of the API implemented
# in PyOptiX.

optix.GetLaunchIndex = _optix_GetLaunchIndex
optix.GetLaunchDimensions = _optix_GetLaunchDimensions
optix.GetSbtDataPointer = _optix_GetSbtDataPointer
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
class OptixTrace(ConcreteTemplate):
    key = optix.Trace
    cases = [signature(
        types.void,
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
        uint32, # payload register 0
        uint32, # payload register 1
        uint32, # payload register 2
    )]


@register_attr
class OptixModuleTemplate(AttributeTemplate):
    key = types.Module(optix)

    def resolve_GetLaunchIndex(self, mod):
        return types.Function(OptixGetLaunchIndex)

    def resolve_GetLaunchDimensions(self, mod):
        return types.Function(OptixGetLaunchDimensions)

    def resolve_GetSbtDataPointer(self, mod):
        return types.Function(OptixGetSbtDataPointer)
    
    def resolve_Trace(self, mod):
        return types.Function(OptixTrace)


# OptiX function lowering

@lower(optix.GetLaunchIndex)
def lower_optix_getLaunchIndex(context, builder, sig, args):
    def get_launch_index(axis):
        asm = ir.InlineAsm(ir.FunctionType(ir.IntType(32), []),
                           f"call ($0), _optix_get_launch_index_{axis}, ();",
                           "=r")
        return builder.call(asm, [])

    index = cgutils.create_struct_proxy(dim3)(context, builder)
    index.x = get_launch_index('x')
    index.y = get_launch_index('y')
    index.z = get_launch_index('z')
    return index._getvalue()


@lower(optix.GetLaunchDimensions)
def lower_optix_getLaunchDimensions(context, builder, sig, args):
    def get_launch_dimensions(axis):
        asm = ir.InlineAsm(ir.FunctionType(ir.IntType(32), []),
                           f"call ($0), _optix_get_launch_dimension_{axis}, ();",
                           "=r")
        return builder.call(asm, [])

    index = cgutils.create_struct_proxy(dim3)(context, builder)
    index.x = get_launch_dimensions('x')
    index.y = get_launch_dimensions('y')
    index.z = get_launch_dimensions('z')
    return index._getvalue()


@lower(optix.GetSbtDataPointer)
def lower_optix_getSbtDataPointer(context, builder, sig, args):
    asm = ir.InlineAsm(ir.FunctionType(ir.IntType(64), []),
                       "call ($0), _optix_get_sbt_data_ptr_64, ();",
                       "=l")
    ptr = builder.call(asm, [])
    ptr = builder.inttoptr(ptr, ir.IntType(8).as_pointer())
    return ptr


@lower(optix.Trace,
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
        uint32, # payload register 0
        uint32, # payload register 1
        uint32, # payload register 2
)
def lower_optix_Trace(context, builder, sig, args):
    # Only implements the version that accepts 3 payload registers

    (handle, rayOrigin, rayDirection, tmin, tmax, rayTime, visibilityMask,
    rayFlags, SBToffset, SBTstride, missSBTIndex, p0, p1, p2) = args

    rayOrigin = cgutils.create_struct_proxy(float3)(context, builder, rayOrigin)
    rayDirection = cgutils.create_struct_proxy(float3)(context, builder, rayDirection)

    ox, oy, oz = rayOrigin.x, rayOrigin.y, rayOrigin.z
    dx, dy, dz = rayDirection.x, rayDirection.y, rayDirection.z

    n_stub_output_operands = 32 - 3
    output_stubs = [builder.alloca(ir.IntType(32)) for _ in range(n_stub_output_operands)]

    asm = ir.InlineAsm(
        ir.FunctionType(ir.VoidType(), []),
        "call "
        "(%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%"
        "29,%30,%31),"
        "_optix_trace_typed_32,"
        "(%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%"
        "59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80);",
        "=r," * 32 + "r,l,f,f,f,f,f,f,f,f,f,r,r,r,r,r,r," + "r," * 31 + "r",
        [p0, p1, p2] + output_stubs + [0, handle, ox, oy, oz, dx, dy, dz, tmin, tmax, rayTime, visibilityMask, rayFlags, SBToffset, SBTstride, missSBTIndex, 3, p0, p1, p2] + output_stubs
    )
    return builder.call(asm, [])


#-------------------------------------------------------------------------------
#
# Util
#
# -------------------------------------------------------------------------------
pix_width = 1024
pix_height = 768


class Logger:
    def __init__(self):
        self.num_mssgs = 0

    def __call__(self, level, tag, mssg):
        print("[{:>2}][{:>12}]: {}".format(level, tag, mssg))
        self.num_mssgs += 1


def log_callback(level, tag, mssg):
    print("[{:>2}][{:>12}]: {}".format(level, tag, mssg))


def round_up(val, mult_of):
    return val if val % mult_of == 0 else val + mult_of - val % mult_of


def get_aligned_itemsize(formats, alignment):
    names = []
    for i in range(len(formats)):
        names.append( 'x'+str(i) )

    temp_dtype = np.dtype( { 
        'names'   : names,
        'formats' : formats, 
        'align'   : True
        } )
    return round_up( temp_dtype.itemsize, alignment )


def array_to_device_memory( numpy_array, stream=cp.cuda.Stream() ):

    byte_size = numpy_array.size*numpy_array.dtype.itemsize

    h_ptr = ctypes.c_void_p( numpy_array.ctypes.data )
    d_mem = cp.cuda.memory.alloc( byte_size )
    d_mem.copy_from_async( h_ptr, byte_size, stream )
    return d_mem


def compile_cuda( cuda_file ):
    with open( cuda_file, 'rb' ) as f:
        src = f.read()
    from pynvrtc.compiler import Program
    prog = Program( src.decode(), cuda_file )
    ptx  = prog.compile( [
        '-use_fast_math', 
        '-lineinfo',
        '-default-device',
        '-std=c++11',
        '-rdc',
        'true',
        #'-IC:\\ProgramData\\NVIDIA Corporation\OptiX SDK 7.2.0\include',
        #'-IC:\\Program Files\\NVIDIA GPU Computing Toolkit\CUDA\\v11.1\include'
        '-I/usr/local/cuda/include',
        f'-I{optix.include_path}'
        ] )
    return ptx


# -------------------------------------------------------------------------------
#
# Optix setup
#
# -------------------------------------------------------------------------------

def init_optix():
    print( "Initializing cuda ..." )
    cp.cuda.runtime.free( 0 )

    print( "Initializing optix ..." )
    optix.init()


def create_ctx():
    print( "Creating optix device context ..." )

    # Note that log callback data is no longer needed.  We can
    # instead send a callable class instance as the log-function
    # which stores any data needed
    global logger
    logger = Logger()
    
    # OptiX param struct fields can be set with optional
    # keyword constructor arguments.
    ctx_options = optix.DeviceContextOptions( 
            logCallbackFunction = logger,
            logCallbackLevel    = 4
            )

    # They can also be set and queried as properties on the struct
    ctx_options.validationMode = optix.DEVICE_CONTEXT_VALIDATION_MODE_ALL 

    cu_ctx = 0 
    return optix.deviceContextCreate( cu_ctx, ctx_options )


def create_accel( ctx ):
        
    accel_options = optix.AccelBuildOptions(
        buildFlags = int( optix.BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS),
        operation  = optix.BUILD_OPERATION_BUILD
        )

    global vertices
    vertices = cp.array( [
       -0.5, -0.5, 0.0,
        0.5, -0.5, 0.0,
        0.0,  0.5, 0.0 
        ], dtype = 'f4')
        
    triangle_input_flags = [ optix.GEOMETRY_FLAG_NONE ]
    triangle_input = optix.BuildInputTriangleArray()
    triangle_input.vertexFormat  = optix.VERTEX_FORMAT_FLOAT3
    triangle_input.numVertices   = len( vertices )
    triangle_input.vertexBuffers = [ vertices.data.ptr ]
    triangle_input.flags         = triangle_input_flags
    triangle_input.numSbtRecords = 1;
        
    gas_buffer_sizes = ctx.accelComputeMemoryUsage( [accel_options], [triangle_input] )

    d_temp_buffer_gas   = cp.cuda.alloc( gas_buffer_sizes.tempSizeInBytes )
    d_gas_output_buffer = cp.cuda.alloc( gas_buffer_sizes.outputSizeInBytes)
    
    gas_handle = ctx.accelBuild( 
        0,    # CUDA stream
        [ accel_options ],
        [ triangle_input ],
        d_temp_buffer_gas.ptr,
        gas_buffer_sizes.tempSizeInBytes,
        d_gas_output_buffer.ptr,
        gas_buffer_sizes.outputSizeInBytes,
        [] # emitted properties
        )

    return (gas_handle, d_gas_output_buffer)


def set_pipeline_options():
    return optix.PipelineCompileOptions(
        usesMotionBlur         = False,
        traversableGraphFlags  = 
            int( optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS ),
        numPayloadValues       = 3,
        numAttributeValues     = 3,
        exceptionFlags         = int( optix.EXCEPTION_FLAG_NONE ),
        pipelineLaunchParamsVariableName = "params",
        usesPrimitiveTypeFlags = optix.PRIMITIVE_TYPE_FLAGS_TRIANGLE
        )


def create_module( ctx, pipeline_options, triangle_ptx ):
    print( "Creating optix module ..." )
    

    module_options = optix.ModuleCompileOptions(
        maxRegisterCount = optix.COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        optLevel         = optix.COMPILE_OPTIMIZATION_DEFAULT,
        debugLevel       = optix.COMPILE_DEBUG_LEVEL_LINEINFO
    )

    module, log = ctx.moduleCreateFromPTX(
            module_options,
            pipeline_options,
            triangle_ptx
            )
    print( "\tModule create log: <<<{}>>>".format( log ) )
    return module


def create_program_groups( ctx, module ):
    print( "Creating program groups ... " )

    program_group_options = optix.ProgramGroupOptions()

    raygen_prog_group_desc                          = optix.ProgramGroupDesc()
    raygen_prog_group_desc.raygenModule             = module
    raygen_prog_group_desc.raygenEntryFunctionName  = "__raygen__rg"
    raygen_prog_group, log = ctx.programGroupCreate(
            [ raygen_prog_group_desc ], 
            program_group_options,
            )
    print( "\tProgramGroup raygen create log: <<<{}>>>".format( log ) )
    
    miss_prog_group_desc                        = optix.ProgramGroupDesc()
    miss_prog_group_desc.missModule             = module
    miss_prog_group_desc.missEntryFunctionName  = "__miss__ms"
    miss_prog_group, log = ctx.programGroupCreate(
            [ miss_prog_group_desc ], 
            program_group_options,
            )
    print( "\tProgramGroup miss create log: <<<{}>>>".format( log ) )
    

    hitgroup_prog_group_desc                             = optix.ProgramGroupDesc()
    hitgroup_prog_group_desc.hitgroupModuleCH            = module
    hitgroup_prog_group_desc.hitgroupEntryFunctionNameCH = "__closesthit__ch"
    hitgroup_prog_group, log = ctx.programGroupCreate(
            [ hitgroup_prog_group_desc ], 
            program_group_options,
            )
    print( "\tProgramGroup hitgroup create log: <<<{}>>>".format( log ) )


    return [ raygen_prog_group[0], miss_prog_group[0], hitgroup_prog_group[0] ]


def create_pipeline( ctx, program_groups, pipeline_compile_options ):
    print( "Creating pipeline ... " )

    max_trace_depth  = 1
    pipeline_link_options               = optix.PipelineLinkOptions() 
    pipeline_link_options.maxTraceDepth = max_trace_depth
    pipeline_link_options.debugLevel    = optix.COMPILE_DEBUG_LEVEL_FULL

    log = ""
    pipeline = ctx.pipelineCreate(
            pipeline_compile_options,
            pipeline_link_options,
            program_groups,
            log)

    stack_sizes = optix.StackSizes()
    for prog_group in program_groups:
        optix.util.accumulateStackSizes( prog_group, stack_sizes )

    (dc_stack_size_from_trav, dc_stack_size_from_state, cc_stack_size) = \
        optix.util.computeStackSizes( 
            stack_sizes, 
            max_trace_depth,
            0,  # maxCCDepth
            0   # maxDCDepth
            )
    
    pipeline.setStackSize( 
            dc_stack_size_from_trav,
            dc_stack_size_from_state, 
            cc_stack_size,
            1  # maxTraversableDepth
            )

    return pipeline


def create_sbt( prog_groups ):
    print( "Creating sbt ... " )

    (raygen_prog_group, miss_prog_group, hitgroup_prog_group ) = prog_groups

    global d_raygen_sbt
    global d_miss_sbt

    header_format = '{}B'.format( optix.SBT_RECORD_HEADER_SIZE )

    #
    # raygen record
    #
    formats  = [ header_format ]
    itemsize = get_aligned_itemsize( formats, optix.SBT_RECORD_ALIGNMENT )
    dtype = np.dtype( { 
        'names'   : ['header' ],
        'formats' : formats, 
        'itemsize': itemsize,
        'align'   : True
        } )
    h_raygen_sbt = np.array( [ 0 ], dtype=dtype )
    optix.sbtRecordPackHeader( raygen_prog_group, h_raygen_sbt )
    global d_raygen_sbt 
    d_raygen_sbt = array_to_device_memory( h_raygen_sbt )
    
    #
    # miss record
    #
    formats  = [ header_format, 'f4', 'f4', 'f4']
    itemsize = get_aligned_itemsize( formats, optix.SBT_RECORD_ALIGNMENT )
    dtype = np.dtype( { 
        'names'   : ['header', 'r', 'g', 'b' ],
        'formats' : formats,
        'itemsize': itemsize,
        'align'   : True
        } )
    h_miss_sbt = np.array( [ (0, 0.3, 0.1, 0.2) ], dtype=dtype )
    optix.sbtRecordPackHeader( miss_prog_group, h_miss_sbt )
    global d_miss_sbt 
    d_miss_sbt = array_to_device_memory( h_miss_sbt )
    
    #
    # hitgroup record
    #
    formats  = [ header_format ]
    itemsize = get_aligned_itemsize( formats, optix.SBT_RECORD_ALIGNMENT )
    dtype = np.dtype( { 
        'names'   : ['header' ],
        'formats' : formats,
        'itemsize': itemsize,
        'align'   : True
        } )
    h_hitgroup_sbt = np.array( [ (0) ], dtype=dtype )
    optix.sbtRecordPackHeader( hitgroup_prog_group, h_hitgroup_sbt )
    global d_hitgroup_sbt
    d_hitgroup_sbt = array_to_device_memory( h_hitgroup_sbt )
    
    sbt = optix.ShaderBindingTable()
    sbt.raygenRecord                = d_raygen_sbt.ptr
    sbt.missRecordBase              = d_miss_sbt.ptr
    sbt.missRecordStrideInBytes     = d_miss_sbt.mem.size
    sbt.missRecordCount             = 1
    sbt.hitgroupRecordBase          = d_hitgroup_sbt.ptr
    sbt.hitgroupRecordStrideInBytes = d_hitgroup_sbt.mem.size
    sbt.hitgroupRecordCount         = 1
    return sbt


def launch( pipeline, sbt, trav_handle ):
    print( "Launching ... " )

    pix_bytes  = pix_width*pix_height*4
    
    h_pix = np.zeros( (pix_width,pix_height,4), 'B' )
    h_pix[0:pix_width, 0:pix_height] = [255, 128, 0, 255]
    d_pix = cp.array( h_pix )


    params = [
        ( 'u8', 'image',        d_pix.data.ptr ),
        ( 'u4', 'image_width',  pix_width      ),
        ( 'u4', 'image_height', pix_height     ),
        ( 'f4', 'cam_eye_x',    0              ),
        ( 'f4', 'cam_eye_y',    0              ),
        ( 'f4', 'cam_eye_z',    2.0            ),
        ( 'f4', 'cam_U_x',      1.10457        ),
        ( 'f4', 'cam_U_y',      0              ),
        ( 'f4', 'cam_U_z',      0              ),
        ( 'f4', 'cam_V_x',      0              ),
        ( 'f4', 'cam_V_y',      0.828427       ),
        ( 'f4', 'cam_V_z',      0              ),
        ( 'f4', 'cam_W_x',      0              ),
        ( 'f4', 'cam_W_y',      0              ),
        ( 'f4', 'cam_W_z',      -2.0           ),
        ( 'u8', 'trav_handle',  trav_handle    )
    ]
    
    formats = [ x[0] for x in params ] 
    names   = [ x[1] for x in params ] 
    values  = [ x[2] for x in params ] 
    itemsize = get_aligned_itemsize( formats, 8 )
    params_dtype = np.dtype( { 
        'names'   : names, 
        'formats' : formats,
        'itemsize': itemsize,
        'align'   : True
        } )
    h_params = np.array( [ tuple(values) ], dtype=params_dtype )
    d_params = array_to_device_memory( h_params )

    stream = cp.cuda.Stream()
    optix.launch( 
        pipeline, 
        stream.ptr, 
        d_params.ptr, 
        h_params.dtype.itemsize, 
        sbt,
        pix_width,
        pix_height,
        1 # depth
        )

    stream.synchronize()

    h_pix = cp.asnumpy( d_pix )
    return h_pix


# Numba compilation
# -----------------

# An equivalent to the compile_cuda function for Python kernels. The types of
# the arguments to the kernel must be provided, if there are any.

def compile_numba(f, sig=(), debug=False):
    # Based on numba.cuda.compile_ptx. We don't just use
    # compile_ptx_for_current_device because it generates a kernel with a
    # mangled name. For proceeding beyond this prototype, an option should be
    # added to compile_ptx in Numba to not mangle the function name.

    nvvm_options = {
        'debug': debug,
        'fastmath': False,
        'opt': 0 if debug else 3,
    }

    cres = numba_compile_cuda(f, None, sig, debug=debug,
                              nvvm_options=nvvm_options)
    fname = cres.fndesc.llvm_func_name
    tgt = cres.target_context
    filename = cres.type_annotation.filename
    linenum = int(cres.type_annotation.linenum)
    lib, kernel = tgt.prepare_cuda_kernel(cres.library, cres.fndesc, debug,
                                          nvvm_options, filename, linenum)
    cc = get_current_device().compute_capability
    ptx = lib.get_asm_str(cc=cc)

    # Demangle name
    mangled_name = kernel.name
    original_name = cres.library.name
    return ptx.replace(mangled_name, original_name)


#-------------------------------------------------------------------------------
#
# User code / kernel - the following section is what we'd expect a user of
# PyOptiX to write.
#
#-------------------------------------------------------------------------------

# vec_math

# Overload for Clamp
def clamp(x, a, b):
    pass

@overload(clamp, target="cuda")
def jit_clamp(x, a, b):
    if isinstance(x, types.Float):
        def clamp_float_impl(x, a, b):
            return max(a, min(x, b))
        return clamp_float_impl
    elif isinstance(x, Float3):
        def clamp_float3_impl(x, a, b):
            return make_float3(clamp(x.x, a, b), clamp(x.y, a, b), clamp(x.z, a, b))
        return clamp_float3_impl


# def dot(a, b):
#     pass

# @overload(dot, target="cuda")
# def jit_dot(a, b):
#     if isinstance(a, Float3) and isinstance(b, Float3):
#         def dot_float3_impl(a, b):
#             return a.x * b.x + a.y * b.y + a.z * b.z
#         return dot_float3_impl

@cuda.jit(device=True)
def dot(a, b):
    return a.x * b.x + a.y * b.y + a.z * b.z


@cuda.jit(device=True)
def normalize(v):
    invLen = float32(1.0) / math.sqrt(dot(v, v))
    return v * invLen


# Helpers

@cuda.jit(device=True)
def toSRGB(c):
    # Use float32 for constants
    invGamma = float32(1.0) / float32(2.4)
    powed = make_float3(math.pow(c.x, invGamma), math.pow(c.x, invGamma), math.pow(c.x, invGamma))
    return make_float3(
        float32(12.92) * c.x if c.x < float32(0.0031308) else float32(1.055) * powed.x - float32(0.055),
        float32(12.92) * c.y if c.y < float32(0.0031308) else float32(1.055) * powed.y - float32(0.055),
        float32(12.92) * c.z if c.z < float32(0.0031308) else float32(1.055) * powed.z - float32(0.055))


@cuda.jit(device=True)
def quantizeUnsigned8Bits(x):
    x = clamp( x, float32(0.0), float32(1.0) )
    N, Np1 = 1 << 8 - 1, 1 << 8
    return uint8(min(uint8(x * float32(Np1)), uint8(N)))

@cuda.jit(device=True)
def make_color(c):
    srgb = toSRGB(clamp(c, float32(0.0), float32(1.0)))
    return make_uchar4(quantizeUnsigned8Bits(srgb.x), quantizeUnsigned8Bits(srgb.y), quantizeUnsigned8Bits(srgb.z), uint8(255))

# ray functions

@cuda.jit(device=True)
def setPayload(p):
    optix.SetPayload_0(float_as_int(p.x))
    optix.SetPayload_1(float_as_int(p.y))
    optix.SetPayload_2(float_as_int(p.z))

@cuda.jit(device=True)
def computeRay(idx, dim, origin, direction):
    U = params.cam_u
    V = params.cam_v
    W = params.cam_w
    d = types.float32(2.0) * make_float2(
            types.float32(idx.x) / types.float32(dim.x),
            types.float32(idx.y) / types.float32(dim.y)
        ) - types.float32(1.0)

    origin = params.cam_eye
    direction = normalize(d.x * U + d.y * V + W)


def __raygen__rg():
    # Lookup our location within the launch grid
    idx = optix.GetLaunchIndex()
    dim = optix.GetLaunchDimensions()

    # Map our launch idx to a screen location and create a ray from the camera
    # location through the screen
    ray_origin = make_float3(float32(0.0), float32(0.0), float32(0.0))
    ray_direction = make_float3(float32(0.0), float32(0.0), float32(0.0))
    computeRay(make_uint3(idx.x, idx.y, 0), dim, ray_origin, ray_direction)

    # Trace the ray against our scene hierarchy
    p0 = cuda.local.array(1, types.int32)
    p1 = cuda.local.array(1, types.int32)
    p2 = cuda.local.array(1, types.int32)
    optix.Trace(
            params.handle,
            ray_origin,
            ray_direction,
            types.float32(0.0),         # Min intersection distance
            types.float32(1e16),        # Max intersection distance
            types.float32(0.0),         # rayTime -- used for motion blur
            OptixVisibilityMask(255),   # Specify always visible
            # OptixRayFlags.OPTIX_RAY_FLAG_NONE,
            uint32(OPTIX_RAY_FLAG_NONE),
            uint32(0),                          # SBT offset   -- See SBT discussion
            uint32(1),                          # SBT stride   -- See SBT discussion
            uint32(0),                          # missSBTIndex -- See SBT discussion
            p0[0], p1[0], p2[0])
    result = make_float3(p0[0], p1[0], p2[0])

    # Record results in our output raster
    params.image[idx.y * params.image_width + idx.x] = make_color( result )


@cuda.jit
def __miss__ms():
    miss_data  = MissData(optix.GetSbtDataPointer())
    setPayload(miss_data.bg_color)


@cuda.jit
def __closesthit__ch():
    # When built-in triangle intersection is used, a number of fundamental
    # attributes are provided by the OptiX API, indlucing barycentric coordinates.
    barycentrics = optix.GetTriangleBarycentrics()

    setPayload(make_float3(barycentrics, float32(1.0)))


#-------------------------------------------------------------------------------
#
# main
#
#-------------------------------------------------------------------------------


def main():
    triangle_ptx = compile_numba(__raygen__rg)
    # triangle_ptx = compile_cuda( "examples/triangle.cu" )
    # print(triangle_ptx)

    init_optix()

    ctx              = create_ctx()
    gas_handle, d_gas_output_buffer = create_accel(ctx)
    pipeline_options = set_pipeline_options()
    module           = create_module( ctx, pipeline_options, triangle_ptx )
    prog_groups      = create_program_groups( ctx, module )
    pipeline         = create_pipeline( ctx, prog_groups, pipeline_options )
    sbt              = create_sbt( prog_groups ) 
    pix              = launch( pipeline, sbt, gas_handle ) 

    print( "Total number of log messages: {}".format( logger.num_mssgs ) )

    pix = pix.reshape( ( pix_height, pix_width, 4 ) )     # PIL expects [ y, x ] resolution
    img = ImageOps.flip( Image.fromarray( pix, 'RGBA' ) ) # PIL expects y = 0 at bottom
    img.save( 'my.png' )
    img.show()


if __name__ == "__main__":
    main()
