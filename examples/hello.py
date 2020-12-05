


import optix
import cupy  as cp
import numpy as np
import ctypes
from PIL import Image


#-------------------------------------------------------------------------------
#
# Util 
#
#-------------------------------------------------------------------------------

class Logger:
    def __init__( self ):
        self.num_mssgs = 0

    def __call__( self, level, tag, mssg ):
        print( "[{:>2}][{:>12}]: {}".format( level, tag, mssg ) )
        self.num_mssgs += 1


def log_callback( level, tag, mssg ):
    print( "[{:>2}][{:>12}]: {}".format( level, tag, mssg ) )
    

def arrayToDeviceMemory( numpy_array, stream=cp.cuda.Stream() ):

    byte_size = numpy_array.size*numpy_array.dtype.itemsize

    h_ptr = ctypes.c_void_p( numpy_array.ctypes.data )
    d_mem = cp.cuda.memory.alloc( byte_size )
    d_mem.copy_from_async( h_ptr, byte_size, stream )
    return d_mem

#-------------------------------------------------------------------------------
#
# Optix setup
#
#-------------------------------------------------------------------------------


def init_optix():
    print( "Initializing cuda ..." )
    cp.cuda.runtime.free( 0 )

    print( "Initializing optix ..." )
    optix.init()


def create_ctx():
    print( "Creating optix device context ..." )
    cu_ctx      = optix.cuda.Context() # TODO: get rid of optix.cuda.Context
    ctx_options = optix.DeviceContextOptions()

    # Note that log callback data is no longer needed
    global logger
    logger = Logger()
    ctx_options.logCallbackLevel    = 4
    ctx_options.logCallbackFunction = logger
    #ctx_options.logCallbackFunction = log_callback 

    return optix.deviceContextCreate( cu_ctx, ctx_options )


def create_module( ctx, pipeline_options ):
    print( "Creating optix module ..." )

    module_options = optix.ModuleCompileOptions()
    # TODO: need to wrap #defines (eg, optix.COMPILE_DEFAULT_MAX_REGISTER_COUNT)
    module_options.maxRegisterCount = 0 
    module_options.optLevel         = optix.COMPILE_OPTIMIZATION_DEFAULT
    module_options.debugLevel       = optix.COMPILE_DEBUG_LEVEL_LINEINFO

    log = ""
    module = ctx.moduleCreateFromPTX(
            module_options,
            pipeline_options,
            draw_color_ptx,
            log
            )
    print( "\tModule create log: <<<{}>>>".format( log ) )
    return module


def create_program_groups( ctx ):
    print( "Creating program groups ... " )
    # TODO: optix.ProgramGroup.Options() ?
    program_group_options = optix.ProgramGroupOptions();

    # TODO: optix.ProgramGroup.Kind.RAYGEN ?
    raygen_prog_group_desc  = optix.ProgramGroupDesc()
    raygen_prog_group_desc.kind                     = optix.PROGRAM_GROUP_KIND_RAYGEN; 
    raygen_prog_group_desc.raygenModule             = module;
    raygen_prog_group_desc.raygenEntryFunctionName  = "__raygen__hello";

    log = ""
    raygen_prog_group = ctx.programGroupCreate(
            [ raygen_prog_group_desc ], 
            program_group_options,
            log
            )

    # Leave miss group's module and entryfunc name null
    miss_prog_group_desc  = optix.ProgramGroupDesc()
    miss_prog_group_desc.kind = optix.PROGRAM_GROUP_KIND_MISS;
    miss_prog_group = ctx.programGroupCreate(
            [ miss_prog_group_desc ],
            program_group_options,
            log
            )

    return ( raygen_prog_group[0], miss_prog_group[0] )


def create_pipeline( ctx, raygen_prog_group, pipeline_compile_options ):
    print( "Creating pipeline ... " )
    max_trace_depth  = 0;
    program_groups = [ raygen_prog_group ]

    pipeline_link_options = optix.PipelineLinkOptions() 
    pipeline_link_options.maxTraceDepth = max_trace_depth;
    pipeline_link_options.debugLevel    = optix.COMPILE_DEBUG_LEVEL_FULL;

    log = ""
    pipeline = ctx.pipelineCreate(
            pipeline_compile_options,
            pipeline_link_options,
            program_groups,
            log
            )

    stack_sizes = optix.StackSizes()
    for prog_group in program_groups:
        optix.util.accumulateStackSizes( prog_group, stack_sizes )

    (dc_stack_size_from_trav, dc_stack_size_from_state, cc_stack_size) = \
        optix.util.computeStackSizes( 
            stack_sizes, 
            0,  # maxTraceDepth
            0,  # maxCCDepth
            0   # maxDCDepth
            )
    
    pipeline.setStackSize( 
            dc_stack_size_from_trav,
            dc_stack_size_from_state, 
            cc_stack_size,
            2  # maxTraversableDepth
            )

    return pipeline


def create_sbt( raygen_prog_group, miss_prog_group ):
    print( "Creating sbt ... " )

    global d_raygen_sbt
    global d_miss_sbt

    #
    # raygen record
    #
    dtype = np.dtype( { 
        'names'   : ['header', 'r', 'g', 'b' ],
        'formats' : ['32B', 'f4', 'f4', 'f4'],
        'itemsize': 48,
        'align'   : True
        } )
    h_raygen_sbt = np.array( [ (0, 0.462, 0.725, 0.0 ) ], dtype=dtype )
    optix.sbtRecordPackHeader( raygen_prog_group, h_raygen_sbt )
    d_raygen_sbt = arrayToDeviceMemory( h_raygen_sbt )
    
    #
    # miss record
    #
    dtype = np.dtype( { 
        'names'   : ['header', 'x' ],
        'formats' : ['32B', 'i4'],
        'itemsize': 48,
        'align'   : True
        } )
    h_miss_sbt = np.array( [ (0, 127 ) ], dtype=dtype )
    optix.sbtRecordPackHeader( miss_prog_group, h_miss_sbt )
    d_miss_sbt = arrayToDeviceMemory( h_miss_sbt )
    
    sbt = optix.ShaderBindingTable();
    sbt.raygenRecord                = d_raygen_sbt.ptr
    sbt.missRecordBase              = d_miss_sbt.ptr
    sbt.missRecordStrideInBytes     = d_miss_sbt.mem.size
    sbt.missRecordCount             = 1
    return sbt


def launch( pipeline, sbt ):
    print( "Launching ... " )

    pix_width  = 512
    pix_height = 512
    pix_bytes  = pix_width*pix_height*4
    
    h_pix = np.zeros( (pix_width,pix_height,4), 'B' )
    h_pix[0:256, 0:256] = [255, 128, 0, 255]
    d_pix = cp.array( h_pix )

    params_dtype = np.dtype( { 
        'names'   : ['image', 'image_width' ],
        'formats' : ['u8', 'u4'],
        'align'   : True
        } )
    h_params = np.array( [ ( d_pix.data.ptr, pix_width ) ], dtype=params_dtype )
    d_params = arrayToDeviceMemory( h_params )

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


#-------------------------------------------------------------------------------
#
# main
#
#-------------------------------------------------------------------------------


draw_color_ptx = ''
with open( "examples/hello.ptx" ) as ptx_file:
    draw_color_ptx = ptx_file.read()

init_optix()
ctx = create_ctx()

pipeline_options = optix.PipelineCompileOptions()
pipeline_options.usesMotionBlur        = False;
pipeline_options.traversableGraphFlags = optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
pipeline_options.numPayloadValues      = 2;
pipeline_options.numAttributeValues    = 2;
pipeline_options.exceptionFlags        = optix.EXCEPTION_FLAG_NONE;
pipeline_options.pipelineLaunchParamsVariableName = "params";

module   = create_module( ctx, pipeline_options )
raygen_prog_group, miss_prog_group = create_program_groups( ctx )
pipeline = create_pipeline( ctx, raygen_prog_group, pipeline_options )
sbt      = create_sbt( raygen_prog_group, miss_prog_group ) 
pix      = launch( pipeline, sbt ) 

print( "Total number of log messages: {}".format( logger.num_mssgs ) )

img = Image.fromarray(pix, 'RGBA')
img.save('my.png')
img.show()


