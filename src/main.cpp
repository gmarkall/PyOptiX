
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <cuda_runtime.h>


namespace py = pybind11;
    
#define PYOPTIX_CHECK( call )                                                  \
    do                                                                         \
    {                                                                          \
        OptixResult res = call;                                                \
        if( res != OPTIX_SUCCESS )                                             \
            throw std::runtime_error( optixGetErrorString( res )  );           \
    } while( 0 )


namespace pyoptix
{

//
// Helpers
//

constexpr size_t LOG_BUFFER_MAX_SIZE = 2048u;

void context_log_cb( unsigned int level, const char* tag, const char* message, void* cbdata  )
{
    py::object cb = py::reinterpret_borrow<py::object>( reinterpret_cast<PyObject*>( cbdata ) );
    cb( level, tag, message );
}


//
// Proxy objets to modify some functionality in the optix param structs
//

struct DeviceContextOptions
{
    // This proxy object exists to pass along a py::object function for the log
    // callback so that it is correctly reference counted
    py::object logCallbackFunction;
    OptixDeviceContextOptions options;
};


struct PipelineCompileOptions
{
    // all char* need to be backed by strings
    std::string pipelineLaunchParamsVariableName;
    OptixPipelineCompileOptions options;
};


struct ProgramGroupDesc
{
    std::string entryFunctionName0;
    std::string entryFunctionName1;
    std::string entryFunctionName2;
    OptixProgramGroupDesc program_group_desc;
};


//
// Opaque type struct wrappers
//

struct DeviceContext
{
    OptixDeviceContext deviceContext = 0;
    py::object         logCallbackFunction;
};

struct Module
{
    OptixModule module = 0;
};

struct ProgramGroup
{
    OptixProgramGroup programGroup = 0;
};

struct Pipeline
{
    OptixPipeline pipeline = 0;
};

struct Denoiser
{
    OptixDenoiser denoiser = 0;
};


//------------------------------------------------------------------------------
//
// OptiX API error checked wrappers
//
//------------------------------------------------------------------------------

void init()
{
    PYOPTIX_CHECK( optixInit() );
}
 

const char* getErrorName( 
       OptixResult result
    )
{
    return optixGetErrorName( result );
}
 
const char* getErrorString( 
       OptixResult result
    )
{
    return optixGetErrorString( result );
}
 
pyoptix::DeviceContext deviceContextCreate( 
       uintptr_t fromContext,
       const pyoptix::DeviceContextOptions& options
    )
{
    pyoptix::DeviceContext ctx{};
    ctx.logCallbackFunction = options.logCallbackFunction;

    PYOPTIX_CHECK( 
        optixDeviceContextCreate(
            reinterpret_cast<CUcontext>( fromContext ),
            &options.options,
            &(ctx.deviceContext)
        )
    );
    return ctx;
}
 
void deviceContextDestroy( 
       pyoptix::DeviceContext context
    )
{
    PYOPTIX_CHECK( 
        optixDeviceContextDestroy(
            context.deviceContext
        )
    );
}
 
void deviceContextGetProperty( 
       pyoptix::DeviceContext context,
       OptixDeviceProperty property,
       void* value,
       size_t sizeInBytes
    )
{
    PYOPTIX_CHECK( 
        optixDeviceContextGetProperty(
            context.deviceContext,
            property,
            value,
            sizeInBytes
        )
    );
}
 
void deviceContextSetLogCallback( 
       pyoptix::DeviceContext context,
       OptixLogCallback   callbackFunction,
       void*              callbackData,
       unsigned int       callbackLevel
    )
{
    PYOPTIX_CHECK( 
        optixDeviceContextSetLogCallback(
            context.deviceContext,
            callbackFunction,
            callbackData,
            callbackLevel
        )
    );
}
 
void deviceContextSetCacheEnabled( 
       pyoptix::DeviceContext context,
       int                enabled
    )
{
    PYOPTIX_CHECK( 
        optixDeviceContextSetCacheEnabled(
            context.deviceContext,
            enabled
        )
    );
}
 
void deviceContextSetCacheLocation( 
       pyoptix::DeviceContext context,
       const char* location
    )
{
    PYOPTIX_CHECK( 
        optixDeviceContextSetCacheLocation(
            context.deviceContext,
            location
        )
    );
}
 
void deviceContextSetCacheDatabaseSizes( 
       pyoptix::DeviceContext context,
       size_t lowWaterMark,
       size_t highWaterMark
    )
{
    PYOPTIX_CHECK( 
        optixDeviceContextSetCacheDatabaseSizes(
            context.deviceContext,
            lowWaterMark,
            highWaterMark
        )
    );
}
 
void deviceContextGetCacheEnabled( 
       pyoptix::DeviceContext context,
       int* enabled
    )
{
    PYOPTIX_CHECK( 
        optixDeviceContextGetCacheEnabled(
            context.deviceContext,
            enabled
        )
    );
}
 
void deviceContextGetCacheLocation( 
       pyoptix::DeviceContext context,
       char* location,
       size_t locationSize
    )
{
    PYOPTIX_CHECK( 
        optixDeviceContextGetCacheLocation(
            context.deviceContext,
            location,
            locationSize
        )
    );
}
 
void deviceContextGetCacheDatabaseSizes( 
       pyoptix::DeviceContext context,
       size_t* lowWaterMark,
       size_t* highWaterMark
    )
{
    PYOPTIX_CHECK( 
        optixDeviceContextGetCacheDatabaseSizes(
            context.deviceContext,
            lowWaterMark,
            highWaterMark
        )
    );
}
 

// TODO: get tid of numProgramGroups
pyoptix::Pipeline pipelineCreate( 
       pyoptix::DeviceContext                 context,
       const pyoptix::PipelineCompileOptions& pipelineCompileOptions,
       const OptixPipelineLinkOptions&    pipelineLinkOptions,
       const py::list&                        programGroups,
       std::string&                           logString
    )
{
    std::vector<OptixProgramGroup> pgs;
    for( const auto list_elem : programGroups )
    {
        pyoptix::ProgramGroup pygroup = list_elem.cast<pyoptix::ProgramGroup>();
        pgs.push_back( pygroup.programGroup );
    }
    
    size_t log_buf_size = LOG_BUFFER_MAX_SIZE;
    char   log_buf[ LOG_BUFFER_MAX_SIZE ];
    log_buf[0] = '\0';

    pyoptix::Pipeline pipeline{};
    PYOPTIX_CHECK( 
        optixPipelineCreate(
            context.deviceContext,
            &pipelineCompileOptions.options,
            &pipelineLinkOptions,
            pgs.data(),
            static_cast<uint32_t>( pgs.size() ),
            log_buf,
            &log_buf_size,
            &pipeline.pipeline
        )
    );

    logString = log_buf;
    return pipeline;
}
 
void pipelineDestroy( 
       pyoptix::Pipeline pipeline
    )
{
    PYOPTIX_CHECK( 
        optixPipelineDestroy(
            pipeline.pipeline
        )
    );
}
 
void pipelineSetStackSize( 
       pyoptix::Pipeline pipeline,
       unsigned int  directCallableStackSizeFromTraversal,
       unsigned int  directCallableStackSizeFromState,
       unsigned int  continuationStackSize,
       unsigned int  maxTraversableGraphDepth
    )
{
    PYOPTIX_CHECK( 
        optixPipelineSetStackSize(
            pipeline.pipeline,
            directCallableStackSizeFromTraversal,
            directCallableStackSizeFromState,
            continuationStackSize,
            maxTraversableGraphDepth
        )
    );
}
 
pyoptix::Module moduleCreateFromPTX( 
       pyoptix::DeviceContext            context,
       OptixModuleCompileOptions*        moduleCompileOptions,
       pyoptix::PipelineCompileOptions*  pipelineCompileOptions,
       const std::string&                PTX,
       std::string&                      logString
       )
{
    size_t log_buf_size = LOG_BUFFER_MAX_SIZE;
    char   log_buf[ LOG_BUFFER_MAX_SIZE ];
    log_buf[0] = '\0';

    pyoptix::Module module;
    pipelineCompileOptions->options.pipelineLaunchParamsVariableName = 
        pipelineCompileOptions->pipelineLaunchParamsVariableName.c_str();

    PYOPTIX_CHECK( 
        optixModuleCreateFromPTX(
            context.deviceContext,
            moduleCompileOptions,
            &pipelineCompileOptions->options,
            PTX.c_str(),
            static_cast<size_t>( PTX.size()+1 ),
            log_buf,
            &log_buf_size,
            &module.module
        )
    );
    logString = log_buf;
    return module;
}
 
void moduleDestroy( 
       pyoptix::Module module
    )
{
    PYOPTIX_CHECK( 
        optixModuleDestroy(
            module.module
        )
    );
}
 
pyoptix::Module builtinISModuleGet( 
       pyoptix::DeviceContext            context,
       OptixModuleCompileOptions*        moduleCompileOptions,
       pyoptix::PipelineCompileOptions*  pipelineCompileOptions,
       const OptixBuiltinISOptions*      builtinISOptions
    )
{
    pyoptix::Module module;
    PYOPTIX_CHECK( 
        optixBuiltinISModuleGet(
            context.deviceContext,
            moduleCompileOptions,
            &pipelineCompileOptions->options,
            builtinISOptions,
            &module.module
        )
    );
    return module;
}

void programGroupGetStackSize( 
       pyoptix::ProgramGroup programGroup,
       OptixStackSizes* stackSizes
    )
{
    PYOPTIX_CHECK( 
        optixProgramGroupGetStackSize(
            programGroup.programGroup,
            stackSizes
        )
    );
}
 
py::list programGroupCreate( 
       pyoptix::DeviceContext          context,
       const py::list&                 programDescriptions,
       const OptixProgramGroupOptions& options,
       std::string&                    logString
    )
{
    size_t log_buf_size = LOG_BUFFER_MAX_SIZE;
    char   log_buf[ LOG_BUFFER_MAX_SIZE ];
    log_buf[0] = '\0';

    std::vector<OptixProgramGroupDesc> program_groups_descs;
    for( auto list_elem : programDescriptions )
    {
        pyoptix::ProgramGroupDesc& pydesc = 
		list_elem.cast<pyoptix::ProgramGroupDesc&>();
        switch( pydesc.program_group_desc.kind )
        {
            case OPTIX_PROGRAM_GROUP_KIND_RAYGEN:
            case OPTIX_PROGRAM_GROUP_KIND_MISS:
            case OPTIX_PROGRAM_GROUP_KIND_EXCEPTION:
                pydesc.program_group_desc.raygen.entryFunctionName = 
                    !pydesc.entryFunctionName0.empty() ? 
                    pydesc.entryFunctionName0.c_str() : 
                    nullptr;
            case OPTIX_PROGRAM_GROUP_KIND_HITGROUP:
                pydesc.program_group_desc.hitgroup.entryFunctionNameCH = 
                    !pydesc.entryFunctionName0.empty() ? 
                    pydesc.entryFunctionName0.c_str() :
                    nullptr;
                pydesc.program_group_desc.hitgroup.entryFunctionNameAH = 
                    !pydesc.entryFunctionName1.empty() ? 
                    pydesc.entryFunctionName1.c_str() :
                    nullptr;
                pydesc.program_group_desc.hitgroup.entryFunctionNameIS = 
                    !pydesc.entryFunctionName2.empty() ? 
                    pydesc.entryFunctionName2.c_str() :
                    nullptr;
                break;
            case OPTIX_PROGRAM_GROUP_KIND_CALLABLES:
                pydesc.program_group_desc.callables.entryFunctionNameDC = 
                    !pydesc.entryFunctionName0.empty() ? 
                    pydesc.entryFunctionName0.c_str() :
                    nullptr;
                pydesc.program_group_desc.callables.entryFunctionNameCC = 
                    !pydesc.entryFunctionName1.empty() ? 
                    pydesc.entryFunctionName1.c_str() :
                    nullptr;
                break;

        }
        program_groups_descs.push_back( pydesc.program_group_desc );
    }
    std::vector<OptixProgramGroup> program_groups( programDescriptions.size() );

    PYOPTIX_CHECK( 
        optixProgramGroupCreate(
            context.deviceContext,
            program_groups_descs.data(),
            static_cast<uint32_t>( program_groups_descs.size() ),
            &options,
            log_buf,
            &log_buf_size,
            program_groups.data()
        )
    );
    logString = log_buf;
    
    py::list pygroups;
    for( auto& group : program_groups )
        pygroups.append( pyoptix::ProgramGroup{ group } );

    return pygroups;
}
 
void programGroupDestroy( 
       pyoptix::ProgramGroup programGroup
    )
{
    PYOPTIX_CHECK( 
        optixProgramGroupDestroy(
            programGroup.programGroup
        )
    );
}
 
void launch( 
       pyoptix::Pipeline              pipeline,
       uintptr_t                      stream,
       CUdeviceptr                    pipelineParams,
       size_t                         pipelineParamsSize,
       const OptixShaderBindingTable& sbt,
       uint32_t                       width,
       uint32_t                       height,
       uint32_t                       depth
    )
{
    char buf[128];
    cudaError res = cudaMemcpy( 
        buf, 
	(void*)sbt.raygenRecord, 
	48, 
	cudaMemcpyDeviceToHost 
	);
    res = cudaMemcpy( buf, (void*)sbt.missRecordBase, 48, cudaMemcpyDeviceToHost );
    PYOPTIX_CHECK( 
        optixLaunch(
            pipeline.pipeline,
            reinterpret_cast<CUstream>( stream ),
            pipelineParams,
            pipelineParamsSize,
            &sbt,
            width,
            height,
            depth
        )
    );
}
 
void sbtRecordPackHeader( 
       pyoptix::ProgramGroup programGroup,
       py::buffer sbtRecord
    )
{
    py::buffer_info binfo = sbtRecord.request();
    // TODO: sanity check buffer

    PYOPTIX_CHECK( 
        optixSbtRecordPackHeader(
            programGroup.programGroup,
            binfo.ptr 
        )
    );
}
 
void accelComputeMemoryUsage( 
       pyoptix::DeviceContext            context,
       const OptixAccelBuildOptions* accelOptions,
       const OptixBuildInput*        buildInputs,
       unsigned int                  numBuildInputs,
       OptixAccelBufferSizes*        bufferSizes
    )
{
    PYOPTIX_CHECK( 
        optixAccelComputeMemoryUsage(
            context.deviceContext,
            accelOptions,
            buildInputs,
            numBuildInputs,
            bufferSizes
        )
    );
}
 
void accelBuild( 
       pyoptix::DeviceContext        context,
       uintptr_t         stream,
       const OptixAccelBuildOptions* accelOptions,
       const OptixBuildInput*        buildInputs,
       unsigned int                  numBuildInputs,
       CUdeviceptr                   tempBuffer,
       size_t                        tempBufferSizeInBytes,
       CUdeviceptr                   outputBuffer,
       size_t                        outputBufferSizeInBytes,
       OptixTraversableHandle*       outputHandle,
       const OptixAccelEmitDesc*     emittedProperties,
       unsigned int                  numEmittedProperties
    )
{
    PYOPTIX_CHECK( 
        optixAccelBuild(
            context.deviceContext,
            reinterpret_cast<CUstream>( stream ),
            accelOptions,
            buildInputs,
            numBuildInputs,
            tempBuffer,
            tempBufferSizeInBytes,
            outputBuffer,
            outputBufferSizeInBytes,
            outputHandle,
            emittedProperties,
            numEmittedProperties
        )
    );
}
 
void accelGetRelocationInfo( 
       pyoptix::DeviceContext context,
       OptixTraversableHandle handle,
       OptixAccelRelocationInfo* info
    )
{
    PYOPTIX_CHECK( 
        optixAccelGetRelocationInfo(
            context.deviceContext,
            handle,
            info
        )
    );
}
 
void accelCheckRelocationCompatibility( 
       pyoptix::DeviceContext context,
       const OptixAccelRelocationInfo* info,
       int* compatible
    )
{
    PYOPTIX_CHECK( 
        optixAccelCheckRelocationCompatibility(
            context.deviceContext,
            info,
            compatible
        )
    );
}
 
void accelRelocate( 
       pyoptix::DeviceContext          context,
       uintptr_t           stream,
       const OptixAccelRelocationInfo* info,
       CUdeviceptr                     instanceTraversableHandles,
       size_t                          numInstanceTraversableHandles,
       CUdeviceptr                     targetAccel,
       size_t                          targetAccelSizeInBytes,
       OptixTraversableHandle*         targetHandle
    )
{
    PYOPTIX_CHECK( 
        optixAccelRelocate(
            context.deviceContext,
            reinterpret_cast<CUstream>( stream ),
            info,
            instanceTraversableHandles,
            numInstanceTraversableHandles,
            targetAccel,
            targetAccelSizeInBytes,
            targetHandle
        )
    );
}
 
void accelCompact( 
       pyoptix::DeviceContext  context,
       uintptr_t   stream,
       OptixTraversableHandle  inputHandle,
       CUdeviceptr             outputBuffer,
       size_t                  outputBufferSizeInBytes,
       OptixTraversableHandle* outputHandle
    )
{
    PYOPTIX_CHECK( 
        optixAccelCompact(
            context.deviceContext,
            reinterpret_cast<CUstream>( stream ),
            inputHandle,
            outputBuffer,
            outputBufferSizeInBytes,
            outputHandle
        )
    );
}
 
void convertPointerToTraversableHandle( 
       pyoptix::DeviceContext  onDevice,
       CUdeviceptr             pointer,
       OptixTraversableType    traversableType,
       OptixTraversableHandle* traversableHandle
    )
{
    PYOPTIX_CHECK( 
        optixConvertPointerToTraversableHandle(
            onDevice.deviceContext,
            pointer,
            traversableType,
            traversableHandle
        )
    );
}
 
void denoiserCreate( 
       pyoptix::DeviceContext context,
       const OptixDenoiserOptions* options,
       pyoptix::Denoiser* denoiser
    )
{
    PYOPTIX_CHECK( 
        optixDenoiserCreate(
            context.deviceContext,
            options,
            &denoiser->denoiser
        )
    );
}
 
void denoiserSetModel( 
       pyoptix::Denoiser denoiser,
       OptixDenoiserModelKind kind,
       void* data,
       size_t sizeInBytes
    )
{
    PYOPTIX_CHECK( 
        optixDenoiserSetModel(
            denoiser.denoiser,
            kind,
            data,
            sizeInBytes
        )
    );
}
 
void denoiserDestroy( 
       pyoptix::Denoiser denoiser
    )
{
    PYOPTIX_CHECK( 
        optixDenoiserDestroy(
            denoiser.denoiser
        )
    );
}
 
void denoiserComputeMemoryResources( 
       const pyoptix::Denoiser denoiser,
       unsigned int        outputWidth,
       unsigned int        outputHeight,
       OptixDenoiserSizes* returnSizes
    )
{
    PYOPTIX_CHECK( 
        optixDenoiserComputeMemoryResources(
            denoiser.denoiser,
            outputWidth,
            outputHeight,
            returnSizes
        )
    );
}
 
void denoiserSetup( 
       pyoptix::Denoiser denoiser,
       uintptr_t stream,
       unsigned int  inputWidth,
       unsigned int  inputHeight,
       CUdeviceptr   denoiserState,
       size_t        denoiserStateSizeInBytes,
       CUdeviceptr   scratch,
       size_t        scratchSizeInBytes
    )
{
    PYOPTIX_CHECK( 
        optixDenoiserSetup(
            denoiser.denoiser,
            reinterpret_cast<CUstream>( stream ),
            inputWidth,
            inputHeight,
            denoiserState,
            denoiserStateSizeInBytes,
            scratch,
            scratchSizeInBytes
        )
    );
}
 
void denoiserInvoke( 
       pyoptix::Denoiser          denoiser,
       uintptr_t      stream,
       const OptixDenoiserParams* params,
       CUdeviceptr                denoiserState,
       size_t                     denoiserStateSizeInBytes,
       const OptixImage2D*        inputLayers,
       unsigned int               numInputLayers,
       unsigned int               inputOffsetX,
       unsigned int               inputOffsetY,
       const OptixImage2D*        outputLayer,
       CUdeviceptr                scratch,
       size_t                     scratchSizeInBytes
    )
{
    PYOPTIX_CHECK( 
        optixDenoiserInvoke(
            denoiser.denoiser,
            reinterpret_cast<CUstream>( stream ),
            params,
            denoiserState,
            denoiserStateSizeInBytes,
            inputLayers,
            numInputLayers,
            inputOffsetX,
            inputOffsetY,
            outputLayer,
            scratch,
            scratchSizeInBytes
        )
    );
}
 
void denoiserComputeIntensity( 
       pyoptix::Denoiser   denoiser,
       uintptr_t stream,
       const OptixImage2D* inputImage,
       CUdeviceptr         outputIntensity,
       CUdeviceptr         scratch,
       size_t              scratchSizeInBytes
    )
{
    PYOPTIX_CHECK( 
        optixDenoiserComputeIntensity(
            denoiser.denoiser,
            reinterpret_cast<CUstream>( stream ),
            inputImage,
            outputIntensity,
            scratch,
            scratchSizeInBytes
        )
    );
}
 
void denoiserComputeAverageColor( 
       pyoptix::Denoiser   denoiser,
       uintptr_t stream,
       const OptixImage2D* inputImage,
       CUdeviceptr         outputAverageColor,
       CUdeviceptr         scratch,
       size_t              scratchSizeInBytes
    )
{
    PYOPTIX_CHECK( 
        optixDenoiserComputeAverageColor(
            denoiser.denoiser,
            reinterpret_cast<CUstream>( stream ),
            inputImage,
            outputAverageColor,
            scratch,
            scratchSizeInBytes
        )
    );
}
    

namespace util
{

void accumulateStackSizes( 
        pyoptix::ProgramGroup programGroup,
        OptixStackSizes&  stackSizes 
        )
{
    PYOPTIX_CHECK( 
        optixUtilAccumulateStackSizes( programGroup.programGroup, &stackSizes )
    );
}

py::tuple computeStackSizes( 
        const OptixStackSizes& stackSizes,
        unsigned int           maxTraceDepth,
        unsigned int           maxCCDepth,
        unsigned int           maxDCDepth
        )
{
    uint32_t directCallableStackSizeFromTraversal;
    uint32_t directCallableStackSizeFromState;
    uint32_t continuationStackSize;

    PYOPTIX_CHECK( 
        optixUtilComputeStackSizes(
            &stackSizes,
            maxTraceDepth,
            maxCCDepth,
            maxDCDepth,
            &directCallableStackSizeFromTraversal,
            &directCallableStackSizeFromState,
            &continuationStackSize 
            )
        );
    return py::make_tuple( 
        directCallableStackSizeFromTraversal,
        directCallableStackSizeFromState,
        continuationStackSize );
}

void computeStackSizesDCSplit( 
        const OptixStackSizes* stackSizes,
        unsigned int           dssDCFromTraversal,
        unsigned int           dssDCFromState,
        unsigned int           maxTraceDepth,
        unsigned int           maxCCDepth,
        unsigned int           maxDCDepthFromTraversal,
        unsigned int           maxDCDepthFromState,
        unsigned int*          directCallableStackSizeFromTraversal,
        unsigned int*          directCallableStackSizeFromState,
        unsigned int*          continuationStackSize 
        )
{
    PYOPTIX_CHECK( 
        optixUtilComputeStackSizesDCSplit( 
            stackSizes,
            dssDCFromTraversal,
            dssDCFromState,
            maxTraceDepth,
            maxCCDepth,
            maxDCDepthFromTraversal,
            maxDCDepthFromState,
            directCallableStackSizeFromTraversal,
            directCallableStackSizeFromState,
            continuationStackSize 
            )
        );
}


void computeStackSizesCssCCTree( 
        const OptixStackSizes* stackSizes,
        unsigned int           cssCCTree,
        unsigned int           maxTraceDepth,
        unsigned int           maxDCDepth,
        unsigned int*          directCallableStackSizeFromTraversal,
        unsigned int*          directCallableStackSizeFromState,
        unsigned int*          continuationStackSize 
        )
{
    PYOPTIX_CHECK(
        optixUtilComputeStackSizesCssCCTree( 
            stackSizes,
            cssCCTree,
            maxTraceDepth,
            maxDCDepth,
            directCallableStackSizeFromTraversal,
            directCallableStackSizeFromState,
            continuationStackSize 
            )
        );
}


void computeStackSizesSimplePathTracer(
        pyoptix::ProgramGroup        programGroupRG,
        pyoptix::ProgramGroup        programGroupMS1,
        const pyoptix::ProgramGroup* programGroupCH1,     // TODO: list
        unsigned int                 programGroupCH1Count,
        pyoptix::ProgramGroup        programGroupMS2,
        const pyoptix::ProgramGroup* programGroupCH2,     // TODO: list
        unsigned int                 programGroupCH2Count,
        unsigned int*                directCallableStackSizeFromTraversal,
        unsigned int*                directCallableStackSizeFromState,
        unsigned int*                continuationStackSize 
        )
{
    PYOPTIX_CHECK( 
        optixUtilComputeStackSizesSimplePathTracer( 
            programGroupRG.programGroup,
            programGroupMS1.programGroup,
            &programGroupCH1->programGroup, // TODO:Fix
            programGroupCH1Count,
            programGroupMS2.programGroup,
            &programGroupCH2->programGroup, // TODO:Fix
            programGroupCH2Count,
            directCallableStackSizeFromTraversal,
            directCallableStackSizeFromState,
            continuationStackSize 
            )
        );
}


} // end namespace util
} // end namespace pyoptix


PYBIND11_MODULE( optix, m ) 
{
    m.doc() = R"pbdoc(
        OptiX API 
        -----------------------

        .. currentmodule:: optix

        .. autosummary::
           :toctree: _generate

    )pbdoc";

    //---------------------------------------------------------------------------
    //
    // Module methods 
    //
    //---------------------------------------------------------------------------
    m.def( "init", &pyoptix::init);
    m.def( "deviceContextCreate", &pyoptix::deviceContextCreate);
    m.def( "getErrorName", &pyoptix::getErrorName );
    m.def( "getErrorString", &pyoptix::getErrorString );
    m.def( "launch", &pyoptix::launch );
    m.def( "sbtRecordPackHeader", &pyoptix::sbtRecordPackHeader );
    m.def( "convertPointerToTraversableHandle", &pyoptix::convertPointerToTraversableHandle );


    //--------------------------------------------------------------------------
    //
    // Structs for interfacing with CUDA
    //
    //--------------------------------------------------------------------------
    auto m_util = m.def_submodule( "util", nullptr /*TODO: docstring*/ );
    m_util.def( "accumulateStackSizes", &pyoptix::util::accumulateStackSizes );
    m_util.def( "computeStackSizes", &pyoptix::util::computeStackSizes );
    m_util.def( "computeStackSizesDCSplit", &pyoptix::util::computeStackSizesDCSplit );
    m_util.def( "computeStackSizesCssCCTree", &pyoptix::util::computeStackSizesCssCCTree );
    m_util.def( "computeStackSizesSimplePathTracer", &pyoptix::util::computeStackSizesSimplePathTracer );

    //--------------------------------------------------------------------------
    //
    // Enumerations 
    //
    //--------------------------------------------------------------------------

    py::enum_<OptixResult>(m, "Result")
        .value( "SUCCESS", OPTIX_SUCCESS )
        .value( "ERROR_INVALID_VALUE", OPTIX_ERROR_INVALID_VALUE )
        .value( "ERROR_HOST_OUT_OF_MEMORY", OPTIX_ERROR_HOST_OUT_OF_MEMORY )
        .value( "ERROR_INVALID_OPERATION", OPTIX_ERROR_INVALID_OPERATION )
        .value( "ERROR_FILE_IO_ERROR", OPTIX_ERROR_FILE_IO_ERROR )
        .value( "ERROR_INVALID_FILE_FORMAT", OPTIX_ERROR_INVALID_FILE_FORMAT )
        .value( "ERROR_DISK_CACHE_INVALID_PATH", OPTIX_ERROR_DISK_CACHE_INVALID_PATH )
        .value( "ERROR_DISK_CACHE_PERMISSION_ERROR", OPTIX_ERROR_DISK_CACHE_PERMISSION_ERROR )
        .value( "ERROR_DISK_CACHE_DATABASE_ERROR", OPTIX_ERROR_DISK_CACHE_DATABASE_ERROR )
        .value( "ERROR_DISK_CACHE_INVALID_DATA", OPTIX_ERROR_DISK_CACHE_INVALID_DATA )
        .value( "ERROR_LAUNCH_FAILURE", OPTIX_ERROR_LAUNCH_FAILURE )
        .value( "ERROR_INVALID_DEVICE_CONTEXT", OPTIX_ERROR_INVALID_DEVICE_CONTEXT )
        .value( "ERROR_CUDA_NOT_INITIALIZED", OPTIX_ERROR_CUDA_NOT_INITIALIZED )
        .value( "ERROR_VALIDATION_FAILURE", OPTIX_ERROR_VALIDATION_FAILURE )
        .value( "ERROR_INVALID_PTX", OPTIX_ERROR_INVALID_PTX )
        .value( "ERROR_INVALID_LAUNCH_PARAMETER", OPTIX_ERROR_INVALID_LAUNCH_PARAMETER )
        .value( "ERROR_INVALID_PAYLOAD_ACCESS", OPTIX_ERROR_INVALID_PAYLOAD_ACCESS )
        .value( "ERROR_INVALID_ATTRIBUTE_ACCESS", OPTIX_ERROR_INVALID_ATTRIBUTE_ACCESS )
        .value( "ERROR_INVALID_FUNCTION_USE", OPTIX_ERROR_INVALID_FUNCTION_USE )
        .value( "ERROR_INVALID_FUNCTION_ARGUMENTS", OPTIX_ERROR_INVALID_FUNCTION_ARGUMENTS )
        .value( "ERROR_PIPELINE_OUT_OF_CONSTANT_MEMORY", OPTIX_ERROR_PIPELINE_OUT_OF_CONSTANT_MEMORY )
        .value( "ERROR_PIPELINE_LINK_ERROR", OPTIX_ERROR_PIPELINE_LINK_ERROR )
        .value( "ERROR_INTERNAL_COMPILER_ERROR", OPTIX_ERROR_INTERNAL_COMPILER_ERROR )
        .value( "ERROR_DENOISER_MODEL_NOT_SET", OPTIX_ERROR_DENOISER_MODEL_NOT_SET )
        .value( "ERROR_DENOISER_NOT_INITIALIZED", OPTIX_ERROR_DENOISER_NOT_INITIALIZED )
        .value( "ERROR_ACCEL_NOT_COMPATIBLE", OPTIX_ERROR_ACCEL_NOT_COMPATIBLE )
        .value( "ERROR_NOT_SUPPORTED", OPTIX_ERROR_NOT_SUPPORTED )
        .value( "ERROR_UNSUPPORTED_ABI_VERSION", OPTIX_ERROR_UNSUPPORTED_ABI_VERSION )
        .value( "ERROR_FUNCTION_TABLE_SIZE_MISMATCH", OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH )
        .value( "ERROR_INVALID_ENTRY_FUNCTION_OPTIONS", OPTIX_ERROR_INVALID_ENTRY_FUNCTION_OPTIONS )
        .value( "ERROR_LIBRARY_NOT_FOUND", OPTIX_ERROR_LIBRARY_NOT_FOUND )
        .value( "ERROR_ENTRY_SYMBOL_NOT_FOUND", OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND )
        .value( "ERROR_LIBRARY_UNLOAD_FAILURE", OPTIX_ERROR_LIBRARY_UNLOAD_FAILURE )
        .value( "ERROR_CUDA_ERROR", OPTIX_ERROR_CUDA_ERROR )
        .value( "ERROR_INTERNAL_ERROR", OPTIX_ERROR_INTERNAL_ERROR )
        .value( "ERROR_UNKNOWN", OPTIX_ERROR_UNKNOWN )
        .export_values();

    py::enum_<OptixDeviceProperty>(m, "DeviceProperty")
        .value( "DEVICE_PROPERTY_LIMIT_MAX_TRACE_DEPTH", OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRACE_DEPTH )
        .value( "DEVICE_PROPERTY_LIMIT_MAX_TRAVERSABLE_GRAPH_DEPTH", OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRAVERSABLE_GRAPH_DEPTH )
        .value( "DEVICE_PROPERTY_LIMIT_MAX_PRIMITIVES_PER_GAS", OPTIX_DEVICE_PROPERTY_LIMIT_MAX_PRIMITIVES_PER_GAS )
        .value( "DEVICE_PROPERTY_LIMIT_MAX_INSTANCES_PER_IAS", OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCES_PER_IAS )
        .value( "DEVICE_PROPERTY_RTCORE_VERSION", OPTIX_DEVICE_PROPERTY_RTCORE_VERSION )
        .value( "DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID", OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID )
        .value( "DEVICE_PROPERTY_LIMIT_NUM_BITS_INSTANCE_VISIBILITY_MASK", OPTIX_DEVICE_PROPERTY_LIMIT_NUM_BITS_INSTANCE_VISIBILITY_MASK )
        .value( "DEVICE_PROPERTY_LIMIT_MAX_SBT_RECORDS_PER_GAS", OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_RECORDS_PER_GAS )
        .value( "DEVICE_PROPERTY_LIMIT_MAX_SBT_OFFSET", OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_OFFSET )
        .export_values();

    py::enum_<OptixDeviceContextValidationMode>(m, "DeviceContextValidationMode")
        .value( "DEVICE_CONTEXT_VALIDATION_MODE_OFF", OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF )
        .value( "DEVICE_CONTEXT_VALIDATION_MODE_ALL", OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL )
        .export_values();

    py::enum_<OptixGeometryFlags>(m, "GeometryFlags")
        .value( "GEOMETRY_FLAG_NONE", OPTIX_GEOMETRY_FLAG_NONE )
        .value( "GEOMETRY_FLAG_DISABLE_ANYHIT", OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT )
        .value( "GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL", OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL )
        .export_values();

    py::enum_<OptixHitKind>(m, "HitKind")
        .value( "HIT_KIND_TRIANGLE_FRONT_FACE", OPTIX_HIT_KIND_TRIANGLE_FRONT_FACE )
        .value( "HIT_KIND_TRIANGLE_BACK_FACE", OPTIX_HIT_KIND_TRIANGLE_BACK_FACE )
        .export_values();

    py::enum_<OptixIndicesFormat>(m, "IndicesFormat")
        .value( "INDICES_FORMAT_NONE", OPTIX_INDICES_FORMAT_NONE )
        .value( "INDICES_FORMAT_UNSIGNED_SHORT3", OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3 )
        .value( "INDICES_FORMAT_UNSIGNED_INT3", OPTIX_INDICES_FORMAT_UNSIGNED_INT3 )
        .export_values();

    py::enum_<OptixVertexFormat>(m, "VertexFormat")
        .value( "VERTEX_FORMAT_NONE", OPTIX_VERTEX_FORMAT_NONE )
        .value( "VERTEX_FORMAT_FLOAT3", OPTIX_VERTEX_FORMAT_FLOAT3 )
        .value( "VERTEX_FORMAT_FLOAT2", OPTIX_VERTEX_FORMAT_FLOAT2 )
        .value( "VERTEX_FORMAT_HALF3", OPTIX_VERTEX_FORMAT_HALF3 )
        .value( "VERTEX_FORMAT_HALF2", OPTIX_VERTEX_FORMAT_HALF2 )
        .value( "VERTEX_FORMAT_SNORM16_3", OPTIX_VERTEX_FORMAT_SNORM16_3 )
        .value( "VERTEX_FORMAT_SNORM16_2", OPTIX_VERTEX_FORMAT_SNORM16_2 )
        .export_values();

    py::enum_<OptixTransformFormat>(m, "TransformFormat")
        .value( "TRANSFORM_FORMAT_NONE", OPTIX_TRANSFORM_FORMAT_NONE )
        .value( "TRANSFORM_FORMAT_MATRIX_FLOAT12", OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12 )
        .export_values();

    py::enum_<OptixPrimitiveType>(m, "PrimitiveType")
        .value( "PRIMITIVE_TYPE_CUSTOM", OPTIX_PRIMITIVE_TYPE_CUSTOM )
        .value( "PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE", OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE )
        .value( "PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE", OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE )
        .value( "PRIMITIVE_TYPE_ROUND_LINEAR", OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR )
        .value( "PRIMITIVE_TYPE_TRIANGLE", OPTIX_PRIMITIVE_TYPE_TRIANGLE )
        .export_values();

    py::enum_<OptixPrimitiveTypeFlags>(m, "PrimitiveTypeFlags")
        .value( "PRIMITIVE_TYPE_FLAGS_CUSTOM", OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM )
        .value( "PRIMITIVE_TYPE_FLAGS_ROUND_QUADRATIC_BSPLINE", OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_QUADRATIC_BSPLINE )
        .value( "PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BSPLINE", OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BSPLINE )
        .value( "PRIMITIVE_TYPE_FLAGS_ROUND_LINEAR", OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_LINEAR )
        .value( "PRIMITIVE_TYPE_FLAGS_TRIANGLE", OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE )
        .export_values();

    py::enum_<OptixBuildInputType>(m, "BuildInputType")
        .value( "BUILD_INPUT_TYPE_TRIANGLES", OPTIX_BUILD_INPUT_TYPE_TRIANGLES )
        .value( "BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES", OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES )
        .value( "BUILD_INPUT_TYPE_INSTANCES", OPTIX_BUILD_INPUT_TYPE_INSTANCES )
        .value( "BUILD_INPUT_TYPE_INSTANCE_POINTERS", OPTIX_BUILD_INPUT_TYPE_INSTANCE_POINTERS )
        .value( "BUILD_INPUT_TYPE_CURVES", OPTIX_BUILD_INPUT_TYPE_CURVES )
        .export_values();

    py::enum_<OptixInstanceFlags>(m, "InstanceFlags")
        .value( "INSTANCE_FLAG_NONE", OPTIX_INSTANCE_FLAG_NONE )
        .value( "INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING", OPTIX_INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING )
        .value( "INSTANCE_FLAG_FLIP_TRIANGLE_FACING", OPTIX_INSTANCE_FLAG_FLIP_TRIANGLE_FACING )
        .value( "INSTANCE_FLAG_DISABLE_ANYHIT", OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT )
        .value( "INSTANCE_FLAG_ENFORCE_ANYHIT", OPTIX_INSTANCE_FLAG_ENFORCE_ANYHIT )
        .value( "INSTANCE_FLAG_DISABLE_TRANSFORM", OPTIX_INSTANCE_FLAG_DISABLE_TRANSFORM )
        .export_values();

    py::enum_<OptixBuildFlags>(m, "BuildFlags")
        .value( "BUILD_FLAG_NONE", OPTIX_BUILD_FLAG_NONE )
        .value( "BUILD_FLAG_ALLOW_UPDATE", OPTIX_BUILD_FLAG_ALLOW_UPDATE )
        .value( "BUILD_FLAG_ALLOW_COMPACTION", OPTIX_BUILD_FLAG_ALLOW_COMPACTION )
        .value( "BUILD_FLAG_PREFER_FAST_TRACE", OPTIX_BUILD_FLAG_PREFER_FAST_TRACE )
        .value( "BUILD_FLAG_PREFER_FAST_BUILD", OPTIX_BUILD_FLAG_PREFER_FAST_BUILD )
        .value( "BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS", OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS )
        .export_values();

    py::enum_<OptixBuildOperation>(m, "BuildOperation")
        .value( "BUILD_OPERATION_BUILD", OPTIX_BUILD_OPERATION_BUILD )
        .value( "BUILD_OPERATION_UPDATE", OPTIX_BUILD_OPERATION_UPDATE )
        .export_values();

    py::enum_<OptixMotionFlags>(m, "MotionFlags")
        .value( "MOTION_FLAG_NONE", OPTIX_MOTION_FLAG_NONE )
        .value( "MOTION_FLAG_START_VANISH", OPTIX_MOTION_FLAG_START_VANISH )
        .value( "MOTION_FLAG_END_VANISH", OPTIX_MOTION_FLAG_END_VANISH )
        .export_values();

    py::enum_<OptixAccelPropertyType>(m, "AccelPropertyType")
        .value( "PROPERTY_TYPE_COMPACTED_SIZE", OPTIX_PROPERTY_TYPE_COMPACTED_SIZE )
        .value( "PROPERTY_TYPE_AABBS", OPTIX_PROPERTY_TYPE_AABBS )
        .export_values();

    py::enum_<OptixTraversableType>(m, "TraversableType")
        .value( "TRAVERSABLE_TYPE_STATIC_TRANSFORM", OPTIX_TRAVERSABLE_TYPE_STATIC_TRANSFORM )
        .value( "TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM", OPTIX_TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM )
        .value( "TRAVERSABLE_TYPE_SRT_MOTION_TRANSFORM", OPTIX_TRAVERSABLE_TYPE_SRT_MOTION_TRANSFORM )
        .export_values();

    py::enum_<OptixPixelFormat>(m, "PixelFormat")
        .value( "PIXEL_FORMAT_HALF3", OPTIX_PIXEL_FORMAT_HALF3 )
        .value( "PIXEL_FORMAT_HALF4", OPTIX_PIXEL_FORMAT_HALF4 )
        .value( "PIXEL_FORMAT_FLOAT3", OPTIX_PIXEL_FORMAT_FLOAT3 )
        .value( "PIXEL_FORMAT_FLOAT4", OPTIX_PIXEL_FORMAT_FLOAT4 )
        .value( "PIXEL_FORMAT_UCHAR3", OPTIX_PIXEL_FORMAT_UCHAR3 )
        .value( "PIXEL_FORMAT_UCHAR4", OPTIX_PIXEL_FORMAT_UCHAR4 )
        .export_values();

    py::enum_<OptixDenoiserModelKind>(m, "DenoiserModelKind")
        .value( "DENOISER_MODEL_KIND_USER", OPTIX_DENOISER_MODEL_KIND_USER )
        .value( "DENOISER_MODEL_KIND_LDR", OPTIX_DENOISER_MODEL_KIND_LDR )
        .value( "DENOISER_MODEL_KIND_HDR", OPTIX_DENOISER_MODEL_KIND_HDR )
        .value( "DENOISER_MODEL_KIND_AOV", OPTIX_DENOISER_MODEL_KIND_AOV )
        .export_values();

    py::enum_<OptixRayFlags>(m, "RayFlags")
        .value( "RAY_FLAG_NONE", OPTIX_RAY_FLAG_NONE )
        .value( "RAY_FLAG_DISABLE_ANYHIT", OPTIX_RAY_FLAG_DISABLE_ANYHIT )
        .value( "RAY_FLAG_ENFORCE_ANYHIT", OPTIX_RAY_FLAG_ENFORCE_ANYHIT )
        .value( "RAY_FLAG_TERMINATE_ON_FIRST_HIT", OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT )
        .value( "RAY_FLAG_DISABLE_CLOSESTHIT", OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT )
        .value( "RAY_FLAG_CULL_BACK_FACING_TRIANGLES", OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES )
        .value( "RAY_FLAG_CULL_FRONT_FACING_TRIANGLES", OPTIX_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES )
        .value( "RAY_FLAG_CULL_DISABLED_ANYHIT", OPTIX_RAY_FLAG_CULL_DISABLED_ANYHIT )
        .value( "RAY_FLAG_CULL_ENFORCED_ANYHIT", OPTIX_RAY_FLAG_CULL_ENFORCED_ANYHIT )
        .export_values();

    py::enum_<OptixTransformType>(m, "TransformType")
        .value( "TRANSFORM_TYPE_NONE", OPTIX_TRANSFORM_TYPE_NONE )
        .value( "TRANSFORM_TYPE_STATIC_TRANSFORM", OPTIX_TRANSFORM_TYPE_STATIC_TRANSFORM )
        .value( "TRANSFORM_TYPE_MATRIX_MOTION_TRANSFORM", OPTIX_TRANSFORM_TYPE_MATRIX_MOTION_TRANSFORM )
        .value( "TRANSFORM_TYPE_SRT_MOTION_TRANSFORM", OPTIX_TRANSFORM_TYPE_SRT_MOTION_TRANSFORM )
        .value( "TRANSFORM_TYPE_INSTANCE", OPTIX_TRANSFORM_TYPE_INSTANCE )
        .export_values();

    py::enum_<OptixTraversableGraphFlags>(m, "TraversableGraphFlags")
        .value( "TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY", OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY )
        .value( "TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS", OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS )
        .value( "TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING", OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING )
        .export_values();

    py::enum_<OptixCompileOptimizationLevel>(m, "CompileOptimizationLevel")
        .value( "COMPILE_OPTIMIZATION_DEFAULT", OPTIX_COMPILE_OPTIMIZATION_DEFAULT )
        .value( "COMPILE_OPTIMIZATION_LEVEL_0", OPTIX_COMPILE_OPTIMIZATION_LEVEL_0 )
        .value( "COMPILE_OPTIMIZATION_LEVEL_1", OPTIX_COMPILE_OPTIMIZATION_LEVEL_1 )
        .value( "COMPILE_OPTIMIZATION_LEVEL_2", OPTIX_COMPILE_OPTIMIZATION_LEVEL_2 )
        .value( "COMPILE_OPTIMIZATION_LEVEL_3", OPTIX_COMPILE_OPTIMIZATION_LEVEL_3 )
        .export_values();

    py::enum_<OptixCompileDebugLevel>(m, "CompileDebugLevel")
        .value( "COMPILE_DEBUG_LEVEL_DEFAULT", OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT )
        .value( "COMPILE_DEBUG_LEVEL_NONE", OPTIX_COMPILE_DEBUG_LEVEL_NONE )
        .value( "COMPILE_DEBUG_LEVEL_LINEINFO", OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO )
        .value( "COMPILE_DEBUG_LEVEL_FULL", OPTIX_COMPILE_DEBUG_LEVEL_FULL )
        .export_values();

    py::enum_<OptixProgramGroupKind>(m, "ProgramGroupKind")
        .value( "PROGRAM_GROUP_KIND_RAYGEN", OPTIX_PROGRAM_GROUP_KIND_RAYGEN )
        .value( "PROGRAM_GROUP_KIND_MISS", OPTIX_PROGRAM_GROUP_KIND_MISS )
        .value( "PROGRAM_GROUP_KIND_EXCEPTION", OPTIX_PROGRAM_GROUP_KIND_EXCEPTION )
        .value( "PROGRAM_GROUP_KIND_HITGROUP", OPTIX_PROGRAM_GROUP_KIND_HITGROUP )
        .value( "PROGRAM_GROUP_KIND_CALLABLES", OPTIX_PROGRAM_GROUP_KIND_CALLABLES )
        .export_values();

    py::enum_<OptixProgramGroupFlags>(m, "ProgramGroupFlags")
        .value( "PROGRAM_GROUP_FLAGS_NONE", OPTIX_PROGRAM_GROUP_FLAGS_NONE )
        .export_values();

    py::enum_<OptixExceptionCodes>(m, "ExceptionCodes")
        .value( "EXCEPTION_CODE_STACK_OVERFLOW", OPTIX_EXCEPTION_CODE_STACK_OVERFLOW )
        .value( "EXCEPTION_CODE_TRACE_DEPTH_EXCEEDED", OPTIX_EXCEPTION_CODE_TRACE_DEPTH_EXCEEDED )
        .value( "EXCEPTION_CODE_TRAVERSAL_DEPTH_EXCEEDED", OPTIX_EXCEPTION_CODE_TRAVERSAL_DEPTH_EXCEEDED )
        .value( "EXCEPTION_CODE_TRAVERSAL_INVALID_TRAVERSABLE", OPTIX_EXCEPTION_CODE_TRAVERSAL_INVALID_TRAVERSABLE )
        .value( "EXCEPTION_CODE_TRAVERSAL_INVALID_MISS_SBT", OPTIX_EXCEPTION_CODE_TRAVERSAL_INVALID_MISS_SBT )
        .value( "EXCEPTION_CODE_TRAVERSAL_INVALID_HIT_SBT", OPTIX_EXCEPTION_CODE_TRAVERSAL_INVALID_HIT_SBT )
        .value( "EXCEPTION_CODE_UNSUPPORTED_PRIMITIVE_TYPE", OPTIX_EXCEPTION_CODE_UNSUPPORTED_PRIMITIVE_TYPE )
        .value( "EXCEPTION_CODE_INVALID_RAY", OPTIX_EXCEPTION_CODE_INVALID_RAY )
        .value( "EXCEPTION_CODE_CALLABLE_PARAMETER_MISMATCH", OPTIX_EXCEPTION_CODE_CALLABLE_PARAMETER_MISMATCH )
        .value( "EXCEPTION_CODE_BUILTIN_IS_MISMATCH", OPTIX_EXCEPTION_CODE_BUILTIN_IS_MISMATCH )
        .value( "EXCEPTION_CODE_CALLABLE_INVALID_SBT", OPTIX_EXCEPTION_CODE_CALLABLE_INVALID_SBT )
        .value( "EXCEPTION_CODE_CALLABLE_NO_DC_SBT_RECORD", OPTIX_EXCEPTION_CODE_CALLABLE_NO_DC_SBT_RECORD )
        .value( "EXCEPTION_CODE_CALLABLE_NO_CC_SBT_RECORD", OPTIX_EXCEPTION_CODE_CALLABLE_NO_CC_SBT_RECORD )
        .value( "EXCEPTION_CODE_UNSUPPORTED_SINGLE_LEVEL_GAS", OPTIX_EXCEPTION_CODE_UNSUPPORTED_SINGLE_LEVEL_GAS )
        .export_values();

    py::enum_<OptixExceptionFlags>(m, "ExceptionFlags")
        .value( "EXCEPTION_FLAG_NONE", OPTIX_EXCEPTION_FLAG_NONE )
        .value( "EXCEPTION_FLAG_STACK_OVERFLOW", OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW )
        .value( "EXCEPTION_FLAG_TRACE_DEPTH", OPTIX_EXCEPTION_FLAG_TRACE_DEPTH )
        .value( "EXCEPTION_FLAG_USER", OPTIX_EXCEPTION_FLAG_USER )
        .value( "EXCEPTION_FLAG_DEBUG", OPTIX_EXCEPTION_FLAG_DEBUG )
        .export_values();

    py::enum_<OptixQueryFunctionTableOptions>(m, "QueryFunctionTableOptions")
        .value( "QUERY_FUNCTION_TABLE_OPTION_DUMMY", OPTIX_QUERY_FUNCTION_TABLE_OPTION_DUMMY )
        .export_values();

    
    //---------------------------------------------------------------------------
    //
    // Param types
    //
    //---------------------------------------------------------------------------

    py::class_<pyoptix::DeviceContextOptions>(m, "DeviceContextOptions")
        .def( py::init( []() { 
            return std::unique_ptr<pyoptix::DeviceContextOptions>( 
                new pyoptix::DeviceContextOptions{} 
            ); 
        } ) )
        .def_property( "logCallbackFunction", 
                [](const pyoptix::DeviceContextOptions& self) 
                { return self.logCallbackFunction; }, 
                [](pyoptix::DeviceContextOptions& self, py::object val)
                { 
                    self.logCallbackFunction= val; 
                    self.options.logCallbackFunction = pyoptix::context_log_cb; 
                    self.options.logCallbackData = val.ptr();
                }
            )
        .def_property("logCallbackLevel", 
                [](const pyoptix::DeviceContextOptions& self) 
                { return self.options.logCallbackLevel;}, 
                [](pyoptix::DeviceContextOptions& self, int val) 
                { self.options.logCallbackLevel = val; }
            )
        .def_property("validationMode", 
                [](const pyoptix::DeviceContextOptions& self) 
                { return self.options.validationMode; }, 
                [](pyoptix::DeviceContextOptions& self, OptixDeviceContextValidationMode val) 
                { self.options.validationMode = val; }
            )
        ;

    py::class_<OptixBuildInputTriangleArray>(m, "BuildInputTriangleArray")
        .def( py::init([]() { return std::unique_ptr<OptixBuildInputTriangleArray>(new OptixBuildInputTriangleArray{} ); } ) )
        .def_readwrite( "vertexBuffers", &OptixBuildInputTriangleArray::vertexBuffers )
        .def_readwrite( "numVertices", &OptixBuildInputTriangleArray::numVertices )
        .def_readwrite( "vertexFormat", &OptixBuildInputTriangleArray::vertexFormat )
        .def_readwrite( "vertexStrideInBytes", &OptixBuildInputTriangleArray::vertexStrideInBytes )
        .def_readwrite( "indexBuffer", &OptixBuildInputTriangleArray::indexBuffer )
        .def_readwrite( "numIndexTriplets", &OptixBuildInputTriangleArray::numIndexTriplets )
        .def_readwrite( "indexFormat", &OptixBuildInputTriangleArray::indexFormat )
        .def_readwrite( "indexStrideInBytes", &OptixBuildInputTriangleArray::indexStrideInBytes )
        .def_readwrite( "preTransform", &OptixBuildInputTriangleArray::preTransform )
        .def_readwrite( "flags", &OptixBuildInputTriangleArray::flags )
        .def_readwrite( "numSbtRecords", &OptixBuildInputTriangleArray::numSbtRecords )
        .def_readwrite( "sbtIndexOffsetBuffer", &OptixBuildInputTriangleArray::sbtIndexOffsetBuffer )
        .def_readwrite( "sbtIndexOffsetSizeInBytes", &OptixBuildInputTriangleArray::sbtIndexOffsetSizeInBytes )
        .def_readwrite( "sbtIndexOffsetStrideInBytes", &OptixBuildInputTriangleArray::sbtIndexOffsetStrideInBytes )
        .def_readwrite( "primitiveIndexOffset", &OptixBuildInputTriangleArray::primitiveIndexOffset )
        .def_readwrite( "transformFormat", &OptixBuildInputTriangleArray::transformFormat )
        ;

    py::class_<OptixBuildInputCurveArray>(m, "BuildInputCurveArray")
        .def( py::init([]() { return std::unique_ptr<OptixBuildInputCurveArray>(new OptixBuildInputCurveArray{} ); } ) )
        .def_readwrite( "curveType", &OptixBuildInputCurveArray::curveType )
        .def_readwrite( "numPrimitives", &OptixBuildInputCurveArray::numPrimitives )
        .def_readwrite( "vertexBuffers", &OptixBuildInputCurveArray::vertexBuffers )
        .def_readwrite( "numVertices", &OptixBuildInputCurveArray::numVertices )
        .def_readwrite( "vertexStrideInBytes", &OptixBuildInputCurveArray::vertexStrideInBytes )
        .def_readwrite( "widthBuffers", &OptixBuildInputCurveArray::widthBuffers )
        .def_readwrite( "widthStrideInBytes", &OptixBuildInputCurveArray::widthStrideInBytes )
        .def_readwrite( "normalBuffers", &OptixBuildInputCurveArray::normalBuffers )
        .def_readwrite( "normalStrideInBytes", &OptixBuildInputCurveArray::normalStrideInBytes )
        .def_readwrite( "indexBuffer", &OptixBuildInputCurveArray::indexBuffer )
        .def_readwrite( "indexStrideInBytes", &OptixBuildInputCurveArray::indexStrideInBytes )
        .def_readwrite( "flag", &OptixBuildInputCurveArray::flag )
        .def_readwrite( "primitiveIndexOffset", &OptixBuildInputCurveArray::primitiveIndexOffset )
        ;

    py::class_<OptixAabb>(m, "Aabb")
        .def( py::init([]() { return std::unique_ptr<OptixAabb>(new OptixAabb{} ); } ) )
        .def_readwrite( "minX", &OptixAabb::minX )
        .def_readwrite( "minY", &OptixAabb::minY )
        .def_readwrite( "minZ", &OptixAabb::minZ )
        .def_readwrite( "maxX", &OptixAabb::maxX )
        .def_readwrite( "maxY", &OptixAabb::maxY )
        .def_readwrite( "maxZ", &OptixAabb::maxZ )
        ;

    py::class_<OptixBuildInputCustomPrimitiveArray>(m, "BuildInputCustomPrimitiveArray")
        .def( py::init([]() { return std::unique_ptr<OptixBuildInputCustomPrimitiveArray>(new OptixBuildInputCustomPrimitiveArray{} ); } ) )
        .def_readwrite( "aabbBuffers", &OptixBuildInputCustomPrimitiveArray::aabbBuffers )
        .def_readwrite( "numPrimitives", &OptixBuildInputCustomPrimitiveArray::numPrimitives )
        .def_readwrite( "strideInBytes", &OptixBuildInputCustomPrimitiveArray::strideInBytes )
        .def_readwrite( "flags", &OptixBuildInputCustomPrimitiveArray::flags )
        .def_readwrite( "numSbtRecords", &OptixBuildInputCustomPrimitiveArray::numSbtRecords )
        .def_readwrite( "sbtIndexOffsetBuffer", &OptixBuildInputCustomPrimitiveArray::sbtIndexOffsetBuffer )
        .def_readwrite( "sbtIndexOffsetSizeInBytes", &OptixBuildInputCustomPrimitiveArray::sbtIndexOffsetSizeInBytes )
        .def_readwrite( "sbtIndexOffsetStrideInBytes", &OptixBuildInputCustomPrimitiveArray::sbtIndexOffsetStrideInBytes )
        .def_readwrite( "primitiveIndexOffset", &OptixBuildInputCustomPrimitiveArray::primitiveIndexOffset )
        ;

    py::class_<OptixBuildInputInstanceArray>(m, "BuildInputInstanceArray")
        .def( py::init([]() { return std::unique_ptr<OptixBuildInputInstanceArray>(new OptixBuildInputInstanceArray{} ); } ) )
  //      .def_readwrite( "instances", &OptixBuildInputInstanceArray::instances )
   //     .def_readwrite( "numInstances", &OptixBuildInputInstanceArray::numInstances )
        ;

    py::class_<OptixBuildInput>(m, "BuildInput")
        .def( py::init([]() { return std::unique_ptr<OptixBuildInput>(new OptixBuildInput{} ); } ) )
        .def_readwrite( "type", &OptixBuildInput::type )
        .def_readwrite( "triangleArray", &OptixBuildInput::triangleArray )
        .def_readwrite( "curveArray", &OptixBuildInput::curveArray )
        .def_readwrite( "customPrimitiveArray", &OptixBuildInput::customPrimitiveArray )
        .def_readwrite( "instanceArray", &OptixBuildInput::instanceArray )
        ;

    py::class_<OptixInstance>(m, "Instance")
        .def( py::init([]() { return std::unique_ptr<OptixInstance>(new OptixInstance{} ); } ) )
        .def_readwrite( "instanceId", &OptixInstance::instanceId )
        .def_readwrite( "sbtOffset", &OptixInstance::sbtOffset )
        .def_readwrite( "visibilityMask", &OptixInstance::visibilityMask )
        .def_readwrite( "flags", &OptixInstance::flags )
        .def_readwrite( "traversableHandle", &OptixInstance::traversableHandle )
        ;

    py::class_<OptixMotionOptions>(m, "MotionOptions")
        .def( py::init([]() { return std::unique_ptr<OptixMotionOptions>(new OptixMotionOptions{} ); } ) )
        .def_readwrite( "numKeys", &OptixMotionOptions::numKeys )
        .def_readwrite( "flags", &OptixMotionOptions::flags )
        .def_readwrite( "timeBegin", &OptixMotionOptions::timeBegin )
        .def_readwrite( "timeEnd", &OptixMotionOptions::timeEnd )
        ;

    py::class_<OptixAccelBuildOptions>(m, "AccelBuildOptions")
        .def( py::init([]() { return std::unique_ptr<OptixAccelBuildOptions>(new OptixAccelBuildOptions{} ); } ) )
        .def_readwrite( "buildFlags", &OptixAccelBuildOptions::buildFlags )
        .def_readwrite( "operation", &OptixAccelBuildOptions::operation )
        .def_readwrite( "motionOptions", &OptixAccelBuildOptions::motionOptions )
        ;

    py::class_<OptixAccelBufferSizes>(m, "AccelBufferSizes")
        .def( py::init([]() { return std::unique_ptr<OptixAccelBufferSizes>(new OptixAccelBufferSizes{} ); } ) )
        .def_readwrite( "outputSizeInBytes", &OptixAccelBufferSizes::outputSizeInBytes )
        .def_readwrite( "tempSizeInBytes", &OptixAccelBufferSizes::tempSizeInBytes )
        .def_readwrite( "tempUpdateSizeInBytes", &OptixAccelBufferSizes::tempUpdateSizeInBytes )
        ;

    py::class_<OptixAccelEmitDesc>(m, "AccelEmitDesc")
        .def( py::init([]() { return std::unique_ptr<OptixAccelEmitDesc>(new OptixAccelEmitDesc{} ); } ) )
        .def_readwrite( "result", &OptixAccelEmitDesc::result )
        .def_readwrite( "type", &OptixAccelEmitDesc::type )
        ;

    py::class_<OptixAccelRelocationInfo>(m, "AccelRelocationInfo")
        .def( py::init([]() { return std::unique_ptr<OptixAccelRelocationInfo>(new OptixAccelRelocationInfo{} ); } ) )
        ;

    py::class_<OptixStaticTransform>(m, "StaticTransform")
        .def( py::init([]() { return std::unique_ptr<OptixStaticTransform>(new OptixStaticTransform{} ); } ) )
        .def_readwrite( "child", &OptixStaticTransform::child )
        ;

    py::class_<OptixMatrixMotionTransform>(m, "MatrixMotionTransform")
        .def( py::init([]() { return std::unique_ptr<OptixMatrixMotionTransform>(new OptixMatrixMotionTransform{} ); } ) )
        .def_readwrite( "child", &OptixMatrixMotionTransform::child )
        .def_readwrite( "motionOptions", &OptixMatrixMotionTransform::motionOptions )
        ;

    py::class_<OptixSRTData>(m, "SRTData")
        .def( py::init([]() { return std::unique_ptr<OptixSRTData>(new OptixSRTData{} ); } ) )
        .def_readwrite( "tz", &OptixSRTData::tz )
        ;

    py::class_<OptixSRTMotionTransform>(m, "SRTMotionTransform")
        .def( py::init([]() { return std::unique_ptr<OptixSRTMotionTransform>(new OptixSRTMotionTransform{} ); } ) )
        .def_readwrite( "child", &OptixSRTMotionTransform::child )
        .def_readwrite( "motionOptions", &OptixSRTMotionTransform::motionOptions )
        ;

    py::class_<OptixImage2D>(m, "Image2D")
        .def( py::init([]() { return std::unique_ptr<OptixImage2D>(new OptixImage2D{} ); } ) )
        .def_readwrite( "data", &OptixImage2D::data )
        .def_readwrite( "width", &OptixImage2D::width )
        .def_readwrite( "height", &OptixImage2D::height )
        .def_readwrite( "rowStrideInBytes", &OptixImage2D::rowStrideInBytes )
        .def_readwrite( "pixelStrideInBytes", &OptixImage2D::pixelStrideInBytes )
        .def_readwrite( "format", &OptixImage2D::format )
        ;

    py::class_<OptixDenoiserOptions>(m, "DenoiserOptions")
        .def( py::init([]() { return std::unique_ptr<OptixDenoiserOptions>(new OptixDenoiserOptions{} ); } ) )
        .def_readwrite( "inputKind", &OptixDenoiserOptions::inputKind )
        ;

    py::class_<OptixDenoiserParams>(m, "DenoiserParams")
        .def( py::init([]() { return std::unique_ptr<OptixDenoiserParams>(new OptixDenoiserParams{} ); } ) )
        .def_readwrite( "denoiseAlpha", &OptixDenoiserParams::denoiseAlpha )
        .def_readwrite( "hdrIntensity", &OptixDenoiserParams::hdrIntensity )
        .def_readwrite( "blendFactor", &OptixDenoiserParams::blendFactor )
        .def_readwrite( "hdrAverageColor", &OptixDenoiserParams::hdrAverageColor )
        ;

    py::class_<OptixDenoiserSizes>(m, "DenoiserSizes")
        .def( py::init([]() { return std::unique_ptr<OptixDenoiserSizes>(new OptixDenoiserSizes{} ); } ) )
        .def_readwrite( "stateSizeInBytes", &OptixDenoiserSizes::stateSizeInBytes )
        .def_readwrite( "withOverlapScratchSizeInBytes", &OptixDenoiserSizes::withOverlapScratchSizeInBytes )
        .def_readwrite( "withoutOverlapScratchSizeInBytes", &OptixDenoiserSizes::withoutOverlapScratchSizeInBytes )
        .def_readwrite( "overlapWindowSizeInPixels", &OptixDenoiserSizes::overlapWindowSizeInPixels )
        ;

    py::class_<OptixModuleCompileBoundValueEntry>(m, "ModuleCompileBoundValueEntry")
        .def( py::init([]() { return std::unique_ptr<OptixModuleCompileBoundValueEntry>(new OptixModuleCompileBoundValueEntry{} ); } ) )
        .def_readwrite( "pipelineParamOffsetInBytes", &OptixModuleCompileBoundValueEntry::pipelineParamOffsetInBytes )
        .def_readwrite( "sizeInBytes", &OptixModuleCompileBoundValueEntry::sizeInBytes )
        .def_readwrite( "boundValuePtr", &OptixModuleCompileBoundValueEntry::boundValuePtr )
        .def_readwrite( "annotation", &OptixModuleCompileBoundValueEntry::annotation )
        ;

    py::class_<OptixModuleCompileOptions>(m, "ModuleCompileOptions")
        .def( py::init([]() { return std::unique_ptr<OptixModuleCompileOptions>(new OptixModuleCompileOptions{} ); } ) )
        .def_readwrite( "maxRegisterCount", &OptixModuleCompileOptions::maxRegisterCount )
        .def_readwrite( "optLevel", &OptixModuleCompileOptions::optLevel )
        .def_readwrite( "debugLevel", &OptixModuleCompileOptions::debugLevel )
        .def_readwrite( "boundValues", &OptixModuleCompileOptions::boundValues )
        .def_readwrite( "numBoundValues", &OptixModuleCompileOptions::numBoundValues )
        ;

    /*
    py::class_<pyoptix::ProgramGroupSingleModule>(m, "ProgramGroupSingleModule")
        .def( py::init([]() 
            { return std::unique_ptr<pyoptix::ProgramGroupSingleModule>(new pyoptix::ProgramGroupSingleModule{} ); }
        ) )
        .def_property("module", 
            [](const pyoptix::ProgramGroupSingleModule& self) 
            { return pyoptix::Module{ self.program_group.module }; }, 
            [](pyoptix::ProgramGroupSingleModule& self, const pyoptix::Module &val) 
            { self.program_group.module = val.module; }
        )
        .def_readwrite( "entryFunctionName", &pyoptix::ProgramGroupSingleModule::entryFunctionName )
        ;

    py::class_<OptixProgramGroupHitgroup>(m, "ProgramGroupHitgroup")
        .def( py::init([]() { return std::unique_ptr<OptixProgramGroupHitgroup>(new OptixProgramGroupHitgroup{} ); } ) )
        .def_property("moduleCH", 
                [](const OptixProgramGroupHitgroup& self) { return pyoptix::Module{ self.moduleCH}; }, 
                [](OptixProgramGroupHitgroup& self, const pyoptix::Module &val) { self.moduleCH = val.module; }
                )
        .def_readwrite( "entryFunctionNameCH", &OptixProgramGroupHitgroup::entryFunctionNameCH )
        .def_property("moduleAH", 
                [](const OptixProgramGroupHitgroup& self) { return pyoptix::Module{ self.moduleAH}; }, 
                [](OptixProgramGroupHitgroup& self, const pyoptix::Module &val) { self.moduleAH = val.module; }
                )
        .def_readwrite( "entryFunctionNameAH", &OptixProgramGroupHitgroup::entryFunctionNameAH )
        .def_property("moduleIS", 
                [](const OptixProgramGroupHitgroup& self) { return pyoptix::Module{ self.moduleIS}; }, 
                [](OptixProgramGroupHitgroup& self, const pyoptix::Module &val) { self.moduleIS = val.module; }
                )
        .def_readwrite( "entryFunctionNameIS", &OptixProgramGroupHitgroup::entryFunctionNameIS )
        ;

    py::class_<OptixProgramGroupCallables>(m, "ProgramGroupCallables")
        .def( py::init([]() { return std::unique_ptr<OptixProgramGroupCallables>(new OptixProgramGroupCallables{} ); } ) )
        .def_property("moduleDC", 
                [](const OptixProgramGroupCallables& self) { return pyoptix::Module{ self.moduleDC}; }, 
                [](OptixProgramGroupCallables& self, const pyoptix::Module &val) { self.moduleDC = val.module; }
                )
        .def_readwrite( "entryFunctionNameDC", &OptixProgramGroupCallables::entryFunctionNameDC )
        .def_property("moduleCC", 
                [](const OptixProgramGroupCallables& self) { return pyoptix::Module{ self.moduleCC}; }, 
                [](OptixProgramGroupCallables& self, const pyoptix::Module &val) { self.moduleCC = val.module; }
                )
        .def_readwrite( "entryFunctionNameCC", &OptixProgramGroupCallables::entryFunctionNameCC )
        ;

    */

    py::class_<pyoptix::ProgramGroupDesc>(m, "ProgramGroupDesc")
        .def( py::init([]() 
            { return std::unique_ptr<pyoptix::ProgramGroupDesc>(new pyoptix::ProgramGroupDesc{} ); } 
        ) )
        .def_property( "kind", 
            []( pyoptix::ProgramGroupDesc& self ) 
            { return self.program_group_desc.kind; }, 
            []( pyoptix::ProgramGroupDesc& self, OptixProgramGroupKind kind ) 
            { self.program_group_desc.kind = kind; } 
        )
        .def_property( "flags", 
            []( pyoptix::ProgramGroupDesc& self ) 
            { return self.program_group_desc.flags; }, 
            []( pyoptix::ProgramGroupDesc& self, uint32_t flags ) 
            { self.program_group_desc.flags = flags; } 
        )
        .def_property( "raygenModule", 
            []( pyoptix::ProgramGroupDesc& self ) 
            { return pyoptix::Module{ self.program_group_desc.raygen.module }; }, 
            []( pyoptix::ProgramGroupDesc& self, const pyoptix::Module& module ) 
            { self.program_group_desc.raygen.module = module.module; } 
        )
        .def_readwrite( "raygenEntryFunctionName", &pyoptix::ProgramGroupDesc::entryFunctionName0 )
        /*
        .def_readwrite( "flags", &pyoptix::ProgramGroupDesc::flags )
        .def_property( "raygen", 
                //[](OptixProgramGroupDesc& self) { return pyoptix::ProgramGroupSingleModule{ "", self.raygen }; }, 
                [](OptixProgramGroupDesc& self) { return  self.raygen; }, 
                nullptr
                )
        .def_readwrite( "miss", &OptixProgramGroupDesc::miss )
        .def_readwrite( "exception", &OptixProgramGroupDesc::exception )
        .def_readwrite( "callables", &OptixProgramGroupDesc::callables )
        .def_readwrite( "hitgroup", &OptixProgramGroupDesc::hitgroup )
        */
        ;

    py::class_<OptixProgramGroupOptions>(m, "ProgramGroupOptions")
        .def( py::init([]() { return std::unique_ptr<OptixProgramGroupOptions>(new OptixProgramGroupOptions{} ); } ) )
        .def_readwrite( "placeholder", &OptixProgramGroupOptions::placeholder )
        ;

    py::class_<pyoptix::PipelineCompileOptions>(m, "PipelineCompileOptions")
        .def( py::init( []() 
            { return std::unique_ptr<pyoptix::PipelineCompileOptions>(new pyoptix::PipelineCompileOptions{} ); } 
        ) )
        .def_property( "usesMotionBlur",
            [](const pyoptix::PipelineCompileOptions& self) 
            { return self.options.usesMotionBlur; },
            [](pyoptix::PipelineCompileOptions& self, bool val) 
            { self.options.usesMotionBlur = val; }
        )
        .def_property( "traversableGraphFlags",
            [](const pyoptix::PipelineCompileOptions& self) 
            { return self.options.traversableGraphFlags; },
            [](pyoptix::PipelineCompileOptions& self, OptixTraversableGraphFlags val) 
            { self.options.traversableGraphFlags = val; }
        )
        .def_property( "numPayloadValues",
            [](const pyoptix::PipelineCompileOptions& self) 
            { return self.options.numPayloadValues; },
            [](pyoptix::PipelineCompileOptions& self, int val) 
            { self.options.numPayloadValues = val; }
        )
        .def_property( "numAttributeValues",
            [](const pyoptix::PipelineCompileOptions& self) 
            { return self.options.numAttributeValues; },
            [](pyoptix::PipelineCompileOptions& self, int val) 
            { self.options.numAttributeValues = val; }
        )
        .def_property( "exceptionFlags",
            [](const pyoptix::PipelineCompileOptions& self) 
            { return self.options.exceptionFlags; },
            [](pyoptix::PipelineCompileOptions& self, OptixExceptionFlags val) 
            { self.options.exceptionFlags = val; }
        )
        .def_readwrite( 
            "pipelineLaunchParamsVariableName", 
            &pyoptix::PipelineCompileOptions::pipelineLaunchParamsVariableName 
        )
        .def_property( "usesPrimitiveTypeFlags",
            [](const pyoptix::PipelineCompileOptions& self) 
            { return self.options.usesPrimitiveTypeFlags; },
            [](pyoptix::PipelineCompileOptions& self, bool val) 
            { self.options.usesPrimitiveTypeFlags = val; }
        )
        ;

    py::class_<OptixPipelineLinkOptions>(m, "PipelineLinkOptions")
        .def( py::init([]() { return std::unique_ptr<OptixPipelineLinkOptions>(new OptixPipelineLinkOptions{} ); } ) )
        .def_readwrite( "maxTraceDepth", &OptixPipelineLinkOptions::maxTraceDepth )
        .def_readwrite( "debugLevel", &OptixPipelineLinkOptions::debugLevel )
        ;

    py::class_<OptixShaderBindingTable>(m, "ShaderBindingTable")
        .def( py::init([]() { return std::unique_ptr<OptixShaderBindingTable>(new OptixShaderBindingTable{} ); } ) )
        .def_readwrite( "raygenRecord", &OptixShaderBindingTable::raygenRecord )
        .def_readwrite( "exceptionRecord", &OptixShaderBindingTable::exceptionRecord )
        .def_readwrite( "missRecordBase", &OptixShaderBindingTable::missRecordBase )
        .def_readwrite( "missRecordStrideInBytes", &OptixShaderBindingTable::missRecordStrideInBytes )
        .def_readwrite( "missRecordCount", &OptixShaderBindingTable::missRecordCount )
        .def_readwrite( "hitgroupRecordBase", &OptixShaderBindingTable::hitgroupRecordBase )
        .def_readwrite( "hitgroupRecordStrideInBytes", &OptixShaderBindingTable::hitgroupRecordStrideInBytes )
        .def_readwrite( "hitgroupRecordCount", &OptixShaderBindingTable::hitgroupRecordCount )
        .def_readwrite( "callablesRecordBase", &OptixShaderBindingTable::callablesRecordBase )
        .def_readwrite( "callablesRecordStrideInBytes", &OptixShaderBindingTable::callablesRecordStrideInBytes )
        .def_readwrite( "callablesRecordCount", &OptixShaderBindingTable::callablesRecordCount )
        ;

    py::class_<OptixStackSizes>(m, "StackSizes")
        .def( py::init( []() 
            { return std::unique_ptr<OptixStackSizes>(new OptixStackSizes{} ); }
        ) )
        .def_readwrite( "cssRG", &OptixStackSizes::cssRG )
        .def_readwrite( "cssMS", &OptixStackSizes::cssMS )
        .def_readwrite( "cssCH", &OptixStackSizes::cssCH )
        .def_readwrite( "cssAH", &OptixStackSizes::cssAH )
        .def_readwrite( "cssIS", &OptixStackSizes::cssIS )
        .def_readwrite( "cssCC", &OptixStackSizes::cssCC )
        .def_readwrite( "dssDC", &OptixStackSizes::dssDC )
        ;

    py::class_<OptixBuiltinISOptions>(m, "BuiltinISOptions")
        .def( py::init([]() { return std::unique_ptr<OptixBuiltinISOptions>(new OptixBuiltinISOptions{} ); } ) )
        .def_readwrite( "builtinISModuleType", &OptixBuiltinISOptions::builtinISModuleType )
        .def_readwrite( "usesMotionBlur", &OptixBuiltinISOptions::usesMotionBlur )
        ;


    //---------------------------------------------------------------------------
    //
    // Opaque types
    //
    //---------------------------------------------------------------------------
    
    py::class_<pyoptix::DeviceContext>( m, "DeviceContext" )
        .def( "destroy", &pyoptix::deviceContextDestroy )
        .def( "getProperty", &pyoptix::deviceContextGetProperty )
        .def( "setLogCallback", &pyoptix::deviceContextSetLogCallback )
        .def( "setCacheEnabled", &pyoptix::deviceContextSetCacheEnabled )
        .def( "setCacheLocation", &pyoptix::deviceContextSetCacheLocation )
        .def( "setCacheDatabaseSizes", &pyoptix::deviceContextSetCacheDatabaseSizes )
        .def( "getCacheEnabled", &pyoptix::deviceContextGetCacheEnabled )
        .def( "getCacheLocation", &pyoptix::deviceContextGetCacheLocation )
        .def( "getCacheDatabaseSizes", &pyoptix::deviceContextGetCacheDatabaseSizes )
        .def( "pipelineCreate", &pyoptix::pipelineCreate )
        .def( "moduleCreateFromPTX", &pyoptix::moduleCreateFromPTX )
        .def( "moduleBuiltinISGet", &pyoptix::builtinISModuleGet )
        .def( "programGroupCreate", &pyoptix::programGroupCreate )
        /*
        .def( "accelComputeMemoryUsage", &pyoptix::accelComputeMemoryUsage )
        .def( "accelBuild", &pyoptix::accelBuild )
        .def( "accelGetRelocationInfo", &pyoptix::accelGetRelocationInfo )
        .def( "accelCheckRelocationCompatibility", &pyoptix::accelCheckRelocationCompatibility )
        .def( "accelRelocate", &pyoptix::accelRelocate )
        .def( "accelCompact", &pyoptix::accelCompact )
        .def( "denoiserCreate", &pyoptix::denoiserCreate )
        */
        ;

    py::class_<pyoptix::Module>( m, "Module" )
        .def( "destroy", &pyoptix::moduleDestroy )
        ;

    py::class_<pyoptix::ProgramGroup>( m, "ProgramGroup" )
        .def( "getStackSize", &pyoptix::programGroupGetStackSize )
        .def( "destroy", &pyoptix::programGroupDestroy )
        ;

    py::class_<pyoptix::Pipeline>( m, "Pipeline" )
        .def( "destroy", &pyoptix::pipelineDestroy )
        .def( "setStackSize", &pyoptix::pipelineSetStackSize )
        ;

    py::class_<pyoptix::Denoiser>( m, "Denoiser" )
        .def( "setModel", &pyoptix::denoiserSetModel )
        .def( "destroy", &pyoptix::denoiserDestroy )
        .def( "computeMemoryResources", &pyoptix::denoiserComputeMemoryResources )
        .def( "setup", &pyoptix::denoiserSetup )
        .def( "invoke", &pyoptix::denoiserInvoke )
        .def( "computeIntensity", &pyoptix::denoiserComputeIntensity )
        .def( "computeAverageColor", &pyoptix::denoiserComputeAverageColor )
        ;

}

