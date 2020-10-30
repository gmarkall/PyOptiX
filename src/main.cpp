
#include <pybind11/pybind11.h>

#include <optix.h>
//#include "pyoptix.h"

namespace py = pybind11;
    
namespace pyoptix
{
// Opaque type struct wrappers

struct DeviceContext
{
    OptixDeviceContext deviceContext;
};

struct Module
{
    OptixModule module;
};

struct ProgramGroup
{
    OptixProgramGroup programGroup;
};

struct Pipeline
{
    OptixPipeline pipeline;
};

struct Denoiser
{
    OptixDenoiser denoiser;
};

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

    // Opaque types
    
    py::class_<OptixDeviceContext>( m, "DeviceContext" );

    py::class_<OptixModule>( m, "Module" );

    py::class_<OptixProgramGroup>( m, "ProgramGroup" );

    py::class_<OptixPipeline>( m, "Pipeline" );

    py::class_<OptixDenoiser>( m, "Denoiser" );

}

