#include <pybind11/pybind11.h>
#include <optix.h>

int add(int i, int j) {
    return i + j;
}

PYBIND11_MAKE_OPAQUE( OptixDeviceContext );

namespace py = pybind11;

PYBIND11_MODULE(optix, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: optix 

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    py::class_<OptixDeviceContext>(m, "DeviceContext")
    /*
        .def( 
            py::init( 
                [](CUcontext fromContext, const OptixDeviceContextOptions* options) 
                { 
                    OptixDeviceContext ctx; 
                    optixDeviceContextCreate( fromContext, options, &ctx ); 
                    return ctx;
                }
            )
        )
        */
        .def("destroy", [](OptixDeviceContext ctx) { optixDeviceContextDestroy(  ctx ); } )
        ;

    m.def("add", &add, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");

    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
