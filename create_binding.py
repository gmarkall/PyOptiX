

import sys
import os
import re



main_template = '''
#include <pybind11/pybind11.h>

#include <optix.h>
//#include "pyoptix.h"

namespace py = pybind11;
    
namespace pyoptix
{{
// Opaque type struct wrappers
{opaque_types_structs}
}} // end namespace pyoptix

PYBIND11_MODULE( optix, m ) 
{{
    m.doc() = R"pbdoc(
        OptiX API 
        -----------------------

        .. currentmodule:: optix

        .. autosummary::
           :toctree: _generate

    )pbdoc";

    // Opaque types
    {opaque_types_bindings}
}}
'''


#-------------------------------------------------------------------------------
#
# Parse the optix_7_types header for data types 
#
#-------------------------------------------------------------------------------

#
# Outputs of type parsing phase
#
opaque_types = [] # List of API objects (eg OptixDeviceContext)
struct_types = {} # Structs used as inputs to API funcs: typename -> [members]
enum_types   = {} # Enumeration pairs: enum_name -> [enum_val0, enum_val1, ...]


# regex for parsing
re_opaque_type  = re.compile( "^typedef\s+struct\s+[A-Za-z_]+\*\s+([A-Za-z]+)" )

re_typedef_end  = re.compile( "^\s*}\s+([A-Za-z]+)\s*;" )

re_enum_type    = re.compile( "^typedef\s+enum\s+([A-Za-z]+)" )
re_enum_value   = re.compile( "^\s*([0-9A-Za-z_]+)" )

re_struct_type  = re.compile( "^typedef\s+struct\s+([A-Za-z]+)" )
re_struct_value = re.compile( "^\s*([^/]+)\s+([A-Za-z]+);" )



if len( sys.argv ) != 2:
    print( "Usage: {} <path/to/optix/include>".format( sys.argv[0] ) )
    sys.exit(0)

optix_include = sys.argv[1]
print( "Looking for optix headers in '{}'".format( optix_include ) )

types_path = os.path.join( optix_include, 'optix_7_types.h' ) 
with open( types_path, 'r' ) as types_file:
    print( "Found optix types header {} ...".format( types_path ) )

    cur_struct   = None
    cur_enum     = None 

    for line in types_file:

        if cur_enum:

            match = re_typedef_end.match( line )
            if match:
                for v in enum_types[cur_enum]:
                    print( "\t\t{}".format( v ) )
                cur_enum = None 
            else: 
                match = re_enum_value.match( line )
                if match:
                    enum_types[cur_enum].append( match.groups()[0] )
            continue
        
        if cur_struct:

            match = re_typedef_end.match( line )
            if match:
                for v in struct_types[cur_struct]:
                    print( "\t\t{}".format( v ) )
                cur_struct = None 
            else: 
                match = re_struct_value.match( line )
                if match:
                    struct_types[cur_struct].append( match.groups() )
            continue

        match = re_opaque_type.match( line )
        if match:
            opaque_types.append( match.groups()[0] )
            print( "\tFound opaque type: {}".format( match.groups()[0] ) )
            continue
        
        match = re_enum_type.match( line )
        if match:
            cur_enum = match.groups()[0]
            enum_types[ cur_enum ] = []
            print( "\tFound enum type: {}".format( cur_enum ) )
            continue

        match = re_struct_type.match( line )
        if match:
            cur_struct = match.groups()[0] 
            print( "\tFound struct type: {}".format( cur_struct ) )
            struct_types[ cur_struct ] = []
            continue


#-------------------------------------------------------------------------------
#
# Parse optix_7_host header for API functions 
#
#-------------------------------------------------------------------------------


def optix_type_to_pyoptix( typename ):
    return typename.replace( "Optix", "", 1 )

def optix_func_to_pyoptix( funcname ):
    no_ns = funcname.replace( "optix", "", 1 )
    pyoptix_type = ""
    for ot in opaque_types:
        pot = optix_type_to_pyoptix( ot )
        if no_ns.startswith( pot ):
            pyoptix_type = pot
            break
    return pyoptix_type, funcname.replace( pyoptix_type, "", 1 )



re_function = re.compile( "^\s*([^\n]+?)\s+optix([A-Za-z]+)(\(.*?\))", 
        re.MULTILINE | re.DOTALL 
        )

with open( os.path.join( optix_include, 'optix_7_host.h' ), 'r' ) as host_file:
    print( "Found optix host header ..." )
    matches = re_function.findall( host_file.read() ) 
    for m in matches:
        ret_type = m[0]
        typename, funcname = optix_func_to_pyoptix( m[1] )
        print( "'{}' '{}'".format( typename, funcname ) )
        #params = split_params( m[2] ) 
        print( "========================================" )
        print( m )
        print( "========================================" )



#-------------------------------------------------------------------------------
#
#
#
#-------------------------------------------------------------------------------

opaque_type_struct_template = '''
struct {pyoptix_name}
{{
    {optix_name} {var_name};
}};
'''

opaque_type_binding_template = '''
    py::class_<{optix_name}>( m, "{pyoptix_name}" );
'''
opaque_types_structs  = []
opaque_types_bindings = []

for optix_name in opaque_types:
    pyoptix_name = optix_type_to_pyoptix( optix_name )
    var_name     = pyoptix_name[0].lower() + pyoptix_name[1:] 
    opaque_types_structs.append( opaque_type_struct_template.format( 
        optix_name   = optix_name, 
        pyoptix_name = pyoptix_name, 
        var_name     = var_name 
        )
    )
    opaque_types_bindings.append( opaque_type_binding_template.format( 
        optix_name   = optix_name, 
        pyoptix_name = pyoptix_name
        )
    )




main = main_template.format( 
        opaque_types_structs = "".join( opaque_types_structs ),
        opaque_types_bindings = "".join( opaque_types_bindings ) 
        )

with open( "bindings.cpp", "w" ) as outfile:
    print( main, file=outfile )
