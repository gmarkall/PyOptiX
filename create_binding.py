

import sys
import os
import re




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

re_struct_type  = re.compile( "^typedef\s+struct\s+([A-Za-z0-9]+)" )
re_struct_value = re.compile( "^\s*([^/]+)\s+([A-Za-z]+);" )



if len( sys.argv ) != 2:
    print( "Usage: {} <path/to/optix/include>".format( sys.argv[0] ) )
    sys.exit(0)

optix_include = sys.argv[1]
print( "Looking for optix headers in '{}'".format( optix_include ) )

types_path = os.path.join( optix_include, 'optix_7_types.h' ) 
print( "<<<{}>>>".format( types_path ) )
with open( types_path, 'r' ) as types_file:
    print( "Found optix types header {} ...".format( types_path ) )

    cur_struct   = None
    cur_enum     = None 
    optional     = False

    for line in types_file:

        if '#ifdef' in line and 'OPTIONAL' in line:
            optional = True
            continue
        if '#if' in line and 'CUDACC' in line:
            optional = True
            continue
        if optional:
            if '#endif' in line:
                optional = False
            continue

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

#
# Outputs of type parsing phase
#
api_funcs = [] # List of API functions in form 
               # ( ret_type,             eg OptixResuylt
               #   api_type or None,     eg DeviceContext
               #   func_name,            eg GetProperty
               #   [ (parm1_type, parm1_name), (parm2_type, parm2_name) ... ] 
               # )

def optix_type_to_pyoptix( typename ):
    return typename.replace( "Optix", "", 1 )

def optix_funcname_to_pyoptix( funcname ):
    no_ns = funcname.replace( "optix", "", 1 )
    pyoptix_type = ""
    for ot in opaque_types:
        pot = optix_type_to_pyoptix( ot )
        if no_ns.startswith( pot ):
            pyoptix_type = pot
            break
    if no_ns.startswith( 'BuiltinISModule' ):
        pyoptix_type = "Module"
    pyoptix_name = funcname.replace( pyoptix_type, "", 1 )
    pyoptix_name = pyoptix_name[0].lower() + pyoptix_name[1:]
    return pyoptix_type, pyoptix_name 

def optix_func_to_pyoptix( typename, funcname, params ):
    p = []
        
    split_params = [ x.strip() for x in params.split(',') ]

    param_idx = 0
    for x in split_params:
        is_first_param = param_idx == 0
        is_last_param  = param_idx == len( split_params ) - 1
        param_idx += 1

        param_type = " ".join( x.split()[0:-1:1] )
        param_name =  x.split()[-1]
    
        # handle API type 'member' functions
        if typename and is_first_param: 
            print( "looking for '{}' in first: '{}'".format( typename, param_type ) )
            if typename in param_type:
                continue
        
        # make accel funcs DeviceContext member methods
        if not typename and is_first_param and funcname.startswith( 'accel' ):
            typename = 'DeviceContext'
            p        = p[1:] # remove DeviceContext arg
            continue

        # handle api object creation functions which we will make context member functions
        if typename and is_last_param and ( 'create' in funcname or 'Get' in funcname ) and typename+'*' in param_type:
            print( "looking for '{}' in last   '{}'".format( typename, param_type ) )
            if typename == 'DeviceContext':
                typename = '@DeviceContext'
            else:
                funcname = typename[0].lower() + typename [1:] + funcname[0].upper() + funcname[1:]
                typename = 'DeviceContext'
                p        = p[1:] # remove DeviceContext arg
            continue
        

        # special case some functions
        if typename == 'DeviceContext' and funcname == 'getProperty':
            pass
        elif typename == 'DeviceContext' and funcname == 'setLogCallback':
            pass
         
        # TODO
        # special case logString

        p.append( (param_type, param_name ) )
    return ( typename, funcname, p )


api_func_wrappers = [] 
api_func_wrapper_template = '''
void {wrapper_name}( 
{params}
    )
{{
    PYOPTIX_CHECK( 
        {optix_name}(
{optix_args}
        )
    );
}}
'''

re_function = re.compile( "^\s*([^\n]+?)\s+optix([A-Za-z]+)\((.*?)\)", 
        re.MULTILINE | re.DOTALL 
        )

with open( os.path.join( optix_include, 'optix_7_host.h' ), 'r' ) as host_file:
    print( "Found optix host header ..." )
    matches = re_function.findall( host_file.read() ) 
    for m in matches:
        ret_type = m[0]
        typename, funcname = optix_funcname_to_pyoptix( m[1] )
        typename, funcname, params = optix_func_to_pyoptix( typename, funcname, m[2] ) 
        api_funcs.append( ( ret_type, typename, funcname, params ) )
        api_func_wrappers.append( api_func_wrapper_template.format(
            wrapper_name = m[1][0].lower() + m[1][1:],
            params = ",\n".join( [ "       " + x.strip() for x in m[2].split(",") ] ),
            optix_name = 'optix'+m[1],
            optix_args = ",\n".join( [ "            " + x.strip().split()[-1] for x in m[2].split(",") ] )
            ) 
        )
    
for ret, type_, name, params in api_funcs:
    print( "'{}' '{}' '{}'".format( ret, type_, name) )
    for p in params:
        print( "\t'{}'".format( p ) )

#-------------------------------------------------------------------------------
#
# Generate wrappers
#
#-------------------------------------------------------------------------------

main_template = '''
#include <pybind11/pybind11.h>
#include <optix.h>

#include <memory>
#include <stdexcept>


namespace py = pybind11;
    
#define PYOPTIX_CHECK( call )                                                  \\
    do                                                                         \\
    {{                                                                          \\
        OptixResult res = call;                                                \\
        if( res != OPTIX_SUCCESS )                                             \\
            throw std::runtime_error( optixGetErrorString( res )  );           \\
    }} while( 0 )

namespace pyoptix
{{
// Opaque type struct wrappers
{opaque_types_structs}

// Error checking api func wrappers
{api_func_wrappers}

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

    //---------------------------------------------------------------------------
    //
    // Module methods 
    //
    //---------------------------------------------------------------------------
{module_defs}
    
    //---------------------------------------------------------------------------
    //
    // Enumerations 
    //
    //---------------------------------------------------------------------------
{module_enums}
    
    //---------------------------------------------------------------------------
    //
    // Param types
    //
    //---------------------------------------------------------------------------
{param_types}

    //---------------------------------------------------------------------------
    //
    // Opaque types
    //
    //---------------------------------------------------------------------------
    {opaque_types_bindings}
}}
'''


opaque_type_struct_template = '''
struct {pyoptix_name}
{{
    {optix_name} {var_name};
}};
'''

opaque_type_binding_template = '''
    py::class_<{optix_name}>( m, "{pyoptix_name}" )
{defs}        ;
'''

enum_template = '''
    py::enum_<{optix_enum_name}>(m, "{pyoptix_enum_name}")
{enum_values}        .export_values();
'''


param_type_template = '''
    py::class_<{optix_type_name}>(m, "{pyoptix_type_name}")
        .def( py::init([]() {{ return std::unique_ptr<{optix_type_name}>(new {optix_type_name}{{}} ); }} ) )
{read_writes}        ;
'''


opaque_types_structs  = []
opaque_types_bindings = []

module_defs = ""
for d in filter( lambda x : x[1] == "", api_funcs ):
    module_defs += "    m.def( \"{}\", &pyoptix::{} );\n".format( d[2], d[2][0].upper() + d[2][1:] )

#enum_types   = {} # Enumeration pairs: enum_name -> [enum_val0, enum_val1, ...]
module_enums = ""
for optix_enum_name, optix_values in enum_types.items():
    enum_values = ""
    for value in optix_values:
        enum_values += "        .value( \"{}\", {} )\n".format( value.replace( "OPTIX_", "", 1 ), value )
    module_enums += enum_template.format( 
            optix_enum_name = optix_enum_name,
            pyoptix_enum_name = optix_enum_name.replace( "Optix", "", 1 ),
            enum_values = enum_values
            )
    print( "{} {}".format( optix_enum_name, optix_values ) )



#.def_readwrite("name", &Pet::name)
param_types = ""
for optix_type_name, optix_members in struct_types.items():
    read_writes = ""
    for member in optix_members:
        read_writes += "        .def_readwrite( \"{}\", &{}::{} )\n".format( member[1], optix_type_name, member[1] )
    param_types += param_type_template.format(
            optix_type_name = optix_type_name,
            pyoptix_type_name = optix_type_name.replace( "Optix", "", 1 ),
            read_writes = read_writes
            )

for optix_name in opaque_types:
    pyoptix_name = optix_type_to_pyoptix( optix_name )
    var_name     = pyoptix_name[0].lower() + pyoptix_name[1:] 
    opaque_types_structs.append( opaque_type_struct_template.format( 
        optix_name   = optix_name, 
        pyoptix_name = pyoptix_name, 
        var_name     = var_name 
        )
    )
    defs = ""
    for d in filter( lambda x : x[1] == pyoptix_name, api_funcs ):
        if d[1] == 'DeviceContext' and 'Create' in d[2]:
            defs += "        .def( \"{}\", &pyoptix::{} )\n".format( d[2], d[2][0].lower()+d[2][1:] )
        else:
            defs += "        .def( \"{}\", &pyoptix::{}{} )\n".format( d[2], d[1][0].lower()+d[1][1:], d[2][0].upper() + d[2][1:] )

    opaque_types_bindings.append( opaque_type_binding_template.format( 
        optix_name   = optix_name, 
        pyoptix_name = pyoptix_name,
        defs         = defs
        )
    )


main = main_template.format( 
        api_func_wrappers = " ".join( api_func_wrappers ),
        module_defs = module_defs,
        module_enums = module_enums,
        param_types = param_types,
        opaque_types_structs = "".join( opaque_types_structs ),
        opaque_types_bindings = "".join( opaque_types_bindings ) 
        )

with open( "bindings.cpp", "w" ) as outfile:
    print( main, file=outfile )
