#!/usr/bin/env python

import os
import re 
import sys
import string

lexical_scopes = [
    'Context',
    'Program',
    'Selector',
    'GeometryInstance',
    'Geometry',
    'Material'
    ]


rt_types = [
    'Variable',     # Needs to be before lexical scopes
    'Acceleration',
    'Buffer',
    'GeometryGroup',
    'GeometryInstance',
    'Geometry',
    'Group',
    'Material',
    'Program',
    'RemoteDevice',
    'Selector',
    'TextureSampler',
    'Transform',
    'Context',      # needs to be last since it acts as factory
    ]

custom_funcs = {
        'Variable'         : [ 'setUint', 'setInt', 'setFloat' ],
        'Acceleration'     : [],
        'Buffer'           : [],
        'GeometryGroup'    : [],
        'GeometryInstance' : [],
        'Geometry'         : [],
        'Group'            : [],
        'Material'         : [],
        'Program'          : [],
        'RemoteDevice'     : [],
        'Selector'         : [],
        'TextureSampler'   : [],
        'Transform'        : [],
        'Context'          : [ 'createAcceleration',
                               'createBuffer',
                               'createGeometry',
                               'createGeometryGroup',
                               'createGeometryInstance',
                               'createGroup',
                               'createMaterial',
                               'createProgramFromPTXFile',
                               'createTextureSampler',
                               'createTransform',
                               ],
        'optix'            : [ 'createContext' ],
        }



################################################################################
#
# Template strings
#
################################################################################

type_reg_template = string.Template( '''
  if( PyType_Ready( &${rt_type}Type ) < 0 )
    return NULL;
  Py_INCREF( &${rt_type}Type );
  PyModule_AddObject(
      mod,
      "${rt_type}",
      (PyObject*)( &${rt_type}Type )
      );
''')


type_declare_template = string.Template( '''
typedef struct 
{
    PyObject_HEAD
    ${opaque_type} p;
} ${rt_type};

static PyTypeObject ${rt_type}Type;

static PyObject* ${rt_type}New( void* p )
{
  ${rt_type}* self = (${rt_type}*)PyObject_New( ${rt_type}, &${rt_type}Type );
  if( !self )
    return 0;
  
  self->p = (${opaque_type})p;
  return (PyObject*)self;
}


''')


type_template = string.Template( '''
/******************************************************************************\
 *
 * ${rt_type} object
 *
\******************************************************************************/

static void ${rt_type}_dealloc( ${rt_type}* self )
{
  Py_TYPE( self )->tp_free((PyObject*)self);
}


${rt_type_methods}

${rt_type_getitem}

static PyTypeObject ${rt_type}Type =
{
    PyVarObject_HEAD_INIT(NULL, 0)
    "optix.${rt_type}",        /*tp_name*/
    sizeof(${rt_type}),        /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)${rt_type}_dealloc,  /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    ${getitem_meth},                /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT,        /*tp_flags*/
    "${rt_type} objects",      /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    ${rt_type}_methods,        /* tp_methods */
    0,                         /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    0,                         /* tp_init */
    0,                         /* tp_alloc */
    PyType_GenericNew,         /* tp_new */
};
''')


enum_template = string.Template( '''\
  v = PyLong_FromLong( RT_${enum_name} );
  PyObject_SetAttrString( mod, "${enum_name}", v );
  Py_DECREF(v);
''')



methods_struct_template = string.Template( '''
static PyMethodDef ${rt_type}_methods[] =
{
  ${method_registrations}
  {NULL}  /* Sentinel */
};
''')


method_registration_template = string.Template('''
  {
    "${method_name}",
    (PyCFunction)${rt_type}_${method_name},
    METH_VARARGS | METH_KEYWORDS,
    "${method_name}"
  },
''')


arg_parse_template = string.Template('''
  if( !PyArg_ParseTupleAndKeywords( args, kwds, "${format_string}", kwlist${parse_args} ) )
    return NULL;
''')


type_method_def_template = string.Template('''
static PyObject* ${rt_type}_${method_name}( ${rt_type}* self, PyObject* args, PyObject* kwds )
{
  if( !self->p )                                                                 
  {                                                                                 
    PyErr_SetString( PyExc_RuntimeError, "${rt_type}.${method_name}() called on uninitialized object" );
    return 0;                                                                    
  }                                                                              

${arg_parsing}
                                                                                 
  RTcontext ctx;                                                               
  rt${rt_type}GetContext( self->p, &ctx );                                          
  CHECK_RT_RESULT( ${optix_func_name}( ${args} ), ctx, "${rt_type}.${method_name}" );

  return Py_BuildValue( ${ret_args} );
}
''')


type_method_def_context_template = string.Template('''
static PyObject* ${rt_type}_${method_name}( ${rt_type}* self, PyObject* args, PyObject* kwds )
{
  if( !self->p )                                                                 
  {                                                                                 
    PyErr_SetString( PyExc_RuntimeError, "${rt_type}.${method_name}() called on uninitialized object" );
    return 0;                                                                    
  }                                                                              

${arg_parsing}
                                                                                 
  CHECK_RT_RESULT( ${optix_func_name}( ${args} ), self->p, "${rt_type}.${method_name}" );

  return Py_BuildValue( ${ret_args} );
}
''')


type_method_def_remotedevice_template= string.Template('''
static PyObject* ${rt_type}_${method_name}( ${rt_type}* self, PyObject* args, PyObject* kwds )
{
  if( !self->p )                                                                 
  {                                                                                 
    PyErr_SetString( PyExc_RuntimeError, "${rt_type}.${method_name}() called on uninitialized object" );
    return 0;                                                                    
  }                                                                              

${arg_parsing}
                                                                                 
  CHECK_RT_RESULT( ${optix_func_name}( ${args} ), 0, "${rt_type}.${method_name}" );

  return Py_BuildValue( ${ret_args} );
}
''')


type_method_def_no_res_template = string.Template('''
static PyObject* ${rt_type}_${method_name}( ${rt_type}* self, PyObject* args, PyObject* kwds )
{
  if( !self->p )                                                                 
  {                                                                                 
    PyErr_SetString( PyExc_RuntimeError, "${rt_type}.${method_name}() called on uninitialized object" );
    return 0;                                                                    
  }                                                                              

${arg_parsing}
                                                                                 
  ${optix_func_name}( ${args} );

  return Py_BuildValue( ${ret_args} );
}
''')

method_def= string.Template('''
static PyObject* optix_${method_name}( PyObject* self, PyObject* args, PyObject* kwds )
{
${arg_parsing}
                                                                                 
  ${optix_func_name}( ${args} );

  return Py_BuildValue( ${ret_args} );
}
''')


decls_file_template = string.Template( '''
/*
 *
 */

#include <Python.h>
#include <optix_host.h>


${type_decls}

''' )


getitem_template = string.Template( '''
static PyObject* ${rt_type}GetItem( PyObject* self, PyObject* key )
{
  if( !PyUnicode_Check( key ) ) 
  {
    PyErr_SetString( PyExc_TypeError, "${rt_type}.getItem() called with non-string key" );
    return 0;
  }

  PyObject * temp_bytes = PyUnicode_AsEncodedString( key, "ASCII", "strict" );
  const char* str = NULL;
  if (temp_bytes != NULL)
  {
    str = PyBytes_AS_STRING( temp_bytes );
    str = strdup( str );
    Py_DECREF( temp_bytes );
  } else {
    PyErr_SetString( PyExc_TypeError, "${rt_type}.getItem() called with non-askii key" );
    return 0;
  }

  ${opaque_type} p = ( (${rt_type}*)self )->p;

  RTvariable v;
  CHECK_RT_RESULT( rt${rt_type}QueryVariable( p, str, &v ), 0, "${rt_type}.__getitem__" );
  if( !v )
    CHECK_RT_RESULT( rt${rt_type}DeclareVariable( p, str, &v ), 0, "${rt_type}.__getitem__" );

  return Py_BuildValue( "O&", VariableNew, v );
}

static PyMappingMethods ${rt_type}MappingMethods = { 0, ${rt_type}GetItem, 0 };
''')


module_file_template = string.Template( '''
/*
 *
 */

#include "PyOptiXUtil.h" 
#include <stdio.h>


${types}

${module_methods}

static struct PyModuleDef module_def =
{
  PyModuleDef_HEAD_INIT,
  "optix",
  NULL,
  -1,
  optix_methods,
  NULL,
  NULL,
  NULL,
  NULL
};

PyObject* PyInit_optix()
{
  Py_Initialize();
  import_array();

  PyObject* mod = PyModule_Create( &module_def );

${type_registrations}

  PyObject* v = 0;

${enum_registrations}

  return mod;
}
        
''')

################################################################################
#
# Helpers
#
################################################################################

def get_opaque_type( rt_type ):
    return 'RT' + rt_type.lower()


def get_rt_prefix( rt_type ):
    return 'rt' + rt_type


def is_create_method( func ):
    return re.match( 'rt[A-Z][a-z]+[A-Za-z]*Create', func ) != None


def strip_rt_prefix( rt_type, rt_func ):
    return rt_func[ len( get_rt_prefix( rt_type ) ) : ]


def rt_funcname_to_methodname( rt_type, rt_func ):
    if rt_type and rt_func.startswith( get_rt_prefix( rt_type ) ):
        name = strip_rt_prefix( rt_type, rt_func )
        return str( name[0] ).lower() + name[1:]
    else:
        name = rt_func[2:]
        return str( name[0] ).lower() + name[1:]


def parseFunctionPrototype( line ):
    line = line.strip()
    line = line.replace( 'RTAPI', '' )

    front, back = line.split( '(' )
    back = back.split( ')' )[0]

    ret, funcname = front.split()

    rt_type = None
    for t in rt_types:
        tt = 'rt'+t
        if tt in funcname:
            rt_type = t
            break

    if funcname == 'rtRemoteDeviceCreate' or funcname == 'rtContextCreate':
        rt_type = None

    params = back.split( ',' )
    params = [ x.split() for x in params ]
    params = [ ( ' '.join( x[0:-1] ), x[-1]) if x else x for x in params ]

    return ( rt_type, ret, funcname, params )


def parse_enums():
    enum_list = []

    with open( os.path.join( sys.argv[1], 'internal', 'optix_declarations.h' ) ) as optix_decls:

        optix_decls_string = optix_decls.read()

        enums_re     = re.compile( 'typedef\s+enum\s+{[^\}]+}\s+\w+;', re.DOTALL )
        enum_val_re  = re.compile( '^\s+RT_(\w+)[^,}]*[,}]', re.DOTALL|re.MULTILINE)
        enum_name_re = re.compile( '\}\s+(RT[a-z]+)', re.DOTALL | re.MULTILINE )

        enums = enums_re.findall( optix_decls_string )
        for enum in enums:
            enum_name = enum_name_re.search( enum ).group(1)
            enum_vals = enum_val_re.findall( enum )
            enum_list.append( ( enum_name, enum_vals ) )

    return enum_list


def parse_funcs():
    funcs  = {
            'Acceleration' : [],
            'Buffer' : [],
            'Context' : [],
            'Geometry' : [],
            'GeometryGroup' : [],
            'GeometryInstance' : [],
            'Group' : [],
            'Material' : [],
            'Program' : [],
            'RemoteDevice' : [],
            'Selector' : [],
            'TextureSampler' : [],
            'Transform' : [],
            'Variable' : [],
            None : [],
            }

    with open( os.path.join( sys.argv[1], 'optix_host.h' ) ) as  optix_header:
        ignore_re = re.compile( 'rtVariableSet[1-4][a-z][a-z]?v|rtVariableGet[1-4][a-z][a-z]?v' )
        for line in optix_header:
            line = line.strip()
            if line and line[0] == '#':
                continue
            tokens = line.split()
            if len( tokens ) >= 3 and tokens[1] == 'RTAPI':
                (rt_object, ret, funcname, params ) = parseFunctionPrototype( line )
                if ignore_re.match( funcname ):
                    #print '>>>>>> ignoring {}'.format( funcname )
                    continue

                funcs[ rt_object ].append( (ret, funcname, params ) )

    return funcs



                                    # ( param_type, param_format_str, param_init,    arg_decorator, parse_arg_decorator )
C_to_Py = {
        'float'                     : ( 'float'             , 'f' ,  '=0.0f'       , '{}'       ,  '&{}'                         ),
        'float*'                    : ( 'float'             , 'f' ,  '=0.0f'       , '&{}'      ,  '{}'                        ),
        'const float*'              : ( 'const float'       , 'O' ,  '[16]={0.0f}' , '{}'       ,  '{}'                         ),
        #'const float**'             : ( 'const float*'      , ''  ,  '[16]={0.0f}' , '&{}'      ,  '&{}'                        ),
        'double'                    : ( 'double'            , 'd' ,  '=0.0'        , '{}'       ,  '&{}'                         ),
        'double*'                   : ( 'double'            , 'd' ,  '=0.0'        , '&{}'      ,  '{}'                        ),
        'int'                       : ( 'int'               , 'i' ,  '=0'          , '{}'       ,  '&{}'                         ),
        'int*'                      : ( 'int'               , 'i' ,  '=0'          , '&{}'      ,  '{}'                        ),
        'const int*'                : ( 'int'               , 'i' ,  '[16]={0}'    , '{}'       ,  '{}'                         ),
        'unsigned int'              : ( 'unsigned int'      , 'I' ,  '=0u'         , '{}'       ,  '&{}'                         ),
        'unsigned int*'             : ( 'unsigned int'      , 'I' ,  '=0u'         , '&{}'      ,  '{}'                         ),
        'const char*'               : ( 'const char*'       , 's' ,  '=0'          , '{}'       ,  '&{}'                         ),
        'const char**'              : ( 'const char*'       , 's' ,  '=0'          , '&{}'      ,  '{}'                        ),
        'void*'                     : ( 'void*'             , 'i' ,  '=0'          , '{}'       ,  '{}'                         ),
        'const void*'               : ( 'const char'        , 'O' ,  '[1024]={0}'  , '{}'       ,  '{}'                         ),
        'void**'                    : ( 'void*'             , 'O' ,  '=0'          , '&{}'      ,  'createNumpyArray( self->p, {} )' ),
        'RTsize'                    : ( 'RTsize'            , 'n' ,  '=0'          , '{}'       ,  '&{}'                         ),
        'RTsize*'                   : ( 'RTsize'            , ''  ,  '=0'          , '&{}'      ,  '{}'                        ),
        'const RTsize*'             : ( 'RTsize'            , ''  ,  '[3]={0ull}'  , '{}'       ,  '{}'                         ),
        'RTtimeoutcallback'         : ( 'RTtimeoutcallback' , 'O' ,  '=0'          , '{}'       ,  '{}'                         ),
        'RTformat'                  : ( 'int'               , 'i' ,  '=0'          , '{}'       ,  '&{}'                         ),
        'RTformat*'                 : ( 'RTformat'          , ''  ,  '=0'          , '&{}'      ,  '{}'                        ),
        'RTobjecttype'              : ( 'int'               , 'i' ,  '=0'          , '{}'       ,  '&{}'                         ),
        'RTobjecttype*'             : ( 'RTobjecttype'      , ''  ,  '=0'          , '&{}'      ,  '{}'                        ),
        'RTwrapmode'                : ( 'int'               , 'i' ,  '=0'          , '{}'       ,  '&{}'                         ),
        'RTwrapmode*'               : ( 'RTwrapmode'        , ''  ,  '=0'          , '&{}'      ,  '{}'                        ),
        'RTfiltermode'              : ( 'int'               , 'i' ,  '=0'          , '{}'       ,  '&{}'                         ),
        'RTfiltermode*'             : ( 'RTfiltermode'      , ''  ,  '=0'          , '&{}'      ,  '{}'                        ),
        'RTtexturereadmode'         : ( 'int'               , 'i' ,  '=0'          , '{}'       ,  '&{}'                         ),
        'RTtexturereadmode*'        : ( 'RTtexturereadmode' , ''  ,  '=0'          , '&{}'      ,  '{}'                        ),
        'RTtextureindexmode'        : ( 'int'               , 'i' ,  '=0'          , '{}'       ,  '&{}'                         ),
        'RTtextureindexmode*'       : ( 'RTtextureindexmode', ''  ,  '=0'          , '&{}'      ,  '{}'                        ),
        'RTexception'               : ( 'int'               , 'i' ,  '=0'          , '{}'       ,  '&{}'                         ),
        'RTresult'                  : ( 'int'               , 'i' ,  '=0'          , '{}'       ,  '&{}'                         ),
        'RTdeviceattribute'         : ( 'int'               , 'i' ,  '=0'          , '{}'       ,  '&{}'                         ),
        'RTremotedeviceattribute'   : ( 'int'               , 'i' ,  '=0'          , '{}'       ,  '&{}'                         ),
        'RTremotedevicestatus'      : ( 'int'               , 'i' ,  '=0'          , '{}'       ,  '&{}'                         ),
        'RTcontextattribute'        : ( 'int'               , 'i' ,  '=0'          , '{}'       ,  '&{}'                         ),
        'RTbufferattribute'         : ( 'int'               , 'i' ,  '=0'          , '{}'       ,  '&{}'                         ),
        'RTbufferidnull'            : ( 'int'               , 'i' ,  '=0'          , '{}'       ,  '&{}'                         ),
        'RTprogramidnull'           : ( 'int'               , 'i' ,  '=0'          , '{}'       ,  '&{}'                         ),
        'RTtextureidnull'           : ( 'int'               , 'i' ,  '=0'          , '{}'       ,  '&{}'                         ),
        'RTvariable'                : ( 'Variable*'         , 'O!',  '=0'          , '{}->p'    ,  '&VariableType, &{}'         ),
        'RTvariable*'               : ( 'RTvariable'        , 'O&',  '=0'          , '&{}'      ,  'VariableNew, {}'            ),
        'RTacceleration'            : ( 'Acceleration*'     , 'O!',  '=0'          , '{}->p'    ,  '&AccelerationType, &{}'     ),
        'RTacceleration*'           : ( 'RTacceleration'    , 'O&',  '=0'          , '&{}'      ,  'AccelerationNew, {}'        ),
        'RTbuffer'                  : ( 'Buffer*'           , 'O!',  '=0'          , '{}->p'    ,  '&BufferType, &{}'           ),
        'RTbuffer*'                 : ( 'RTbuffer'          , 'O&',  '=0'          , '&{}'      ,  'BufferNew, {}'              ),
        'RTgeometrygroup'           : ( 'GeometryGroup*'    , 'O!',  '=0'          , '{}->p'    ,  '&GeometryGroupType, &{}'    ),
        'RTgeometrygroup*'          : ( 'RTgeometrygroup'   , 'O&',  '=0'          , '&{}'      ,  'GeometryGroupNew, {}'       ),
        'RTgeometryinstance'        : ( 'GeometryInstance*' , 'O!',  '=0'          , '{}->p'    ,  '&GeometryInstanceType, &{}' ),
        'RTgeometryinstance*'       : ( 'RTgeometryinstance', 'O&',  '=0'          , '&{}'      ,  'GeometryInstanceNew, {}'    ),
        'RTgeometry'                : ( 'Geometry*'         , 'O!',  '=0'          , '{}->p'    ,  '&GeometryType, &{}'         ),
        'RTgeometry*'               : ( 'RTgeometry'        , 'O&',  '=0'          , '&{}'      ,  'GeometryNew, {}'            ),
        'RTgroup'                   : ( 'Group*'            , 'O!',  '=0'          , '{}->p'    ,  '&GroupType, &{}'            ),
        'RTgroup*'                  : ( 'RTgroup'           , 'O&',  '=0'          , '&{}'      ,  'GroupNew, {}'               ),
        'RTmaterial'                : ( 'Material*'         , 'O!',  '=0'          , '{}->p'    ,  '&MaterialType, &{}'         ),
        'RTmaterial*'               : ( 'RTmaterial'        , 'O&',  '=0'          , '&{}'      ,  'MaterialNew, {}'            ),
        'RTprogram'                 : ( 'Program*'          , 'O!',  '=0'          , '{}->p'    ,  '&ProgramType, &{}'          ),
        'RTprogram*'                : ( 'RTprogram'         , 'O&',  '=0'          , '&{}'      ,  'ProgramNew, {}'             ),
        'RTobject'                  : ( 'Program*'          , 'O' ,  '=0'          , '{}->p'    ,  '&{}'                        ),
        'RTobject*'                 : ( 'RTobject'          , 'O&',  '=0'          , '&{}'      ,  '{}'                         ),
        'RTremotedevice'            : ( 'RemoteDevice*'     , 'O!',  '=0'          , '{}->p'    ,  '&RemoteDeviceType, &{}'     ),
        'RTremotedevice*'           : ( 'RTremotedevice'    , 'O&',  '=0'          , '&{}'      ,  'RemoteDeviceNew, {}'        ),
        'RTselector'                : ( 'Selector*'         , 'O!',  '=0'          , '{}->p'    ,  '&SelectorType, &{}'         ),
        'RTselector*'               : ( 'RTselector'        , 'O&',  '=0'          , '&{}'      ,  'SelectorNew, {}'            ),
        'RTtexturesampler'          : ( 'TextureSampler*'   , 'O!',  '=0'          , '{}->p'    ,  '&TextureSamplerType, &{}'   ),
        'RTtexturesampler*'         : ( 'RTtexturesampler'  , 'O&',  '=0'          , '&{}'      ,  'TextureSamplerNew, {}'      ),
        'RTtransform'               : ( 'Transform*'        , 'O!',  '=0'          , '{}->p'    ,  '&TransformType, &{}'        ),
        'RTtransform*'              : ( 'RTtransform'       , 'O&',  '=0'          , '&{}'      ,  'TransformNew, {}'           ),
        'RTcontext'                 : ( 'Context*'          , 'O!',  '=0'          , '{}->p'    ,  '&ContextType, &{}'          ),
        'RTcontext*'                : ( 'RTcontext'         , 'O&',  '=0'          , '&{}'      ,  'ContextNew, {}'             ),
        }


def is_output_param( param ):
    ptype = param[0]
    if ptype == 'char*' or ptype == 'const char*' or ptype == 'const float*' or ptype == 'const void*':
        return False
    return ptype[-1] == '*'


def do_error_checking( rt_type, method_name ):
    #if rt_type == 'Context':
    #    return False
    if rt_type == 'RemoteDevice':
        return False
    return True


def create_method_code( rt_type, method_name, func ):
    ( ret, funcname, params ) = func 

    format_string = ''
    ret_format_string = ''
    arg_decls     = ''
    parse_args    = [] 
    args = [ 'self->p' ] if rt_type else [] 
    rets          = []
    kws        = []

    #print params
    #print params[1 if rt_type else 0:]
    #for param in params:
    for param in params[1 if rt_type else 0:]:
        if param[0] not in C_to_Py:
            print( '*************NOT FOUND \'{}\''.format( param[0] ), file=sys.stderr )

        ( param_type, param_format_str, param_init, arg_decorator, parse_arg_decorator ) = C_to_Py[ param[0] ]
        arg_decls += '  {} {}{};\n'.format( param_type, param[1], param_init )
        args.append( arg_decorator.format( param[1] ) )
        if is_output_param( param ):
            #rets.append( param[1] )
            rets.append( parse_arg_decorator.format( param[1] ) )
            ret_format_string += param_format_str 
        else:
            parse_args.append( parse_arg_decorator.format( param[1] ) )
            kws.append( '"{}"'.format( param[1] ) )
            format_string += param_format_str

    args = ', '.join( args )
    parse_args = ', '.join( parse_args )
    if format_string:
        format_string += ':{}.{}'.format( rt_type, funcname )

    kws.append( '0' );
    kwlist = '\n  static char* kwlist[] = {{ {} }};\n'.format( ', '.join( kws ) )
    arg_parse = arg_parse_template.substitute(
            format_string = format_string,
            parse_args    = ', ' + parse_args if parse_args else ''
            )
    arg_parsing = arg_decls + kwlist + arg_parse

    if not rets:
        ret_args = '""'
    else:
        ret_args = '"{}"{}'.format( ret_format_string, '' if not rets else ', ' + ', '.join( rets ) ) 
        rets = '  ;'.format( rets[0] )

    if not rt_type:
        return method_def.substitute( 
                rt_type         = rt_type,
                method_name     = method_name,
                optix_func_name = funcname,
                arg_parsing     = arg_parsing,
                args            = args,
                ret_args        = ret_args
                )
    elif ret == 'void':
        return type_method_def_no_res_template.substitute( 
                rt_type         = rt_type,
                method_name     = method_name,
                optix_func_name = funcname,
                arg_parsing     = arg_parsing,
                args            = args,
                ret_args        = ret_args
                )
    if rt_type == 'Context':
        return type_method_def_context_template.substitute( 
                rt_type         = rt_type,
                method_name     = method_name,
                optix_func_name = funcname,
                arg_parsing     = arg_parsing,
                args            = args,
                ret_args        = ret_args
                )
    if rt_type == 'RemoteDevice':
        return type_method_def_remotedevice_template.substitute( 
                rt_type         = rt_type,
                method_name     = method_name,
                optix_func_name = funcname,
                arg_parsing     = arg_parsing,
                args            = args,
                ret_args        = ret_args
                )
    else: 
        return type_method_def_template.substitute( 
                rt_type         = rt_type,
                method_name     = method_name,
                optix_func_name = funcname,
                arg_parsing     = arg_parsing,
                args            = args,
                ret_args        = ret_args
                )


def create_mod_method( func ):
    method_name = rt_funcname_to_methodname( None, func[1] )
    method_registration = method_registration_template.substitute( 
            rt_type='optix',
            method_name=method_name
            )
    method = create_method_code( None, method_name, func )
    return ( method_registration, method )


def create_type_method( rt_type, func ):
    
    method_name = rt_funcname_to_methodname( rt_type, func[1] )
    method_registration = method_registration_template.substitute( 
            rt_type=rt_type,
            method_name=method_name
            )
    method = create_method_code( rt_type, method_name, func )
    return ( method_registration, method )


def get_type_creaters( rt_type, func_decls, context_func_decls ):
    for func in func_decls[:]:
        if is_create_method( func[1] ):
            func_decls.remove( func )

            new_funcname = func[1] 
            #new_funcname = 'rtContextCreate{}{}'.format(
            #        rt_type,
            #        func[1][ len( get_rt_prefix( rt_type )+'Create' ): ]
            #        )
            context_func_decls.append( (func[0], new_funcname, func[2] ) )


def create_mod_methods( func_decls ):
    type_methods = ''
    method_registrations = ''
    for func in func_decls:
        ( method_registration, method ) = create_mod_method( func )
        type_methods += method
        method_registrations += method_registration

    for custom_func in custom_funcs[ 'optix' ]:
        method_registrations += method_registration_template.substitute( 
            rt_type='optix',
            method_name=custom_func
            )

    #print type_methods
    #print method_registrations

    type_methods += methods_struct_template.substitute( 
        rt_type='optix', 
        method_registrations = method_registrations 
        )
    return type_methods


def create_type_methods( rt_type, func_decls, context_func_decls ):
    if rt_type != 'Context' :
        get_type_creaters( rt_type, func_decls, context_func_decls)
        
    type_methods = ''
    method_registrations = ''
    for func in func_decls:
        ( method_registration, method ) = create_type_method( rt_type, func )
        type_methods += method
        method_registrations += method_registration
    
    for custom_func in custom_funcs[ rt_type ]:
        method_registrations += method_registration_template.substitute( 
            rt_type=rt_type,
            method_name=custom_func
            )


    type_methods += methods_struct_template.substitute( 
        rt_type=rt_type, 
        method_registrations = method_registrations 
        )
    return type_methods


def create_types( funcs ):    
    type_decls  = ''
    for rt_type in rt_types:
        type_decls += type_declare_template.substitute(  
                rt_type         = rt_type,
                opaque_type     = get_opaque_type( rt_type ),
                )

    type_defs   = ''
    for rt_type in rt_types:
        rt_type_methods = create_type_methods( rt_type, funcs[rt_type], funcs['Context'])
        rt_type_getitem = getitem_template.substitute( rt_type = rt_type, opaque_type = get_opaque_type( rt_type ) ) if rt_type in lexical_scopes else ''
        type_defs += type_template.substitute( 
                rt_type         = rt_type,
                rt_type_getitem = rt_type_getitem,
                getitem_meth    = '&{}MappingMethods'.format( rt_type ) if rt_type in lexical_scopes else '0',
                opaque_type     = get_opaque_type( rt_type ),
                rt_type_methods = rt_type_methods 
                )
    return ( type_decls, type_defs )


def create_type_registrations():
    type_registrations = ''
    for rt_type in rt_types:
        type_registrations += type_reg_template.substitute( rt_type=rt_type )
    return type_registrations


def create_module_methods( funcs ):
    mod_methods = create_mod_methods( funcs )
    return mod_methods 


def create_enum_registrations( enums ):
    enum_registrations = ''
    for enum in enums:
        enum_registrations += '/*\n *\n enum {}\n */'.format( enum[0] )
        for enum_name in enum[1]:
            enum_registrations += enum_template.substitute( enum_name = enum_name )
    return enum_registrations


###############################################################################
#
# main
#
###############################################################################

enums = parse_enums()
funcs = parse_funcs()


print( '\n*********** Generating optix module ***********\n', file=sys.stderr )

( type_decls, type_defs ) = create_types( funcs )

with open( 'PyOptiXDecls.h', 'w' ) as decls_file:
    print( decls_file_template.substitute( type_decls=type_decls ),
           file=decls_file )

with open( 'PyOptiXModule.c', 'w' ) as module_file:
    print( module_file_template.substitute(
        types              = type_defs,
        module_methods     = create_module_methods( funcs[ None ] ),
        type_registrations = create_type_registrations(),
        enum_registrations = create_enum_registrations( enums )
        ),
        file=module_file )

