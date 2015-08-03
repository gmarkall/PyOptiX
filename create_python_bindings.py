#!/usr/bin/env python

import os
import re 
import sys
import string

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


################################################################################
#
# Template strings
#
################################################################################

type_reg_template = string.Template( '''
  if( PyType_Ready( &${rt_type}Type ) < 0 )
    return;
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
''')

type_template = string.Template( '''
/******************************************************************************\
 *
 * ${rt_type} object
 *
\******************************************************************************/

typedef struct 
{
    PyObject_HEAD
    ${opaque_type} p;
} ${rt_type};


static void ${rt_type}_dealloc( ${rt_type}* self )
{
  self->ob_type->tp_free((PyObject*)self);
}


${rt_type_methods}


static PyTypeObject ${rt_type}Type =
{
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
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
    0,                         /*tp_as_mapping*/
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
  v = PyLong_FromLong( ${enum_name} );
  PyObject_SetAttrString( mod, "${enum_name}", v );
  Py_DECREF(v);
''')


create_template = string.Template( '''
static PyObject *
Context_create_${rt_type}( Context* self, PyObject* args )
{
    ${rt_type}* o = PyObject_New( ${rt_type}, &${rt_type}Type );
    RTresult res = rt${rt_type}Create( self->p, &o->p );
    if( res != RT_SUCCESS )
    {
       PyObject_Del( (PyObject*)o );
       o = 0;
    }
    return (PyObject*)o;
}
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
    METH_VARARGS,
    "${method_name}"
  },
''')

type_method_def_template = string.Template('''
static PyObject* ${rt_type}_${method_name}( ${rt_type}* self, PyObject* args )
{
  if( !self->p )                                                                 
  {                                                                                 
    PyErr_SetString( PyExc_RuntimeError, "${rt_type}.${method_name}() called on uninitialized object" );
    return 0;                                                                    
  }                                                                              
                                                                                 
  RTresult res = ${optix_func_name}( self->p );                                              
  if( res != RT_SUCCESS )                                                        
  {                                                                              
    const char* optix_err_str = 0;                                                     
    char  err_str[512];                                                          
    RTcontext ctx;                                                               
    rt${rt_type}GetContext( self->p, &ctx );                                          
    rtContextGetErrorString( ctx, res, &optix_err_str );                                     
    
    snprintf( err_str, 512, "${rt_type}.${method_name}() failed with error '%s'", optix_err_str );
    PyErr_SetString( PyExc_RuntimeError, err_str );                              
    return 0;                                                                    
  }                                                                              
                                                                                 
  return 0;              

}
''')


file_template = string.Template( '''
/*
 *
 */

#include <Python.h>
#include <optix_host.h>

#include <stdio.h>


${types}

${module_methods}

void initoptix()
{
  PyObject* mod = Py_InitModule("optix", optixMethods);

${type_registrations}

  PyObject* v = 0;

${enum_registrations}
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
    name = strip_rt_prefix( rt_type, rt_func )
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

    params = back.split( ',' )
    params = [ x.split() for x in params ]
    params = [ ( ' '.join( x[0:-1] ), x[-1]) if x else x for x in params ]

    return ( rt_type, ret, funcname, params )


def parse_enums():
    enum_list = []

    with open( os.path.join( sys.argv[1], 'internal', 'optix_declarations.h' ) ) as optix_decls:

        optix_decls_string = optix_decls.read()

        enums_re     = re.compile( 'typedef\s+enum\s+{[^\}]+}\s+\w+;', re.DOTALL )
        enum_val_re  = re.compile( '^\s+(RT_\w+)[^,}]*[,}]', re.DOTALL|re.MULTILINE)
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
        for line in optix_header:
            line = line.strip()
            if line and line[0] == '#':
                continue
            tokens = line.split()
            if len( tokens ) >= 3 and tokens[1] == 'RTAPI':
                (rt_object, ret, funcname, params ) = parseFunctionPrototype( line )
                funcs[ rt_object ].append( (ret, funcname, params ) )

    return funcs




def create_method_code( rt_type, method_name, func ):
    ( ret, funcname, params ) = func 

    return type_method_def_template.substitute( 
            rt_type=rt_type,
            method_name=method_name,
            optix_func_name=funcname
            )


def create_type_method( rt_type, func ):
    # keith
    
    method_name = rt_funcname_to_methodname( rt_type, func[1] )
    method_registration = method_registration_template.substitute( 
            rt_type=rt_type,
            method_name=method_name
            )
    method = create_method_code( rt_type, method_name, func )
    return ( method_registration, method )


def get_type_creaters( rt_type, func_decls, context_func_decls ):
    for func in func_decls:
        if is_create_method( func[1] ):
            func_decls.remove( func )

            new_funcname = 'rtContextCreate{}{}'.format(
                    rt_type,
                    func[1][ len( get_rt_prefix( rt_type )+'Create' ): ]
                    )
            print >> sys.stderr, '{} : {}'.format( func[1], new_funcname )
            context_func_decls.append( (func[0], new_funcname, func[2] ) )


def create_type_methods( rt_type, func_decls, context_func_decls ):
    if rt_type != 'Context' :
        get_type_creaters( rt_type, func_decls, context_func_decls)
        
    for func_decl in func_decls:
        print >> sys.stderr, func_decl

    type_methods = ''
    method_registrations = ''
    for func in func_decls:
        ( method_registration, method ) = create_type_method( rt_type, func )
        type_methods += method
        method_registrations += method_registration

    print >> sys.stderr, type_methods
    print >> sys.stderr, method_registrations

    type_methods += methods_struct_template.substitute( 
        rt_type=rt_type, 
        method_registrations = method_registrations 
        )
    return type_methods


def create_types( funcs ):    
    types = ''
    for rt_type in rt_types:
        rt_type_methods = create_type_methods( rt_type, funcs[rt_type], funcs['Context'])
        types += type_template.substitute( 
                rt_type         = rt_type,
                opaque_type     = get_opaque_type( rt_type ),
                rt_type_methods = rt_type_methods 
                )
    return types


def create_type_registrations():
    type_registrations = ''
    for rt_type in rt_types:
        type_registrations += type_reg_template.substitute( rt_type=rt_type )
    return type_registrations


def create_module_methods( funcs ):
    return '''
static PyMethodDef optixMethods[] =
  {
     { NULL,    NULL,       0,            NULL              }
  };
'''


def create_enum_registrations():
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

print file_template.substitute(
    types              = create_types( funcs ),
    module_methods     = create_module_methods( funcs[ None ] ),
    type_registrations = create_type_registrations(),
    enum_registrations = create_enum_registrations()
    )











