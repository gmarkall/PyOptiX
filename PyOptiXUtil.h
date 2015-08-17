
#include "PyOptixDecls.h"
#include <numpy/arrayobject.h>

#define CHECK_RT_RESULT( res, ctx, py_funcname )                                \
{                                                                               \
  if( ( res ) != RT_SUCCESS )                                                   \
  {                                                                             \
    const char* optix_err_str = 0;                                              \
    char  err_str[1024];                                                        \
    rtContextGetErrorString( ctx, res, &optix_err_str );                        \
                                                                                \
    snprintf( err_str, 1024, "%s() failed with error '%s'", py_funcname, optix_err_str ); \
    PyErr_SetString( PyExc_RuntimeError, err_str );                             \
    return 0;                                                                   \
  }                                                                             \
}



static void getNumpyElementType( RTformat format, int* element_py_array_type, unsigned int* element_dimensionality )
{
  switch( format )
  {
    case RT_FORMAT_HALF:
    case RT_FORMAT_HALF2:
    case RT_FORMAT_HALF3:
    case RT_FORMAT_HALF4:
      *element_py_array_type  = NPY_FLOAT16;
      *element_dimensionality = 1 + format - RT_FORMAT_HALF;
      return;

    case RT_FORMAT_FLOAT:
    case RT_FORMAT_FLOAT2:
    case RT_FORMAT_FLOAT3:
    case RT_FORMAT_FLOAT4:
      *element_py_array_type  = NPY_FLOAT32;
      *element_dimensionality = 1 + format - RT_FORMAT_FLOAT;
      return;

    case RT_FORMAT_BYTE:
    case RT_FORMAT_BYTE2:
    case RT_FORMAT_BYTE3:
    case RT_FORMAT_BYTE4:
      *element_py_array_type  = NPY_INT8;
      *element_dimensionality = 1 + format - RT_FORMAT_BYTE;
      return;

    case RT_FORMAT_UNSIGNED_BYTE:
    case RT_FORMAT_UNSIGNED_BYTE2:
    case RT_FORMAT_UNSIGNED_BYTE3:
    case RT_FORMAT_UNSIGNED_BYTE4:
      *element_py_array_type  = NPY_UINT8;
      *element_dimensionality = 1 + format - RT_FORMAT_UNSIGNED_BYTE;
      return;

    case RT_FORMAT_SHORT:
    case RT_FORMAT_SHORT2:
    case RT_FORMAT_SHORT3:
    case RT_FORMAT_SHORT4:
      *element_py_array_type  = NPY_INT16;
      *element_dimensionality = 1 + format - RT_FORMAT_SHORT;
      return;

    case RT_FORMAT_UNSIGNED_SHORT:
    case RT_FORMAT_UNSIGNED_SHORT2:
    case RT_FORMAT_UNSIGNED_SHORT3:
    case RT_FORMAT_UNSIGNED_SHORT4:
      *element_py_array_type  = NPY_UINT16;
      *element_dimensionality = 1 + format - RT_FORMAT_UNSIGNED_SHORT;
      return;

    case RT_FORMAT_INT:
    case RT_FORMAT_INT2:
    case RT_FORMAT_INT3:
    case RT_FORMAT_INT4:
      *element_py_array_type  = NPY_INT32;
      *element_dimensionality = 1 + format - RT_FORMAT_INT;
      return;

    case RT_FORMAT_UNSIGNED_INT:
    case RT_FORMAT_UNSIGNED_INT2:
    case RT_FORMAT_UNSIGNED_INT3:
    case RT_FORMAT_UNSIGNED_INT4:
      *element_py_array_type  = NPY_UINT32;
      *element_dimensionality = 1 + format - RT_FORMAT_UNSIGNED_INT;
      return;

    case RT_FORMAT_BUFFER_ID:
    case RT_FORMAT_PROGRAM_ID:
      *element_py_array_type  = NPY_INT32;
      *element_dimensionality = 1;
      return;

    case RT_FORMAT_USER:
    case RT_FORMAT_UNKNOWN:
      *element_py_array_type  = 0;
      *element_dimensionality = 0;
      return;

  }
}


static PyObject* createNumpyArray( RTbuffer buffer, void* data )
{

  unsigned int dimensionality;
  rtBufferGetDimensionality( buffer, &dimensionality );

  RTsize rt_dims[3] = {0};
  rtBufferGetSizev( buffer, dimensionality, rt_dims );

  RTformat format;
  rtBufferGetFormat( buffer, &format );

  int element_py_array_type;
  unsigned int element_dimensionality;
  getNumpyElementType( format, &element_py_array_type, &element_dimensionality );

  npy_intp dims[4];
  dims[0] = rt_dims[0];
  dims[1] = rt_dims[1];
  dims[2] = rt_dims[2];
  dims[3] = 0;
  dims[ dimensionality ] = element_dimensionality;
  dimensionality += (int)( element_dimensionality > 1 );

  /*
  npy_intp strides[4];
  strides[0] = element_dimensionality;
  strides[1] = element_dimensionality*dims[0];
  strides[2] = 1; 
  return PyArray_NewFromDescr(
      &PyArray_Type, 
      PyArray_DescrFromType( element_py_array_type ),
      dimensionality,
      dims, 
      0, 
      data, 
      NPY_ARRAY_C_CONTIGUOUS,
      0 );
      */
  return PyArray_SimpleNewFromData(
      dimensionality,
      dims,
      element_py_array_type,
      data 
      );
}


/*

  {
    "createContext",
    (PyCFunction)optix_createContext,
    METH_VARARGS | METH_KEYWORDS,
    "createContext"
  },

 */
static PyObject* optix_createContext( PyObject* self, PyObject* args, PyObject* kwds )
{
  int ray_type_count    = -1;
  int entry_point_count = -1;
  static char* kwlist[] = { "ray_type_count", "entry_point_count", NULL };
  if( !PyArg_ParseTupleAndKeywords( args, kwds, "|ii:optix.createContext", kwlist, &ray_type_count, &entry_point_count ) )
    return NULL; 

  RTcontext context=0;
  CHECK_RT_RESULT( rtContextCreate( &context ), 0, "optix.createContext" );

  if( ray_type_count != -1 )
    CHECK_RT_RESULT( rtContextSetRayTypeCount( context, ray_type_count ), context, "optix.createContext" );

  if( entry_point_count != -1 )
    CHECK_RT_RESULT( rtContextSetEntryPointCount( context, entry_point_count), context, "optix.createContext" );

  return Py_BuildValue( "O&", ContextNew, context );
}

PyObject* ContextGetItem( PyObject* self, PyObject* key )
{
  if( !PyString_Check( key ) )
    PyErr_SetString( PyExc_TypeError, "Context.getItem() called with non-string key" );

  const char* str = PyString_AsString( key ); 
  RTcontext ctx = ( (Context*)self )->p;

  RTvariable v;
  rtContextQueryVariable( ctx, str, &v );
  if( !v )
    rtContextDeclareVariable( ctx, str, &v );

  return Py_BuildValue( "O&", VariableNew, v );
}

static PyMappingMethods ContextMappingMethods = { 0, ContextGetItem, 0 };
