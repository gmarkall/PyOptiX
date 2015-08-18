
#include "PyOptixDecls.h"
#include <stdio.h>
#include <numpy/arrayobject.h>

#define CHECK_RT_RESULT( res, ctx, py_funcname )                                \
do {                                                                            \
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
} while( 0 )



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


static PyObject* Variable_setUint( PyObject* self, PyObject* args, PyObject* kwds )
{
  unsigned int u1 = 0, u2 = 0, u3 = 0, u4 = 0;
  static char* kwlist[] = { "u1", "u2", "u3", "u4", NULL };
  if( !PyList_Check( args ) )
    fprintf( stderr, "ARGS NOT LIST!!!!!!!!!!!\n\n" );
  if( !PyArg_ParseTupleAndKeywords( args, kwds, "I|III:Variable.setUint", kwlist, &u1, &u2, &u3, &u4 ) )
    return 0; 

  Py_ssize_t len = args ? PyList_Size( args ) : 0;
  RTvariable v = ( (Variable*) self )->p;
  if(      len < 2 && !( kwds && PyDict_GetItemString( kwds, "u2" ) ) )
    CHECK_RT_RESULT( rtVariableSet1ui( v, u1 ), 0, "Variable.setUint" );
  else if( len < 3 && !( kwds && PyDict_GetItemString( kwds, "u3" ) ) )
    CHECK_RT_RESULT( rtVariableSet2ui( v, u1, u2 ), 0, "Variable.setUint" );
  else if( len < 4 && !( kwds && PyDict_GetItemString( kwds, "u4" ) ) )
    CHECK_RT_RESULT( rtVariableSet3ui( v, u1, u2, u3 ), 0, "Variable.setUint" );
  else 
    CHECK_RT_RESULT( rtVariableSet4ui( v, u1, u2, u3, u4 ), 0, "Variable.setUint" );

  return Py_BuildValue("");
}


static PyObject* Variable_setInt( PyObject* self, PyObject* args, PyObject* kwds )
{
  int i1 = 0, i2 = 0, i3 = 0, i4 = 0;
  static char* kwlist[] = { "i1", "i2", "i3", "i4", NULL };
  if( !PyList_Check( args ) )
    fprintf( stderr, "ARGS NOT LIST!!!!!!!!!!!\n\n" );
  if( !PyArg_ParseTupleAndKeywords( args, kwds, "i|iii:Variable.setInt", kwlist, &i1, &i2, &i3, &i4 ) )
    return 0; 

  Py_ssize_t len = args ? PyList_Size( args ) : 0;
  RTvariable v = ( (Variable*) self )->p;
  if(      len < 2 && !( kwds && PyDict_GetItemString( kwds, "i2" ) ) )
    CHECK_RT_RESULT( rtVariableSet1i( v, i1 ), 0, "Variable.setInt" );
  else if( len < 3 && !( kwds && PyDict_GetItemString( kwds, "i3" ) ) )
    CHECK_RT_RESULT( rtVariableSet2i( v, i1, i2 ), 0, "Variable.setInt" );
  else if( len < 4 && !( kwds && PyDict_GetItemString( kwds, "i4" ) ) )
    CHECK_RT_RESULT( rtVariableSet3i( v, i1, i2, i3 ), 0, "Variable.setInt" );
  else 
    CHECK_RT_RESULT( rtVariableSet4i( v, i1, i2, i3, i4 ), 0, "Variable.setInt" );

  return Py_BuildValue("");
}


/*static PyObject* Variable_setFloat( PyObject* self, PyObject* args, PyObject* kwds )
 * */
static PyObject* Variable_setFloat( PyObject* self, PyObject* args, PyObject* kwds )
{
  float i1 = 0, i2 = 0, i3 = 0, i4 = 0;
  if( !PyArg_ParseTuple( args, "ffff",  &i1, &i2, &i3, &i4 ) )
    return 0; 

  /*
  float f1 = 0, f2 = 0, f3 = 0, f4 = 0;
  static char* kwlist[] = { "f1", "f2", "f3", "f4", NULL };
  if( !PyList_Check( args ) )
    fprintf( stderr, "ARGS NOT LIST!!!!!!!!!!!\n\n" );
  if( !PyArg_ParseTupleAndKeywords( args, kwds, "f|fff:Variable.setFloat", kwlist, &f1, &f2, &f3, &f4 ) )
    return 0; 

    */
  /*
  Py_ssize_t len = args ? PyObject_Size( args ) : 0;
  fprintf( file, "SIZE: %i\n", len );
  RTvariable v = ( (Variable*) self )->p;
  if(      len < 2 && !( kwds && PyDict_GetItemString( kwds, "f2" ) ) )
    CHECK_RT_RESULT( rtVariableSet1i( v, f1 ), 0, "Variable.setFloat" );
  else if( len < 3 && !( kwds && PyDict_GetItemString( kwds, "f3" ) ) )
    CHECK_RT_RESULT( rtVariableSet2i( v, f1, f2 ), 0, "Variable.setFloat" );
  else if( len < 4 && !( kwds && PyDict_GetItemString( kwds, "f4" ) ) )
    CHECK_RT_RESULT( rtVariableSet3i( v, f1, f2, f3 ), 0, "Variable.setFloat" );
  else 
    CHECK_RT_RESULT( rtVariableSet4i( v, f1, f2, f3, f4 ), 0, "Variable.setFloat" );
    */

  return Py_BuildValue("");
}

