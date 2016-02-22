
#include "PyOptixDecls.h"
#include <stdio.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>


// TODO: right now we leak created optix objects if the creation process fails
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
  static char* kwlist[] = { "ray_type_count", "entry_point_count", 0 };
  if( !PyArg_ParseTupleAndKeywords( args, kwds, "|ii:optix.createContext", kwlist,
        &ray_type_count, &entry_point_count ) )
    return 0; 

  RTcontext context=0;
  CHECK_RT_RESULT( rtContextCreate( &context ), 0, "optix.createContext" );

  if( ray_type_count != -1 )
    CHECK_RT_RESULT( rtContextSetRayTypeCount( context, ray_type_count ), context, "optix.createContext" );

  if( entry_point_count != -1 )
    CHECK_RT_RESULT( rtContextSetEntryPointCount( context, entry_point_count), context, "optix.createContext" );

  return Py_BuildValue( "O&", ContextNew, context );
}


static PyObject* Context_createAcceleration( PyObject* self, PyObject* args, PyObject* kwds )
{
  const char* builder = 0;
  const char* traverser = 0;
  
  static char* kwlist[] = { "builder", "traverser", 0 };
  if( !PyArg_ParseTupleAndKeywords( args, kwds, "|ss:context.createAcceleration", kwlist,
        &builder, &traverser ) )
    return 0; 

  RTcontext context = ( ( Context* )self )->p;
  RTacceleration  p = 0;
  CHECK_RT_RESULT( rtAccelerationCreate( context, &p ), context, "context.createAcceleration" );

  if( builder )
    CHECK_RT_RESULT( rtAccelerationSetBuilder( p, builder ), context, "context.createAcceleration" );
  if( traverser )
    CHECK_RT_RESULT( rtAccelerationSetTraverser( p, traverser ), context, "context.createAcceleration" );

  return Py_BuildValue( "O&", AccelerationNew, p );
}


static PyObject* Context_createBuffer( PyObject* self, PyObject* args, PyObject* kwds )
{
  static char* kwlist[] = { "bufferdesc", "format", "levels", "width", "height", "depth", 0 };
  int bufferdesc = 0;
  int format = -1;
  int levels = -1;
  Py_ssize_t width = 0, height=0, depth=0;
  if( !PyArg_ParseTupleAndKeywords( args, kwds, "i|iinnn:context.createBuffer", kwlist,
        &bufferdesc, &format, &levels, &width, &height, &depth ) )
    return 0; 

  RTcontext context = ( ( Context* )self )->p;
  RTbuffer  p = 0;
  CHECK_RT_RESULT( rtBufferCreate( context, bufferdesc, &p ), context, "context.createBuffer" );
  if( format != -1 )
    CHECK_RT_RESULT( rtBufferSetFormat( p, format ), context, "context.createBuffer" );
  if( levels != -1 )
    CHECK_RT_RESULT( rtBufferSetMipLevelCount( p, levels ), context, "context.createBuffer" );

  if( depth != 0 )
    CHECK_RT_RESULT( rtBufferSetSize3D( p, width, height, depth ), context, "context.createBuffer" );
  else if( height != 0 )
    CHECK_RT_RESULT( rtBufferSetSize2D( p, width, height ), context, "context.createBuffer" );
  else if( width != 0 )
    CHECK_RT_RESULT( rtBufferSetSize1D( p, width ), context, "context.createBuffer" );

  return Py_BuildValue( "O&", BufferNew, p );
}


static PyObject* Context_createGeometry( PyObject* self, PyObject* args, PyObject* kwds )
{
  Py_ssize_t primitive_count = 0;
  Py_ssize_t index_offset = 0;
  Program* bbox = 0;
  Program* isect = 0;
  static char* kwlist[] = {
    "primitive_count",
    "index_offset",
    "bounding_box_program",
    "intersection_program",
    0
  };
  if( !PyArg_ParseTupleAndKeywords( args, kwds, "|nnO!O!:context.createGeometry", kwlist,
        &primitive_count, &index_offset, &ProgramType, &bbox, &ProgramType, &isect ) )
    return 0; 

  RTcontext context = ( ( Context* )self )->p;
  RTgeometry  p = 0;
  CHECK_RT_RESULT( rtGeometryCreate( context, &p ), context, "context.createGeometry" );

  if( primitive_count )
    CHECK_RT_RESULT( rtGeometrySetPrimitiveCount( p, primitive_count ), context, "context.createGeometry" );
  if( index_offset )
    CHECK_RT_RESULT( rtGeometrySetPrimitiveIndexOffset( p, index_offset ), context, "context.createGeometry" );
  if( bbox )
    CHECK_RT_RESULT( rtGeometrySetBoundingBoxProgram( p, bbox->p ), context, "context.createGeometry" );
  if( isect )
    CHECK_RT_RESULT( rtGeometrySetIntersectionProgram( p, isect->p ), context, "context.createGeometry" );

  return Py_BuildValue( "O&", GeometryNew, p );
}


static PyObject* Context_createGeometryGroup( PyObject* self, PyObject* args, PyObject* kwds )
{
  Acceleration* accel = 0;
  PyObject* children = 0;
  static char* kwlist[] = {
    "acceleration",
    "children",
    0
  };
  if( !PyArg_ParseTupleAndKeywords( args, kwds, "|O!O:context.createGeometryGroup", kwlist,
        &AccelerationType, &accel, &children ) )
    return 0; 

  RTcontext context = ( ( Context* )self )->p;
  RTgeometrygroup  p = 0;
  CHECK_RT_RESULT( rtGeometryGroupCreate( context, &p ), context, "context.createGeometryGroup" );

  if( accel )
    CHECK_RT_RESULT( rtGeometryGroupSetAcceleration( p, accel->p ), context, "context.createGeometryGroup" );

  if( children )
  {
    if( !PySequence_Check( children ) )
    {
      rtGeometryGroupDestroy( p );
      PyErr_SetString( PyExc_RuntimeError, "Context.createGeometryGroup passed non-sequence child list" );
      return 0;
    }

    int num_children = PySequence_Size( children );
    CHECK_RT_RESULT( rtGeometryGroupSetChildCount( p, num_children ), context, "context.createGeometryGroup" );
    for( int i = 0; i < num_children; ++i )
    {
      PyObject* child = PySequence_GetItem( children, i );
      if( !PyObject_TypeCheck( child, &GeometryInstanceType ) )
      {
        PyErr_SetString( PyExc_RuntimeError, "Context.createGeometry passed non-GeometryInstance child" );
        CHECK_RT_RESULT( rtGeometryGroupSetChildCount( p, 0 ), context, "context.createGeometryGroup" );
        return 0;
      }
      GeometryInstance* gi = (GeometryInstance*)child;
      CHECK_RT_RESULT( rtGeometryGroupSetChild( p, i, gi->p ), context, "context.createGeometryGroup" );
    }
  }

  return Py_BuildValue( "O&", GeometryGroupNew, p );
}


static PyObject* Context_createGeometryInstance( PyObject* self, PyObject* args, PyObject* kwds )
{
  Geometry* geometry = 0;
  PyObject* materials = 0;
  static char* kwlist[] = {
    "geometry",
    "materials",
    0
  };
  if( !PyArg_ParseTupleAndKeywords( args, kwds, "|O!O:context.createGeometryInstance", kwlist,
        &GeometryType, &geometry, &materials ) )
    return 0; 

  RTcontext context = ( ( Context* )self )->p;
  RTgeometryinstance  p = 0;
  CHECK_RT_RESULT( rtGeometryInstanceCreate( context, &p ), context, "context.createGeometryInstance" );

  if( geometry )
    CHECK_RT_RESULT( rtGeometryInstanceSetGeometry( p, geometry->p ), context, "context.createGeometryInstance" );

  if( materials )
  {
    if( !PySequence_Check( materials ) )
    {
      rtGeometryInstanceDestroy( p );
      PyErr_SetString( PyExc_RuntimeError, "Context.createGeometryInstance passed non-sequence materials list" );
      return 0;
    }

    int num_materials = PySequence_Size( materials );
    CHECK_RT_RESULT( rtGeometryInstanceSetMaterialCount( p, num_materials ), context, "context.createGeometryInstance" );
    for( int i = 0; i < num_materials; ++i )
    {
      PyObject* child = PySequence_GetItem( materials, i );
      if( !PyObject_TypeCheck( child, &MaterialType ) )
      {
        PyErr_SetString( PyExc_RuntimeError, "Context.createGeometryInstance passed non-Material in materials list" );
        CHECK_RT_RESULT( rtGeometryInstanceSetMaterialCount( p, 0 ), context, "context.createGeometryInstance" );
        return 0;
      }
      Material* mtl = (Material*)child;
      CHECK_RT_RESULT( rtGeometryInstanceSetMaterial( p, i, mtl->p ), context, "context.createGeometryInstance" );
    }
  }

  return Py_BuildValue( "O&", GeometryInstanceNew, p );
}


static PyObject* Context_createGroup( PyObject* self, PyObject* args, PyObject* kwds )
{
  Acceleration* accel = 0;
  PyObject* children = 0;
  static char* kwlist[] = {
    "acceleration",
    "children",
    0
  };
  if( !PyArg_ParseTupleAndKeywords( args, kwds, "|O!O:context.createGroup", kwlist,
        &AccelerationType, &accel, &children ) )
    return 0; 

  RTcontext context = ( ( Context* )self )->p;
  RTgroup  p = 0;
  CHECK_RT_RESULT( rtGroupCreate( context, &p ), context, "context.createGroup" );

  if( accel )
    CHECK_RT_RESULT( rtGroupSetAcceleration( p, accel->p ), context, "context.createGroup" );

  if( children )
  {
    if( !PySequence_Check( children ) )
    {
      rtGroupDestroy( p );
      PyErr_SetString( PyExc_RuntimeError, "Context.createGroup passed non-sequence child list" );
      return 0;
    }

    int num_children = PySequence_Size( children );
    CHECK_RT_RESULT( rtGroupSetChildCount( p, num_children ), context, "context.createGroup" );
    for( int i = 0; i < num_children; ++i )
    {
      PyObject* child = PySequence_GetItem( children, i );
      if( !PyObject_TypeCheck( child, &GroupType         ) &&
          !PyObject_TypeCheck( child, &SelectorType      ) &&
          !PyObject_TypeCheck( child, &GeometryGroupType ) &&
          !PyObject_TypeCheck( child, &TransformType     )
          )
      {
        PyErr_SetString( PyExc_RuntimeError, "Context.createGroup passed child with invalid type" );
        return 0;
      }
      Group* group = (Group*)child; /* pick one of the allowed types randomly.  It will be typecast internally */
      CHECK_RT_RESULT( rtGroupSetChild( p, i, group->p ), context, "context.createGroup" );
    }
  }

  return Py_BuildValue( "O&", GroupNew, p );
}


static PyObject* Context_createMaterial( PyObject* self, PyObject* args, PyObject* kwds )
{
  PyObject* closest_hits = 0;
  PyObject* any_hits = 0;
  static char* kwlist[] = {
    "closest_hit_programs",
    "any_hit_programs",
    0
  };
  if( !PyArg_ParseTupleAndKeywords( args, kwds, "|OO:context.createMaterial", kwlist,
        &closest_hits, &any_hits ) )
    return 0; 

  RTcontext context = ( ( Context* )self )->p;
  RTmaterial  p = 0;
  CHECK_RT_RESULT( rtMaterialCreate( context, &p ), context, "context.createMaterial" );

  if( closest_hits )
  {
    if( !PySequence_Check( closest_hits ) )
    {
      rtMaterialDestroy( p );
      PyErr_SetString( PyExc_RuntimeError, "Context.createMaterial passed non-sequence closest_hit_programs" );
      return 0;
    }

    int num_closest_hits = PySequence_Size( closest_hits );
    for( int i = 0; i < num_closest_hits; ++i )
    {
      PyObject* closest_hit = PySequence_GetItem( closest_hits, i );
      if( closest_hit == Py_None ) /* TODO: check this is correct */
        continue;
      
      if( !PyObject_TypeCheck( closest_hit, &ProgramType ) )
      {
        rtMaterialDestroy( p );
        PyErr_SetString( PyExc_RuntimeError, "Context.createMateriasl passed non-Program in closest_hit_programs list" );
        return 0;
      }
      Program* prog = (Program*)closest_hit;
      CHECK_RT_RESULT( rtMaterialSetClosestHitProgram( p, i, prog->p ), context, "context.creatematerial" );
    }
  }
  
  if( any_hits )
  {
    if( !PySequence_Check( any_hits ) )
    {
      rtMaterialDestroy( p );
      PyErr_SetString( PyExc_RuntimeError, "Context.createMaterial passed non-sequence any_hit_programs" );
      return 0;
    }

    int num_any_hits = PySequence_Size( any_hits );
    for( int i = 0; i < num_any_hits; ++i )
    {
      PyObject* any_hit = PySequence_GetItem( any_hits, i );
      if( any_hit == Py_None ) /* TODO: check this is correct */
        continue;

      if( !PyObject_TypeCheck( any_hit, &ProgramType ) )
      {
        rtMaterialDestroy( p );
        PyErr_SetString( PyExc_RuntimeError, "Context.createMateriasl passed non-Program in any_hit_programs list" );
        return 0;
      }
      Program* prog = (Program*)any_hit;
      CHECK_RT_RESULT( rtMaterialSetAnyHitProgram( p, i, prog->p ), context, "context.creatematerial" );
    }
  }

  return Py_BuildValue( "O&", MaterialNew, p );
}


static PyObject* Context_createProgramFromPTXFile( PyObject* self, PyObject* args, PyObject* kwds )
{
  const char* filename = 0;
  const char* program = 0;
  static char* kwlist[] = {
    "filename",
    "program",
    0
  };
  if( !PyArg_ParseTupleAndKeywords( args, kwds, "ss:context.createProgram", kwlist, &filename, &program ) )
    return 0; 

  RTcontext context = ( ( Context* )self )->p;
  RTprogram  p = 0;
  CHECK_RT_RESULT( rtProgramCreateFromPTXFile( context, filename, program, &p ), context, "context.createProgram" );

  return Py_BuildValue( "O&", ProgramNew, p );
}


static PyObject* Context_createTextureSampler( PyObject* self, PyObject* args, PyObject* kwds )
{
  PyObject* wrap_mode            = 0; // (dimension, wrapmode) 
  int       filter_minification  = 0;
  int       filter_magnification = 0;
  int       filter_mipmapping    = 0;
  float     anisotropy_max       = 0.0f;
  PyObject* miplevel_clamp       = 0; // (min, max)
  float     miplevel_bias        = 0.0f;
  int       read_mode            = -1;
  int       indexing_mode        = -1;
  Buffer*   buffer               = 0;


  static char* kwlist[] = {
    "wrap_mode",
    "filter_minification",
    "filter_magnification",
    "filter_mipmapping",
    "anisotropy_max",
    "miplevel_clamp",
    "miplevel_bias",
    "read_mode",
    "indexing_mode",
    "buffer",
    0
  };


  if( !PyArg_ParseTupleAndKeywords( args, kwds, "|OiiifOfii!O:context.createTextureSampler", kwlist,
        &wrap_mode,
        &filter_minification,
        &filter_magnification,
        &filter_mipmapping,
        &anisotropy_max,
        &miplevel_clamp,
        &miplevel_bias,
        &read_mode,
        &indexing_mode,
        &BufferType,
        &buffer
        ) )
    return 0; 
  
  RTcontext context = ( ( Context* )self )->p;
  RTtexturesampler p = 0;
  CHECK_RT_RESULT( rtTextureSamplerCreate( context, &p ), context, "context.createTextureSampler" );

  if( wrap_mode )
  {
    if( !PySequence_Check( wrap_mode) )
    {
      PyErr_SetString( PyExc_RuntimeError, "Context.createTextureSampler passed non-sequence wrap_mode argument" );
      return 0;
    }

    const int wrap_mode_size = PySequence_Size( wrap_mode );
    if( wrap_mode_size != 2 )
    {
      PyErr_SetString( PyExc_RuntimeError, "Context.createTextureSampler passed wrap_mode with size != 2" );
      rtTextureSamplerDestroy( p );
      return 0;
    }

    PyObject* dim = PySequence_GetItem( wrap_mode, 0 );
    PyObject* int_dim = PyNumber_Long( dim );
    if( !int_dim )
    {
      PyErr_SetString(
          PyExc_RuntimeError,
          "Context.createTextureSampler passed non-int convertible wrap_mode dimension"
          );
      rtTextureSamplerDestroy( p );
      return 0;
    }
    
    PyObject* mode = PySequence_GetItem( wrap_mode, 1 );
    PyObject* int_mode = PyNumber_Long( mode );
    if( !int_mode )
    {
      PyErr_SetString(
          PyExc_RuntimeError,
          "Context.createTextureSampler passed non-int convertible wrap_mode mode"
          );
      rtTextureSamplerDestroy( p );
      return 0;
    }
  
    CHECK_RT_RESULT(
        rtTextureSamplerSetWrapMode( p, PyLong_AsLong( int_dim ), PyLong_AsLong( int_mode ) ),
        context,
        "context.createTextureSampler"
        );
  }
  
  if( filter_minification || filter_magnification || filter_mipmapping )
  {  
    CHECK_RT_RESULT( 
        rtTextureSamplerSetFilteringModes( p, filter_minification, filter_magnification, filter_mipmapping ),
        context,
        "context.createTextureSampler"
        );
  }

  if( anisotropy_max != 0.0f )
  {
    CHECK_RT_RESULT( 
        rtTextureSamplerSetMaxAnisotropy( p, anisotropy_max ), 
        context,
        "context.createTextureSampler"
        );
  }

  if( miplevel_clamp )
  {
    if( !PySequence_Check( miplevel_clamp) )
    {
      PyErr_SetString( PyExc_RuntimeError, "Context.createTextureSampler passed non-sequence miplevel_clamp argument" );
      return 0;
    }

    const int miplevel_clamp_size = PySequence_Size( miplevel_clamp );
    if( miplevel_clamp_size != 2 )
    {
      PyErr_SetString( PyExc_RuntimeError, "Context.createTextureSampler passed miplevel_clamp with size != 2" );
      rtTextureSamplerDestroy( p );
      return 0;
    }

    PyObject* min = PySequence_GetItem( miplevel_clamp, 0 );
    PyObject* float_min = PyNumber_Float( min );
    if( !float_min )
    {
      PyErr_SetString(
          PyExc_RuntimeError,
          "Context.createTextureSampler passed non-float convertible miplevel_clamp min value"
          );
      rtTextureSamplerDestroy( p );
      return 0;
    }
    
    PyObject* max = PySequence_GetItem( miplevel_clamp, 1 );
    PyObject* float_max = PyNumber_Float( max );
    if( !float_max )
    {
      PyErr_SetString(
          PyExc_RuntimeError,
          "Context.createTextureSampler passed non-float convertible miplevel_clamp max_value"
          );
      rtTextureSamplerDestroy( p );
      return 0;
    }
  
    CHECK_RT_RESULT(
        rtTextureSamplerSetMipLevelClamp( p, PyFloat_AsDouble( float_min ), PyFloat_AsDouble( float_max ) ),
        context,
        "context.createTextureSampler"
        );
  }

  if( miplevel_bias != 0.0f )
  {
    CHECK_RT_RESULT(
        rtTextureSamplerSetMipLevelBias( p, miplevel_bias ),
        context,
        "context.createTextureSampler"
        );
  }

  if( read_mode != -1 )
  {
    CHECK_RT_RESULT(
        rtTextureSamplerSetReadMode( p, read_mode ),
        context,
        "context.createTextureSampler"
        );
  }
  
  if( indexing_mode != -1 )
  {
    CHECK_RT_RESULT(
        rtTextureSamplerSetIndexingMode( p, indexing_mode ),
        context,
        "context.createTextureSampler"
        );
  }

  if( buffer )
  {
    const unsigned deprecated = 0;
    CHECK_RT_RESULT(
        rtTextureSamplerSetBuffer( p, deprecated, deprecated, buffer->p ),
        context,
        "context.createTextureSampler"
        );
  }

  return Py_BuildValue( "O&", TextureSamplerNew, p );
}


static int get_matrix_data( PyObject* matrix, float matrix_data[16] )
{
  if( !PySequence_Check( matrix ) )
  {
    PyErr_SetString( PyExc_RuntimeError, "Context.createTransform passed non-sequence matrix argument" );
    return 0;
  }

  const int matrix_size = PySequence_Size( matrix );
  if( matrix_size != 16 )
  {
    PyErr_SetString( PyExc_RuntimeError, "Context.createTransform passed matrix with size != 16" );
    return 0;
  }

  for( int i = 0; i < matrix_size; ++i )
  {
    PyObject* elmt = PySequence_GetItem( matrix, i );
    PyObject* float_elmt = PyNumber_Float( elmt );
    if( !float_elmt )
    {
      PyErr_SetString( PyExc_RuntimeError, "Context.createTransform passed matrix with non-float convertible element" );
      return 0;
    }
    matrix_data[i] = PyFloat_AsDouble( float_elmt );
  }
  //fprintf( stderr, "%f, %f, %f\n\n", matrix_data[0], matrix_data[1], matrix_data[2] );

  return 1;
}


static PyObject* Context_createTransform( PyObject* self, PyObject* args, PyObject* kwds )
{
  PyObject* child     = 0;
  int       transpose = 0;
  PyObject* matrix    = 0;
  PyObject* inverse   = 0;


  static char* kwlist[] = {
    "child",
    "transpose",
    "matrix",
    "inverse_matrix",
    0
  };
  if( !PyArg_ParseTupleAndKeywords( args, kwds, "|OiOO:context.createTransform", kwlist,
        &child, &transpose, &matrix, &inverse ) )
    return 0; 

  RTcontext context = ( ( Context* )self )->p;
  RTtransform  p = 0;
  CHECK_RT_RESULT( rtTransformCreate( context, &p ), context, "context.createTransform" );

  if( child )
  {
      if( !PyObject_TypeCheck( child, &GroupType         ) &&
          !PyObject_TypeCheck( child, &SelectorType      ) &&
          !PyObject_TypeCheck( child, &GeometryGroupType ) &&
          !PyObject_TypeCheck( child, &TransformType     )
          )
      {
        PyErr_SetString( PyExc_RuntimeError, "Context.createTransform passed child with invalid type" );
        return 0;
      }
      Group* group = (Group*)child; /* pick one of the allowed types randomly.  It will be typecast internally */
      CHECK_RT_RESULT( rtTransformSetChild( p, group->p ), context, "context.createTransform" );
  }

  if( matrix )
  {
    float matrix_data[16] = {0.0f};
    if( !get_matrix_data( matrix, matrix_data ) )
    {
      rtTransformDestroy( p );
      return 0;
    }
    
    float inverse_matrix_data[16] = {0.0f};
    if( inverse )
    {
      if( !get_matrix_data( inverse, inverse_matrix_data ) )
      {
        rtTransformDestroy( p );
        return 0;
      }

      CHECK_RT_RESULT( 
          rtTransformSetMatrix( p, transpose, matrix_data, inverse_matrix_data ),
          context,
          "context.createTransform"
          );
    }
    else
    {
      CHECK_RT_RESULT( 
          rtTransformSetMatrix( p, transpose, matrix_data, 0 ),
          context,
          "context.createTransform"
          );
    }
  }

  return Py_BuildValue( "O&", TransformNew, p );
}

/*
static PyObject* Context_createAcceleration( PyObject* self, PyObject* args, PyObject* kwds )
{
  static char* kwlist[] = {
    0
  };
  if( !PyArg_ParseTupleAndKeywords( args, kwds, ":context.createAcceleration", kwlist ) )
    return 0; 

  RTcontext context = ( ( Context* )self )->p;
  RTacceleration  p = 0;
  CHECK_RT_RESULT( rtAccelerationCreate( context, &p ), context, "context.createAcceleration" );

  return Py_BuildValue( "O&", AccelerationNew, p );
}
*/


static PyObject* Variable_setUint( PyObject* self, PyObject* args, PyObject* kwds )
{
  unsigned int u1 = 0, u2 = 0, u3 = 0, u4 = 0;
  static char* kwlist[] = { "u1", "u2", "u3", "u4", 0 };
  if( !PyArg_ParseTupleAndKeywords( args, kwds, "I|III:Variable.setUint", kwlist, &u1, &u2, &u3, &u4 ) )
    return 0; 

  Py_ssize_t len = args ? PyObject_Size( args ) : 0;
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
  static char* kwlist[] = { "i1", "i2", "i3", "i4", 0 };
  if( !PyArg_ParseTupleAndKeywords( args, kwds, "i|iii:Variable.setInt", kwlist, &i1, &i2, &i3, &i4 ) )
    return 0; 

  Py_ssize_t len = args ? PyObject_Size( args ) : 0;
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


static PyObject* Variable_setFloat( PyObject* self, PyObject* args, PyObject* kwds )
{
  float f1 = 0, f2 = 0, f3 = 0, f4 = 0;
  static char* kwlist[] = { "f1", "f2", "f3", "f4", 0 };
  if( !PyArg_ParseTupleAndKeywords( args, kwds, "f|fff:Variable.setFloat", kwlist, &f1, &f2, &f3, &f4 ) )
    return 0; 

  Py_ssize_t len = args ? PyObject_Size( args ) : 0;
  RTvariable v = ( (Variable*) self )->p;
  if(      len < 2 && !( kwds && PyDict_GetItemString( kwds, "f2" ) ) )
    CHECK_RT_RESULT( rtVariableSet1f( v, f1 ), 0, "Variable.setFloat" );
  else if( len < 3 && !( kwds && PyDict_GetItemString( kwds, "f3" ) ) )
    CHECK_RT_RESULT( rtVariableSet2f( v, f1, f2 ), 0, "Variable.setFloat" );
  else if( len < 4 && !( kwds && PyDict_GetItemString( kwds, "f4" ) ) )
    CHECK_RT_RESULT( rtVariableSet3f( v, f1, f2, f3 ), 0, "Variable.setFloat" );
  else 
    CHECK_RT_RESULT( rtVariableSet4f( v, f1, f2, f3, f4 ), 0, "Variable.setFloat" );

  return Py_BuildValue("");
}


