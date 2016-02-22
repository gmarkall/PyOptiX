#!/usr/bin/env python3

import optix 
import os
import sys 
#import Image
import numpy

def write_png(buf, width, height):
    """
    buf: must be bytes or a bytearray in py3, a regular string in py2.
    formatted RGBARGBA...
    """
    import zlib, struct

    # reverse the vertical line order and add null bytes at the start
    width_byte_4 = width * 4
    raw_data = b''.join(b'\x00' + buf[span:span + width_byte_4]
                        for span in range((height - 1) * width * 4, -1, - width_byte_4))

    def png_pack(png_tag, data):
        chunk_head = png_tag + data
        return (struct.pack("!I", len(data)) +
                chunk_head +
                struct.pack("!I", 0xFFFFFFFF & zlib.crc32(chunk_head)))

    return b''.join([
        b'\x89PNG\r\n\x1a\n',
        png_pack(b'IHDR', struct.pack("!2I5B", width, height, 8, 6, 0, 0, 0)),
        png_pack(b'IDAT', zlib.compress(raw_data, 9)),
        png_pack(b'IEND', b'')])

class Sample5:

    #WIDTH  = 1024
    #HEIGHT = 768 
    
    WIDTH  = 128 
    HEIGHT = 96 


    def __init__( self, ptx_path ): 
        self.context       = None
        self.sphere        = None
        self.material      = None
        self.output_buffer = None
        self.ptx_path      = ptx_path 


    def get_ptx_path( self, filename ):
        return os.path.join( self.ptx_path, 'sample5_generated_' + filename + '.cu.ptx' )
  

    def createContext( self ):
        context = optix.createContext(
                ray_type_count    = 1,
                entry_point_count = 1
                )
  
        xform = context.createTransform( 
                matrix =         [ 1.0, 0.0, 0.0, 0.0,
                                   0.0, 1.0, 0.0, 0.0,
                                   0.0, 0.0, 1.0, 0.0,
                                   0.0, 0.0, 0.0, 1.0 ],
                inverse_matrix = [ 3.0, 4.0, 1.2, 0.0,
                                   0.0, 1.0, 0.0, 0.0,
                                   0.0, 0.0, 1.0, 0.0,
                                   0.0, 0.0, 0.0, 1.0 ],
                )

        context[ 'radiance_ray_type_int' ].setInt( 0 )
        context[ 'radiance_ray_type' ].setUint( 0 )
        v = context[ 'scene_epsilon'     ]
        v.setFloat( 0.0001 )
  
        self.output_buffer = context.createBuffer( 
               bufferdesc = optix.BUFFER_OUTPUT,
               format     = optix.FORMAT_UNSIGNED_BYTE4,
               width      = self.WIDTH,
               height     = self.HEIGHT 
               )
        context[ 'output_buffer' ].setObject( self.output_buffer ) 
  
        # ray gen prog 
        ray_gen_program = context.createProgramFromPTXFile( self.get_ptx_path( 'pinhole_camera' ), "pinhole_camera" )
        context.setRayGenerationProgram( 0, ray_gen_program )
        context[ "eye" ].setFloat( 0, 0, 5 )
        context[ "U" ].setFloat( 2.88675, 0      ,  0 )
        context[ "V" ].setFloat( 0      , 2.16506,  0 )
        context[ "W" ].setFloat( 0      , 0      , -5 )
  
        # exception program
        exception_program = context.createProgramFromPTXFile( 
                filename = self.get_ptx_path( 'pinhole_camera' ),
                program  = 'exception'
                )
        context.setExceptionProgram( 0, exception_program )
        context[ "bad_color" ].setFloat( 1.0, 1.0, 0.0 )
  
        # miss prog
        miss_program = context.programCreateFromPTXFile(
                self.get_ptx_path( 'constantbg' ),
                'miss'
                )
        context.setMissProgram( 0, miss_program)
        context[ "bg_color" ].setFloat( 0.3, 0.1, 0.1 )
  
        return context
  

    def createGeometry( self ):
        ptx_path = self.get_ptx_path( 'sphere' )
        sphere = self.context.createGeometry(
                primitive_count = 1,
                bounding_box_program = self.context.programCreateFromPTXFile(
                    ptx_path, 'bounds' ),
                intersection_program = self.context.programCreateFromPTXFile(
                    ptx_path, 'intersect' )
                )
        sphere[ 'sphere' ].setFloat( 0, 0, 0, 1.5 )
        return sphere;


    def createMaterial( self ):
        chp = self.context.createProgramFromPTXFile( 
                filename = self.get_ptx_path( 'normal_shader' ),
                program  = 'closest_hit_radiance'
                )
        matl = self.context.materialCreate()
        matl.setClosestHitProgram( 0, chp )
        return matl
  
  
    def createInstance( self ):
        # Create geometry instance
        gi = self.context.createGeometryInstance(
                geometry  = self.sphere,
                materials = [ self.material ]
                )
  
        # Create geometry group
        accel =  self.context.createAcceleration( 
                builder   = 'NoAccel',
                traverser = 'NoAccel'
                )
        geometrygroup = self.context.createGeometryGroup( 
                acceleration = accel,
                children     = [ gi ]
                )
  
        self.context[ "top_object" ].setObject( geometrygroup )
  

    def run( self ):
        print( "here 0" )
        self.context  = self.createContext()
        print( "here 1" )
        self.sphere   = self.createGeometry()
        print( "here 2" )
        self.material = self.createMaterial()
        print( "here 3" )
        self.createInstance()
  
        print( "here 4" )
        self.context.validate()
        print( "here 5" )
        self.context.compile()
        print( "here 6" )
        self.context.launch2D( 0, self.WIDTH, self.HEIGHT)
  
        # display
        print( "here 7" )
        #print( 'Map: ' + str( self.output_buffer.map() )
        #self.output_buffer.writeToPPM "foo.ppm" 

        array = self.output_buffer.map()
        print( array.size )
        print( array.shape )
        data = write_png( array.tostring(), array.shape[0], array.shape[1] )
        with open("my_image.png", 'wb') as fd:
            fd.write(data)

        '''
        #print( dir( array )
        with open( 'output.ppm', 'wb' ) as ifile:
            ifile.write( bytes( 'P3', 'UTF-8' ) )
            ifile.write( bytes( '{} {}'.format(array.shape[0], array.shape[1]), 'UTF-8' ) )
            ifile.write( bytes( '255', 'UTF-8' ) )

            for i in range( array.shape[0] ):
                for j in range( array.shape[1] ):
                    ifile.write( bytes( '{}'.format(array[i][j][2]), 'UTF-8' ) )
                    ifile.write( bytes( '{}'.format(array[i][j][1]), 'UTF-8' ) )
                    ifile.write( bytes( '{}'.format(array[i][j][2]), 'UTF-8' ) )
            #flat = array.flatten( 'A' )
            #for i in flat:
            #    print( >> image_file, '{}'.format( i ),

       '''
            
  
        self.context.destroy()
        print( "here 8" )
  
  
ptx_path = sys.argv[1]
Sample5( ptx_path ).run()
