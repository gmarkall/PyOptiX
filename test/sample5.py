#!/usr/bin/env python

import optix 
import os
import sys 
#from PIL import Image
import numpy

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
        print '\there 0.0'
        context = optix.contextCreate()
        print '\t\there 0.0.0'
        context.setRayTypeCount( 1 )
        print '\t\there 0.0.1'
        context.setEntryPointCount( 1 )
  
        print '\there 0.1'
        # should be context["radiance_ray_type"].set( 0 )
        variable = context.declareVariable( "radiance_ray_type" )
        variable.set1ui( 0 )
        variable = context.declareVariable( "scene_epsilon" )
        variable.set1f( 0.0001 )
  
        print '\there 0.2'
        output_buffer = context.declareVariable( "output_buffer" )
        print '\t\there 0.2.0'
        self.output_buffer = context.bufferCreate( optix.RT_BUFFER_OUTPUT )
        print '\t\there 0.2.1'
        self.output_buffer.setFormat( optix.RT_FORMAT_UNSIGNED_BYTE4 )
        print '\t\there 0.2.2'
        self.output_buffer.setSize2D( self.WIDTH, self.HEIGHT )
        print '\t\there 0.2.3'
        output_buffer.setObject( self.output_buffer ) # TODO:test this
  
        print '\there 0.3'
        # ray gen prog 
        # TODO: make context.createProgramFromPTXFile( string1, string2)
        ray_gen_program = context.programCreateFromPTXFile( self.get_ptx_path( 'pinhole_camera' ), "pinhole_camera" )
        print '\t\there 0.3.0'
        context.setRayGenerationProgram( 0, ray_gen_program )
  
        print '\there 0.4'
        variable = context.declareVariable( "eye" )
        variable.set3f( f1=0, f2=0, f3=5 )
        #variable.set3f( 0, 0, 5 )
        variable = context.declareVariable( "U" )
        variable.set3f( 2.88675, 0, 0 )
        variable = context.declareVariable( "V" )
        variable.set3f( 0, 2.16506, 0 )
        variable = context.declareVariable( "W" )
        variable.set3f( 0, 0, -5 )
  
        print '\there 0.5'
        # exception program
        exception_program = context.programCreateFromPTXFile( self.get_ptx_path( 'pinhole_camera' ), 'exception' )
        context.setExceptionProgram( 0, exception_program )
        variable = context.declareVariable( "bad_color" )
        variable.set3f( 1.0, 1.0, 0.0 )
  
        print '\there 0.6'
        # miss prog
        miss_program = context.programCreateFromPTXFile( self.get_ptx_path( 'constantbg' ), 'miss' )
        context.setMissProgram( 0, miss_program)
        variable = context.declareVariable( "bg_color" )
        #variable.set3f( 0.3, 0.1, 0.2 )
        variable.set3f( 0.3, 0.1, 0.1 )
  
        return context
  
    def createGeometry( self ):
        sphere = self.context.geometryCreate()
        sphere.setPrimitiveCount( 1 )
        ptx_path = self.get_ptx_path( 'sphere' )
        sphere.setBoundingBoxProgram( self.context.programCreateFromPTXFile( ptx_path, 'bounds' ) )
        sphere.setIntersectionProgram( self.context.programCreateFromPTXFile( ptx_path, 'intersect' ) )
        variable = sphere.declareVariable( "sphere" )
        variable.set4f( 0, 0, 0, 1.5 )
        return sphere;

    def createMaterial( self ):
        chp = self.context.programCreateFromPTXFile( self.get_ptx_path( 'normal_shader' ), 'closest_hit_radiance' )
        matl = self.context.materialCreate()
        matl.setClosestHitProgram( 0, chp )
        return matl
  
  
    def createInstance( self ):
        # Create geometry instance
        gi = self.context.geometryInstanceCreate()
        gi.setMaterialCount( 1 )
        gi.setGeometry( self.sphere )
        gi.setMaterial( 0, self.material )
  
        # Create geometry group
        geometrygroup = self.context.geometryGroupCreate()
        geometrygroup.setChildCount( 1 )
        print 'geometrygroup child count: {}'.format( geometrygroup.getChildCount() )
        geometrygroup.setChild( 0, gi )
  
        accel =  self.context.accelerationCreate() # TODO: Context.createAcceleration( trav, build )
        accel.setTraverser( "NoAccel" )
        accel.setBuilder( "NoAccel" )
        geometrygroup.setAcceleration( accel )
  
        variable = self.context.declareVariable( "top_object" )
        variable.setObject( geometrygroup )
  

    def run( self ):
        print "here 0"
        self.context  = self.createContext()
        print "here 1"
        self.sphere   = self.createGeometry()
        print "here 2"
        self.material = self.createMaterial()
        print "here 3"
        self.createInstance()
  
        print "here 4"
        self.context.validate()
        print "here 5"
        self.context.compile()
        print "here 6"
        self.context.launch2D( 0, self.WIDTH, self.HEIGHT)
  
        # display
        print "here 7"
        #print 'Map: ' + str( self.output_buffer.map() )
        #self.output_buffer.writeToPPM "foo.ppm" 
        '''
        im = Image.fromarray( self.output_buffer.map() )
        im.save("your_file.jpeg")
        '''
        array = self.output_buffer.map()
        print array.size
        print array.shape
        #print dir( array )
        with open( 'output.ppm', 'wb' ) as image_file:
            print >> image_file, 'P3'
            print >> image_file, '{} {}'.format( array.shape[0], array.shape[1] )
            print >> image_file, '255'

            for i in range( array.shape[0] ):
                for j in range( array.shape[1] ):
                    print >> image_file, '{}'.format( array[i][j][2] ),
                    print >> image_file, '{}'.format( array[i][j][1] ),
                    print >> image_file, '{}'.format( array[i][j][0] ),
            #flat = array.flatten( 'A' )
            #for i in flat:
            #    print >> image_file, '{}'.format( i ),

            
  
        self.context.destroy()
        print "here 8"
  
  
ptx_path = sys.argv[1]
Sample5( ptx_path ).run()
