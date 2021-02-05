


import cupy as cp
import optix
import pytest 


import tutil

class Logger:
    def __init__( self ):
        self.num_mssgs = 0
    
    def __call__( self, level, tag, mssg ):
        print( "[{:>2}][{:>12}]: {}".format( level, tag, mssg ) )
        self.num_mssgs += 1
    

def log_callback( level, tag, mssg ):
    print( "[{:>2}][{:>12}]: {}".format( level, tag, mssg ) )


class TestContext:

    def test_optix_init( self ):
        tutil.optix_init()


    def test_context_options_ctor( self ):
        tutil.optix_init()
        
        ctx_options = optix.DeviceContextOptions()

        ctx_options = optix.DeviceContextOptions( logCallbackLevel = 2 )
        assert ctx_options.logCallbackLevel == 2

        logger = Logger()
        ctx_options = optix.DeviceContextOptions(
                logCallbackFunction = logger,
                logCallbackLevel    = 3
                )
        assert ctx_options.logCallbackFunction == logger
        assert ctx_options.logCallbackLevel    == 3


    def test_context_options_props( self ):
        tutil.optix_init()

        ctx_options = optix.DeviceContextOptions()
        ctx_options.logCallbackLevel = 1
        assert ctx_options.logCallbackLevel == 1

        ctx_options.logCallbackFunction = log_callback
        assert ctx_options.logCallbackFunction == log_callback 


    def test_create_destroy( self ):
        ctx = tutil.create_default_ctx();
        ctx.destroy()


    def test_get_property( self ):
        ctx = tutil.create_default_ctx();
        v = ctx.getProperty( optix.DEVICE_PROPERTY_LIMIT_NUM_BITS_INSTANCE_VISIBILITY_MASK )
        assert type( v ) is int
        assert v > 1 and v <= 16  # at time of writing, was 8
        ctx.destroy()

    
    def test_set_log_callback( self ):
        ctx = tutil.create_default_ctx();
        
        logger = Logger()
        ctx.setLogCallback( logger, 3 )
        ctx.setLogCallback( None, 2 )
        ctx.setLogCallback( log_callback, 1 )
        ctx.destroy()


    def test_set_get_cache( self ):
        ctx = tutil.create_default_ctx();

        ctx.setCacheEnabled( True )
        assert ctx.getCacheEnabled() == True
        ctx.setCacheEnabled( False )
        assert ctx.getCacheEnabled() == False

        db_sizes = ( 1024, 1024*1024 )
        ctx.setCacheDatabaseSizes( *db_sizes )
        assert ctx.getCacheDatabaseSizes() == db_sizes 

        v = ctx.getCacheLocation() 
        assert type(v) is str

        loc =  "/dev/null"
        with pytest.raises( RuntimeError ):
            ctx.setCacheLocation( loc ) # not valid dir

        ctx.destroy()



