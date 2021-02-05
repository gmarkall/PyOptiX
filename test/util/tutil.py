

import optix
import cupy as cp


class Logger:
    def __init__( self ):
        self.num_mssgs = 0
    
    def __call__( self, level, tag, mssg ):
        print( "[{:>2}][{:>12}]: {}".format( level, tag, mssg ) )
        self.num_mssgs += 1
    

def log_callback( level, tag, mssg ):
    print( "[{:>2}][{:>12}]: {}".format( level, tag, mssg ) )


def optix_init():
    cp.cuda.runtime.free( 0 )
    optix.init()


def create_default_ctx():
    optix_init()
    ctx_options = optix.DeviceContextOptions()

    cu_ctx = 0 
    return optix.deviceContextCreate( cu_ctx, ctx_options )
