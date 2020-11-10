
import optix
import cupy

class Logger:
    def __call__( self, level, tag, mssg ):
        print( "[{:>2}][{:>12}]: {}".format( level, tag, mssg ) )

def log_callback( level, tag, mssg ):
    print( "[{:>2}][{:>12}]: {}".format( level, tag, mssg ) )


print( "Initializing cuda ..." )
cupy.cuda.runtime.free( 0 )

print( "Initializing optix ..." )
optix.init()

print( "Creating optix device context ..." )
cu_ctx = optix.cuda.Context()
dco    = optix.DeviceContextOptions()
#dco.logCallbackFunction = log_callback 
logger = Logger()
dco.logCallbackFunction = logger
dco.logCallbackLevel    = 4

ctx    = optix.deviceContextCreate( cu_ctx, dco )
print( "\t{}".format( ctx ) )
