
import optix
import cupy

print( "Initializing cuda ..." )
cupy.array( [0, 1] )

print( "Initializing optix ..." )
optix.init()

cu_ctx = optix.cuda.Context()
dco    = optix.DeviceContextOptions()

print( "Creating optix device context ..." )
ctx    = optix.deviceContextCreate( cu_ctx, dco )
print( "\t{}".format( ctx ) )
