#!/usr/bin/env python

import optix


print 'Testing optix module methods ============================================='
print 'Available devices: {}'.format( optix.deviceGetDeviceCount() )
print 'OptiX version    : {}'.format( optix.getVersion() )

ctx = optix.contextCreate()

print 'OptiX ctx create : {}'.format( ctx )

print dir( optix.Context )
print 'Testing optix module methods'
print 'Testing context methods =================================================='
print 'ctx devices      : {}'.format( ctx.getDeviceCount() )
