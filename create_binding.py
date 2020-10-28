

import sys
import os
import re


re_opaque_type = re.compile( "^typedef\s+struct\s+[A-Za-z_]+\s+([A-Za-z]+)" )
re_opaque_type = re.compile( "^typedef\s+struct\s+[A-Za-z_]+" )#\s+([A-Za-z]+)" )

if len( sys.argv ) != 2:
    print( "Usage: {} <path/to/optix/include>".format( sys.argv[0] ) )
    sys.exit(0)

optix_include = sys.argv[1]
print( "Looking for optix headers in '{}'".format( optix_include ) )

with open( os.path.join( optix_include, 'optix_7_types.h' ), 'r' ) as types_file:
    print( "Found optix types header ..." )
    for line in types_file:
        match = re_opaque_type.match( line )
        if match:
            print( "Opaque type: {}".format( line ) )
            print( "Opaque type: {}".format( match.groups() ) )



