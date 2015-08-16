from distutils.core import setup, Extension
import numpy

module1 = Extension(
        'optix', 
        sources       = [ 'PyOptiXModule.c' ],
        depends       = [ 'PyOptiXUtil.h', 'PyOptiXDecls.h' ],
        libraries     = [ 'optix' ],
        #include_dirs  = [ '/Users/kmorley/Code/rtsdk/rtmain/include' ],
        #library_dirs  = [ '/Users/kmorley/Code/rtsdk/rtmain/build_debug/lib' ],
        include_dirs  = [ '/Users/keith/Code/rtsdk/rtmain/include', numpy.get_include() ],
        library_dirs  = [ '/Users/keith/Code/rtsdk/rtmain/build_debug/lib' ],
        )

setup(
        name        = 'PackageName',
        version     = '1.0',
        description = 'This is a demo package',
        ext_modules = [module1]
        )
