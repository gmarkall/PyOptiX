from distutils.core import setup, Extension

module1 = Extension(
        'optix', 
        sources       = [ 'optixmodule.c' ],
        libraries     = [ 'optix' ],
        #include_dirs  = [ '/Users/kmorley/Code/rtsdk/rtmain/include' ],
        #library_dirs  = [ '/Users/kmorley/Code/rtsdk/rtmain/build_debug/lib' ],
        include_dirs  = [ '/Users/keith/Code/rtsdk/rtmain/include' ],
        library_dirs  = [ '/Users/keith/Code/rtsdk/rtmain/build_debug/lib' ],
        )

setup(
        name        = 'PackageName',
        version     = '1.0',
        description = 'This is a demo package',
        ext_modules = [module1]
        )
