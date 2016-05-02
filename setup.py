from distutils.core import setup, Extension


optix_path = '/home/kmorley/Code/rtsdk/rtmain/'
module1 = Extension(
        'optix', 
        sources       = [ 'PyOptiXModule.c' ],
        depends       = [ 'PyOptiXUtil.h', 'PyOptiXDecls.h' ],
        libraries     = [ 'optix' ],
        include_dirs  = [ '{}/include'.format( optix_path ) ],
        library_dirs  = [ '{}/build_debug/lib'.format( optix_path ) ],
        extra_compile_args = ['-std=c99', '-g']
        )

setup(
        name        = 'PackageName',
        version     = '1.0',
        description = 'This is a demo package',
        ext_modules = [module1]
        )
