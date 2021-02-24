

import optix
import cupy as cp
import tutil 

import array 
import pytest 


class TestModule:

    def test_compile_bound_value_entry( self ):
        bound_value = array.array( 'f', [0.1, 0.2, 0.3] )
        bound_value_entry = optix.ModuleCompileBoundValueEntry(
            pipelineParamOffsetInBytes = 4,
            boundValue  = bound_value,
            annotation  = "my_bound_value"
            )

        assert bound_value_entry.pipelineParamOffsetInBytes == 4
        with pytest.raises( AttributeError ):
            print( bound_value_entry.boundValue )
        assert bound_value_entry.annotation == "my_bound_value" 

        bound_value_entry.pipelineParamOffsetInBytes = 8
        assert bound_value_entry.pipelineParamOffsetInBytes == 8
        bound_value_entry.annotation = "new_bound_value" 
        assert bound_value_entry.annotation == "new_bound_value" 


    def test_options( self ):
        mod_opts = optix.ModuleCompileOptions(
            maxRegisterCount = 64,
            optLevel         = optix.COMPILE_OPTIMIZATION_DEFAULT,
            debugLevel       = optix.COMPILE_DEBUG_LEVEL_LINEINFO,
            boundValues      = [] 
        )
        assert mod_opts.maxRegisterCount == 64
        assert mod_opts.optLevel         == optix.COMPILE_OPTIMIZATION_DEFAULT
        assert mod_opts.debugLevel       == optix.COMPILE_DEBUG_LEVEL_LINEINFO
        # optix.ModuleCompileOptions.boundValues is write-only
        with pytest.raises( AttributeError ):
            print( mod_opts.boundValues )

        mod_opts = optix.ModuleCompileOptions()
        mod_opts.maxRegisterCount = optix.COMPILE_DEFAULT_MAX_REGISTER_COUNT
        mod_opts.optLevel         = optix.COMPILE_OPTIMIZATION_LEVEL_1
        mod_opts.debugLevel       = optix.COMPILE_DEBUG_LEVEL_DEFAULT
        mod_opts.boundValues = [ optix.ModuleCompileBoundValueEntry() ];
        assert mod_opts.maxRegisterCount == optix.COMPILE_DEFAULT_MAX_REGISTER_COUNT
        assert mod_opts.optLevel         == optix.COMPILE_OPTIMIZATION_LEVEL_1
        assert mod_opts.debugLevel       == optix.COMPILE_DEBUG_LEVEL_DEFAULT


    def test_create_destroy( self ):
        ctx = tutil.create_default_ctx();

        module_opts   = optix.ModuleCompileOptions()
        pipeline_opts = optix.PipelineCompileOptions()
        mod, log = ctx.moduleCreateFromPTX(
            module_opts,
            pipeline_opts,
            tutil.ptx_string,
            )
        assert type(mod) is optix.Module 
        assert type(log) is str

        mod.destroy()
        ctx.destroy()


    def test_builtin_is_module_get():
        ctx = tutil.create_default_ctx();
        module_opts     = optix.ModuleCompileOptions()
        pipeline_opts   = optix.PipelineCompileOptions()
        builtin_is_opts = optix.BuiltinISOptions()

        is_mod = ctx.builtinISModuleGet(
                module_opts,
                pipeline_opts,
                builtin_is_opts
                )
        assert type( is_mod ) is optix.Module
        is_mod.destroy()
        ctx.destroy()

      

