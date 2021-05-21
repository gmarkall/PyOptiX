

import optix
import cupy as cp
import tutil 

import array 
import pytest 


class TestProgramGroup:
    def create_prog_group( self ):
        ctx, module = tutil.create_default_module()

        prog_group_opts = optix.ProgramGroupOptions()

        prog_group_desc                          = optix.ProgramGroupDesc()
        prog_group_desc.raygenModule             = module
        prog_group_desc.raygenEntryFunctionName  = "__raygen__hello"

        prog_groups, log = ctx.programGroupCreate( [ prog_group_desc ], prog_group_opts )
        return prog_groups[0]


    def test_options( self ):
        prog_group_opts = optix.ProgramGroupOptions()
        assert type(prog_group_opts) is optix.ProgramGroupOptions

        
    def test_desc( self ):
        def test_keyword_ctor():
            ctx, module = tutil.create_default_module()
            prog_group_desc = optix.ProgramGroupDesc(
                    raygenModule             = module,
                    raygenEntryFunctionName  = "__raygen__hello"
                    )
            assert prog_group_desc.raygenModule             == module
            assert prog_group_desc.raygenEntryFunctionName  == "__raygen__hello"
        test_keyword_ctor()

        def test_attributes():
            ctx, module = tutil.create_default_module()
            prog_group_desc                          = optix.ProgramGroupDesc()
            prog_group_desc.raygenModule             = module
            prog_group_desc.raygenEntryFunctionName  = "__raygen__hello"

            assert prog_group_desc.raygenModule             == module
            assert prog_group_desc.raygenEntryFunctionName  == "__raygen__hello"
        test_attributes()


    def test_create_raygen( self ):
        ctx, module = tutil.create_default_module()

        prog_group_opts = optix.ProgramGroupOptions()

        prog_group_desc                          = optix.ProgramGroupDesc()
        prog_group_desc.raygenModule             = module
        prog_group_desc.raygenEntryFunctionName  = "__raygen__hello"

        prog_groups, log = ctx.programGroupCreate( [ prog_group_desc ], prog_group_opts )
        assert len(prog_groups) == 1 
        assert type(prog_groups[0]) is optix.ProgramGroup

        prog_groups[0].destroy()

    
    def test_create_miss( self ):
        ctx, module = tutil.create_default_module()

        prog_group_opts = optix.ProgramGroupOptions()

        prog_group_desc                          = optix.ProgramGroupDesc()
        prog_group_desc.missModule             = module 
        prog_group_desc.missEntryFunctionName  = "__miss__noop"

        prog_groups, log = ctx.programGroupCreate( [ prog_group_desc ], prog_group_opts )
        assert len(prog_groups) == 1 
        assert type(prog_groups[0]) is optix.ProgramGroup

        prog_groups[0].destroy()
    

    def test_create_callables( self ):
        ctx, module = tutil.create_default_module()

        prog_group_opts = optix.ProgramGroupOptions()

        prog_group_desc                               = optix.ProgramGroupDesc()
        prog_group_desc.callablesModuleDC             = module 
        prog_group_desc.callablesModuleCC             = module 
        prog_group_desc.callablesEntryFunctionNameCC  = "__continuation_callable__noop"
        prog_group_desc.callablesEntryFunctionNameDC  = "__direct_callable__noop"

        prog_groups, log = ctx.programGroupCreate( [ prog_group_desc ], prog_group_opts )
        assert len(prog_groups) == 1 
        assert type(prog_groups[0]) is optix.ProgramGroup

        prog_groups[0].destroy()


    def test_create_hitgroup( self ):
        ctx, module = tutil.create_default_module()

        prog_group_opts = optix.ProgramGroupOptions()

        prog_group_desc                              = optix.ProgramGroupDesc()
        prog_group_desc.hitgroupModuleCH             = module 
        prog_group_desc.hitgroupModuleAH             = module 
        prog_group_desc.hitgroupModuleIS             = module 
        prog_group_desc.hitgroupEntryFunctionNameCH  = "__closesthit__noop"
        prog_group_desc.hitgroupEntryFunctionNameAH  = "__anyhit__noop"
        prog_group_desc.hitgroupEntryFunctionNameIS  = "__intersection__noop"

        prog_groups, log = ctx.programGroupCreate( [ prog_group_desc ], prog_group_opts )
        assert len(prog_groups) == 1 
        assert type(prog_groups[0]) is optix.ProgramGroup

        prog_groups[0].destroy()


    def test_get_stack_size( self ):
        prog_group = self.create_prog_group()
        stack_size = prog_group.getStackSize()
        assert type(stack_size) is optix.StackSizes 



