using Ferrite: Ferrite, Vec, get_node_coordinate, getcells, getcoordinates, getfacetset

@testset "Grid and coordinates" begin
    gp = GridParams(0.3, 0.2, 0.0105, 5, 4, 3)
    grid = setup_grid(gp)
    coords = [get_node_coordinate(n) for n in grid.nodes]

    @testset "nodes lie exactly on the grid lines, x fastest" begin
        xs, ys, zs = range(-0.15, 0.15, 6), range(-0.1, 0.1, 5), range(0, 0.0105, 4)
        @test coords == vec([Vec((x, y, z)) for x in xs, y in ys, z in zs])
    end

    @testset "top face is z = 0, bottom face is z = lz" begin
        face_z(name) = unique(coords[n][3] for (cell, facet) in getfacetset(grid, name)
                                           for n in Ferrite.facets(getcells(grid, cell))[facet])
        @test face_z("top") == [0.0]
        @test face_z("bottom") == [gp.lz]
    end

    @testset "create_coordinate_grids gives the cell centres" begin
        centres = [sum(getcoordinates(grid, i)) / 8 for i in eachindex(grid.cells)]
        for (d, A) in enumerate(create_coordinate_grids(gp))
            @test size(A) == (gp.nx, gp.ny, gp.nz)
            @test vec(A) ≈ [c[d] for c in centres]
        end
    end

    @testset "setup_material_properties" begin
        materials = Dict(1 => (sigma = 0.1, epsilon_re = 1e-10, epsilon_im = 0.0, VHC = 1e6, k = 0.5),
                         2 => (sigma = 0.2, epsilon_re = 2e-8, epsilon_im = 1e-9, VHC = 2e6, k = 0.6))
        sigma, epsilon_re, epsilon_im, VHC, k = setup_material_properties([1 2; 2 1], materials)
        @test sigma == [0.1 0.2; 0.2 0.1]
        @test epsilon_re == [1e-10 2e-8; 2e-8 1e-10]
        @test epsilon_im == [0.0 1e-9; 1e-9 0.0]
        @test VHC == [1e6 2e6; 2e6 1e6]
        @test k == [0.5 0.6; 0.6 0.5]
    end
end
