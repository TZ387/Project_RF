using Ferrite: get_node_coordinate
using GLMakie: GLMakie

const EXAMPLES = joinpath(pkgdir(prrf), "Examples")

GLMakie.activate!(visible = false)  # Examples 7 and 8 create a GLMakie.Screen themselves

# Runs an example script in its own module. The full-size solve, the plots and the file output
# are replaced by stubs that record what the example passes to them.
function run_example(file)
    m = Module()
    Core.eval(m, :(using prrf))
    Core.eval(m, quote
        const recorded = Dict{Symbol, Any}()
        run_simulation(gp, rf, bc) = (recorded[:rf] = (; gp, rf, bc); (nothing, nothing, zero(rf.sigma), nothing, nothing, nothing))
        run_heat_simulation(Qel, gp, hp; kwargs...) = (recorded[:heat] = hp; nothing)
        plot_graphs(args...) = nothing
        plot_slices(args...; kwargs...) = nothing
        display(args...) = nothing
        save_simulation(args...; kwargs...) = nothing
    end)
    Base.include(m, joinpath(EXAMPLES, file))
    return m
end

# Checks the recorded set-up, then solves the example's geometry and boundary conditions on a coarse grid
function test_example(m)
    (; gp, rf, bc) = m.recorded[:rf]
    @test size(rf.sigma) == size(rf.epsilon_im) == (gp.nx, gp.ny, gp.nz)
    @test all(>(0), rf.sigma)
    @test keys(bc) ⊆ ("top", "bottom", "left", "right", "front", "back")

    coarse = GridParams(gp.lx, gp.ly, gp.lz, cld(gp.nx, 10), cld(gp.ny, 10), cld(gp.nz, 5))
    sigma, epsilon_im, VHC, k = setup_material_properties(m.geometryDefinition(coarse), m.define_material_properties())
    grid, _, Qel, _, _, V = @test_logs min_level = Logging.Warn run_simulation(coarse, RFParams(sigma, epsilon_im, rf.ω), bc)
    @test all(isfinite, V) && all(>=(0), Qel)

    # Every node where a boundary condition gives a value holds that value
    coords = [get_node_coordinate(n) for n in grid.nodes]
    on_face = Dict("top" => x -> x[3] == 0, "bottom" => x -> x[3] == gp.lz,
                   "left" => x -> x[1] == -gp.lx / 2, "right" => x -> x[1] == gp.lx / 2,
                   "front" => x -> x[2] == -gp.ly / 2, "back" => x -> x[2] == gp.ly / 2)
    for (face, f) in bc
        nodes = filter(n -> on_face[face](coords[n]) && f(coords[n], 0.0) !== nothing, eachindex(coords))
        @test !isempty(nodes) && V[nodes] == [f(coords[n], 0.0) for n in nodes]
    end

    # With uniform electrodes on top and bottom only and tissue layered in z (Example 1), the cell
    # layers act as resistors in series and the solution is known exactly
    σ = sigma .+ rf.ω .* epsilon_im
    uniform(face, z) = haskey(bc, face) && allequal(bc[face](x, 0.0) for x in coords if x[3] == z)
    if length(bc) == 2 && uniform("top", 0) && uniform("bottom", gp.lz) && all(σ .== σ[1:1, 1:1, :])
        V_top, V_bottom = bc["top"](coords[1], 0.0), bc["bottom"](coords[end], 0.0)
        R = (coarse.lz / coarse.nz) ./ σ[1, 1, :]
        J = (V_top - V_bottom) / sum(R)
        @test V ≈ repeat(reshape(V_top .- J .* [0; cumsum(R)], 1, 1, :), coarse.nx + 1, coarse.ny + 1)
        @test Qel ≈ repeat(reshape(0.5 * J^2 ./ σ[1, 1, :], 1, 1, :), coarse.nx, coarse.ny)
    end

    # Insulated boundaries: all heat deposited while the source is on is stored
    if haskey(m.recorded, :heat)
        hp = m.recorded[:heat]
        T0 = first(hp.T_initial)
        @test all(==(T0), hp.T_initial)
        T = @test_logs min_level = Logging.Warn run_heat_simulation(Qel, coarse, HeatParams(hp.schedule, 0, fill(T0, size(Qel)), VHC, k))
        t_on = sum(duration for (state, duration) in hp.schedule if state == :on)
        @test sum(VHC .* (T .- T0)) ≈ sum(Qel) * t_on rtol = 1e-9
    end
end

@testset "Examples" begin
    @testset "$file" for file in filter(endswith(".jl"), readdir(EXAMPLES))
        m = run_example(file)
        Base.invokelatest(test_example, m)  # the example's functions are newer than this code
    end
end
