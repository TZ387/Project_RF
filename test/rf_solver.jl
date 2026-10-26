using Ferrite: get_node_coordinate

@testset "RF solver" begin
    @testset "layered plates match the analytical solution" begin
        # Four layers between an electrode on top and ground at the bottom, in series like
        # impedances L/γ with γ = σ + ωε0(ε'' − iε'). The potential is linear within each layer, so
        # linear elements reproduce it at the nodes.
        σ = [0.3, 0.22, 0.025, 0.5]
        ε_re = [80, 1833, 27.2, 1836]  # relative permittivity
        ε_im = [0, 0, 500, 0]
        ω = 2π * 1e6
        γ = σ .+ ω .* ε0 .* (ε_im .- im .* ε_re)
        L = [0.5e-3, 1.5e-3, 3.5e-3, 5.0e-3]
        gp = GridParams(0.01, 0.01, sum(L), 3, 3, 21)  # layer interfaces fall on nodes
        layer = [findfirst(>(z), cumsum(L)) for z in create_coordinate_grids(gp)[3]]
        bc = Dict("top" => (x, t) -> 10.0, "bottom" => (x, t) -> 0.0)
        _, _, Qel, E_mag, E_vec, V = @test_logs min_level = Logging.Warn run_simulation(gp, RFParams(σ[layer], ε_re[layer], ε_im[layer], ω), bc)

        J = 10 / sum(L ./ γ)  # current density
        depth = [0; cumsum(L)]
        V_exact = [10 - J * sum(clamp(z - depth[n], 0, L[n]) / γ[n] for n in 1:4) for z in range(0, gp.lz, gp.nz + 1)]
        @test all(V[:, :, 1] .== 10) && all(V[:, :, end] .== 0)
        @test V ≈ repeat(reshape(V_exact, 1, 1, :), gp.nx + 1, gp.ny + 1) atol = 1e-6
        @test Qel ≈ 0.5 .* real.(γ[layer]) .* abs2.(J ./ γ[layer]) rtol = 1e-6
        @test all(>(0), real.(E_vec[3, :, :, :]))  # E points from the electrode into the tissue
        @test maximum(abs, E_vec[1:2, :, :, :]) < 1e-6 * maximum(E_mag)
    end

    @testset "dielectric loss adds to Qel, ε' does not" begin
        gp = GridParams(0.01, 0.01, 0.01, 2, 2, 4)
        σ, ε_re, ε_im, ω = 0.2, 1e4, 1e3, 1e6
        rf = RFParams(fill(σ, 2, 2, 4), fill(ε_re, 2, 2, 4), fill(ε_im, 2, 2, 4), ω)
        _, _, Qel, _, _, _ = @test_logs min_level = Logging.Warn run_simulation(gp, rf, Dict("top" => (x, t) -> 5.0, "bottom" => (x, t) -> 0.0))
        @test Qel ≈ fill(0.5 * (σ + ω * ε0 * ε_im) * (5.0 / gp.lz)^2, 2, 2, 4) rtol = 1e-6
    end

    @testset "electrode on part of a face, as f(x) and f(x, t)" begin
        gp = GridParams(0.09, 0.09, 0.05, 18, 18, 10)  # cubic cells, 5 mm
        rf = RFParams(fill(0.3, 18, 18, 10), zeros(18, 18, 10), zeros(18, 18, 10), 1e6)
        on_electrode(x) = abs(x[1]) <= 0.02 && abs(x[2]) <= 0.02
        for electrode in (x -> on_electrode(x) ? 100.0 : nothing, (x, t) -> on_electrode(x) ? 100.0 : nothing)
            grid, _, _, _, _, V = @test_logs min_level = Logging.Warn run_simulation(gp, rf, Dict("top" => electrode, "bottom" => x -> 0.0))
            coords = [get_node_coordinate(n) for n in grid.nodes]
            top = [x[3] == 0 for x in coords]
            held = top .& on_electrode.(coords)
            @test count(held) == 81
            @test all(V[held] .== 100)
            @test all(0 .< real.(V[top .& .!held]) .< 100)  # the rest of the face is insulating, not at 0 V
            @test V ≈ reverse(V, dims = 1) && V ≈ reverse(V, dims = 2)
        end
    end
end
