using GLMakie: GLMakie

GLMakie.activate!(visible = false)

@testset "plot_graphs saves all plots" begin
    gp = GridParams(0.09, 0.09, 0.05, 9, 9, 5)
    material_indices = [1 + (l > 2) for i in 1:9, j in 1:9, l in 1:5]
    sigma = [0.2, 0.5][material_indices]
    rf = RFParams(sigma, fill(1e4, size(sigma)), zeros(size(sigma)), 1e6)  # complex V and E
    bc = Dict("top" => (x, t) -> abs(x[1]) <= 0.02 ? 100.0 : nothing, "bottom" => (x, t) -> 0.0)
    _, _, Qel, E_mag, E_vec, V = @test_logs min_level = Logging.Warn run_simulation(gp, rf, bc)
    mktempdir() do dir
        plot_graphs(material_indices, gp, Qel, E_mag, E_vec, V, joinpath(dir, "plot"))
        @test all(isfile(joinpath(dir, "plot_$c.png")) for c in 'a':'e')
    end
end
