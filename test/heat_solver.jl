@testset "Heat solver" begin
    gp = GridParams(0.01, 0.02, 0.005, 4, 5, 6)
    dims = (gp.nx, gp.ny, gp.nz)

    @testset "stable time step" begin
        VHC, k = 2e6, 0.5
        hp = HeatParams([(:on, 1.0)], 0, fill(37.0, dims), fill(VHC, dims), fill(k, dims))
        dx, dy, dz = gp.lx / gp.nx, gp.ly / gp.ny, gp.lz / gp.nz
        # Interior cells have six neighbours and set the limit
        @test compute_stable_dt(hp, gp) ≈ 0.9 * VHC / (2k * (1 / dx^2 + 1 / dy^2 + 1 / dz^2))
    end

    @testset "uniform source heats uniformly, only while on" begin
        Q, VHC = 3e5, 2e6
        hp = HeatParams([(:on, 2.0), (:off, 3.0)], 0, fill(37.0, dims), fill(VHC, dims), fill(0.5, dims))
        T = @test_logs min_level = Logging.Warn run_heat_simulation(fill(Q, dims), gp, hp)
        @test T ≈ fill(37.0 + Q * 2.0 / VHC, dims)
    end

    @testset "insulated boundaries conserve energy" begin
        VHC = [1e6 * (1 + (i + 2j + 3l) % 4) for i in 1:4, j in 1:5, l in 1:6]
        k = [0.2 + 0.1 * ((2i + j + l) % 5) for i in 1:4, j in 1:5, l in 1:6]
        Q = [1e5 * ((i * j + l) % 3) for i in 1:4, j in 1:5, l in 1:6]
        hp = HeatParams([(:on, 1.5), (:off, 2.0), (:on, 0.5)], 0, fill(20.0, dims), VHC, k)
        T = @test_logs min_level = Logging.Warn run_heat_simulation(Q, gp, hp)
        @test sum(VHC .* (T .- 20.0)) ≈ sum(Q) * 2.0 rtol = 1e-10
    end

    @testset "unknown phase state" begin
        hp = HeatParams([(:heat, 1.0)], 0, fill(37.0, dims), fill(1e6, dims), fill(0.5, dims))
        @test_throws ErrorException run_heat_simulation(zeros(dims), gp, hp)
    end
end
