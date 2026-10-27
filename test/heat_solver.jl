@testset "Heat solver" begin
    gp = GridParams(0.01, 0.02, 0.005, 4, 5, 6)
    dims = (gp.nx, gp.ny, gp.nz)

    @testset "stable time step" begin
        VHC, k = 2e6, 0.5
        hp = HeatParams([HeatPhase(duration = 1.0)], 0, fill(37.0, dims), fill(VHC, dims), fill(k, dims))
        dx, dy, dz = gp.lx / gp.nx, gp.ly / gp.ny, gp.lz / gp.nz
        # Interior cells have six neighbours and set the limit
        @test compute_stable_dt(hp, gp) ≈ 0.9 * VHC / (2k * (1 / dx^2 + 1 / dy^2 + 1 / dz^2))
    end

    @testset "uniform source heats uniformly, only while on" begin
        Q, VHC = 3e5, 2e6
        hp = HeatParams([HeatPhase(duration = 2.0, rf_factor = 0.5, Q_extra = 1e5), HeatPhase(duration = 3.0)],
                        0, fill(37.0, dims), fill(VHC, dims), fill(0.5, dims))
        T = @test_logs min_level = Logging.Warn run_heat_simulation(fill(Q, dims), gp, hp)
        @test T ≈ fill(37.0 + (0.5Q + 1e5) * 2.0 / VHC, dims)
    end

    @testset "insulated boundaries conserve energy" begin
        VHC = [1e6 * (1 + (i + 2j + 3l) % 4) for i in 1:4, j in 1:5, l in 1:6]
        k = [0.2 + 0.1 * ((2i + j + l) % 5) for i in 1:4, j in 1:5, l in 1:6]
        Q = [1e5 * ((i * j + l) % 3) for i in 1:4, j in 1:5, l in 1:6]
        no_left_edge = [i > 1 for i in 1:4, j in 1:5, l in 1:6]
        Q_extra = [5e4 * ((i + l) % 2) for i in 1:4, j in 1:5, l in 1:6]
        hp = HeatParams([HeatPhase(duration = 1.5, rf_factor = 1.0), HeatPhase(duration = 2.0),
                         HeatPhase(duration = 0.5, rf_factor = no_left_edge, Q_extra = Q_extra)], 0, fill(20.0, dims), VHC, k)
        T = @test_logs min_level = Logging.Warn run_heat_simulation(Q, gp, hp)
        @test sum(VHC .* (T .- 20.0)) ≈ sum(Q) * 1.5 + (sum(Q[2:end, :, :]) + sum(Q_extra)) * 0.5 rtol = 1e-10
    end

    @testset "cosine mode follows the exact explicit-Euler solution" begin
        # φ is an eigenvector of the insulated difference operator with eigenvalue -Λ, so with Q = S·φ the
        # rise stays a·φ, where a takes explicit Euler steps of VHC·da/dt = S - kΛa
        X, Y, Z = create_coordinate_grids(gp)
        φ = cos.(π .* (X ./ gp.lx .+ 0.5)) .* cos.(π .* (Y ./ gp.ly .+ 0.5)) .* cos.(π .* Z ./ gp.lz)
        Λ = sum(4 / (l / n)^2 * sin(π / (2n))^2 for (l, n) in ((gp.lx, gp.nx), (gp.ly, gp.ny), (gp.lz, gp.nz)))
        S, VHC, k, t = 1e5, 2e6, 0.5, 3.0
        hp = HeatParams([HeatPhase(duration = t, rf_factor = φ)], 0, fill(37.0, dims), fill(VHC, dims), fill(k, dims))
        n = ceil(Int, t / compute_stable_dt(hp, gp))
        a = 0.0
        for _ in 1:n
            a += t / n / VHC * (S - k * Λ * a)
        end
        T = @test_logs min_level = Logging.Warn run_heat_simulation(fill(S, dims), gp, hp)
        @test T .- 37.0 ≈ a .* φ
    end
end
