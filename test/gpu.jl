using CUDA: CUDA

@testset "GPU" begin
    if CUDA.functional()
        @testset "heat solver matches the CPU" begin
            gp = GridParams(0.013, 0.011, 0.007, 13, 11, 7)  # more cells than one GPU block
            idx = CartesianIndices((gp.nx, gp.ny, gp.nz))
            VHC = [1e6 * (1 + (i + 2j + 3l) % 4) for (i, j, l) in Tuple.(idx)]
            k = [0.2 + 0.1 * ((2i + j + l) % 5) for (i, j, l) in Tuple.(idx)]
            Q = [1e5 * ((i * j + l) % 3) for (i, j, l) in Tuple.(idx)]
            hp = HeatParams([(:on, 1.5), (:off, 2.0), (:on, 0.5)], 0, fill(20.0, size(Q)), VHC, k)
            T_gpu = @test_logs min_level = Logging.Warn run_heat_simulation(Q, gp, hp; use_gpu = true)
            @test T_gpu isa Array
            @test T_gpu ≈ run_heat_simulation(Q, gp, hp) rtol = 1e-12
        end

        @testset "RF solver matches the CPU" begin
            gp = GridParams(0.09, 0.09, 0.05, 18, 18, 10)
            sigma = [0.1 + 0.05 * l for i in 1:18, j in 1:18, l in 1:10]
            rf = RFParams(sigma, fill(1e-9, size(sigma)), 1e6)
            electrode(x, t) = abs(x[1]) <= 0.02 && abs(x[2]) <= 0.02 ? 100.0 * (1 + x[2] / 0.02) : nothing
            bc = Dict("top" => electrode, "bottom" => (x, t) -> 0.0)
            on_gpu = @test_logs min_level = Logging.Warn run_simulation(gp, rf, bc; use_gpu = true)
            for (gpu, cpu) in zip(on_gpu[2:end], run_simulation(gp, rf, bc)[2:end])  # V_dof, Qel, E_mag, E_vec, V
                @test gpu ≈ cpu rtol = 1e-6
            end
        end
    else
        @test_skip CUDA.functional()
    end
end
