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
    else
        @test_skip CUDA.functional()
    end
end
