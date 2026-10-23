@testset "Simulation IO" begin
    mktempdir() do dir
        file = joinpath(dir, "simulation.h5")
        gp = GridParams(0.3, 0.2, 0.01, 4, 3, 2)
        material_indices = reshape([1 + i % 3 for i in 1:24], 4, 3, 2)
        Qel = reshape(collect(1.0:24.0), 4, 3, 2)
        Z = complex.(Qel, -Qel)
        redirect_stdout(devnull) do
            save_simulation(file, gp, material_indices; Qel, Z, frequency = 1e6, label = "test")
        end

        gp_loaded, indices_loaded, data = load_simulation(file)
        @test gp_loaded == gp
        @test indices_loaded == material_indices
        @test data.Qel == Qel
        @test data.Z == Z
        @test data.frequency == 1e6
        @test data.label == "test"
    end
end
