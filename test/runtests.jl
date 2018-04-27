using UKF

@testset "UKF" begin

    # Generate some simulated data
    y = sin(0:0.1:100) * amplitude + randn(100) * noise_std

    weights = calc_weights(1e-3, 2, 0, 20)
    #...
end
