@testset "UKF weighting parameters" begin
    num_states = 5

    weight_params = WanMerveWeightingParameters(0.5, 2, 0)
    @test KalmanFilter.lambda(weight_params, num_states) == -3.75
    @test KalmanFilter.calc_mean_weights(weight_params, num_states) == (-3, 0.4)
    @test KalmanFilter.calc_cov_weights(weight_params, num_states) == (-0.25, 0.4)
    @test KalmanFilter.calc_cholesky_weight(weight_params, num_states) == sqrt(1.25)

    weight_params = MeanSetWeightingParameters(0.5)
    @test KalmanFilter.calc_mean_weights(weight_params, num_states) == (0.5, 0.05)
    @test KalmanFilter.calc_cov_weights(weight_params, num_states) == (0.5, 0.05)
    @test KalmanFilter.calc_cholesky_weight(weight_params, num_states) == sqrt(10)

    weight_params = GaussSetWeightingParameters(3)
    @test all(KalmanFilter.calc_mean_weights(weight_params, num_states) .≈ (-2/3, 1/6))
    @test all(KalmanFilter.calc_cov_weights(weight_params, num_states) .≈ (-2/3, 1/6))
    @test KalmanFilter.calc_cholesky_weight(weight_params, num_states) == sqrt(3)

    weight_params = ScaledSetWeightingParameters(0.5, 2, 1)
    @test KalmanFilter.calc_mean_weights(weight_params, num_states) == (-19, 2)
    @test KalmanFilter.calc_cov_weights(weight_params, num_states) == (-16.25, 2)
    @test KalmanFilter.calc_cholesky_weight(weight_params, num_states) == 0.5
end

@testset "Weighting" begin
    #weighted_mean(χ, weight_params)
end
