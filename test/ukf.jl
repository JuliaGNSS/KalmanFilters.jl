@testset "UKF weighting parameters" begin
    num_states = 5

    weight_params = WanMerweWeightingParameters(0.5, 2, 0)
    @test @inferred(KalmanFilter.lambda(weight_params, num_states)) == -3.75
    @test @inferred(KalmanFilter.calc_mean_weights(weight_params, num_states)) == (-3, 0.4)
    @test @inferred(KalmanFilter.calc_cov_weights(weight_params, num_states)) == (-0.25, 0.4)
    @test @inferred(KalmanFilter.calc_cholesky_weight(weight_params, num_states)) == 1.25

    weight_params = MeanSetWeightingParameters(0.5)
    @test @inferred(KalmanFilter.calc_mean_weights(weight_params, num_states)) == (0.5, 0.05)
    @test @inferred(KalmanFilter.calc_cov_weights(weight_params, num_states)) == (0.5, 0.05)
    @test @inferred(KalmanFilter.calc_cholesky_weight(weight_params, num_states)) == 10

    weight_params = GaussSetWeightingParameters(3)
    @test all(@inferred(KalmanFilter.calc_mean_weights(weight_params, num_states)) .≈ (-2/3, 1/6))
    @test all(@inferred(KalmanFilter.calc_cov_weights(weight_params, num_states)) .≈ (-2/3, 1/6))
    @test @inferred(KalmanFilter.calc_cholesky_weight(weight_params, num_states)) == 3

    weight_params = ScaledSetWeightingParameters(0.5, 2, 1)
    @test @inferred(KalmanFilter.calc_mean_weights(weight_params, num_states)) == (-19, 2)
    @test @inferred(KalmanFilter.calc_cov_weights(weight_params, num_states)) == (-16.25, 2)
    @test @inferred(KalmanFilter.calc_cholesky_weight(weight_params, num_states)) == 0.25
end

@testset "UKF weighted means" begin
    χ = KalmanFilter.SigmaPoints(ones(5), ones(5,5) .* 4, ones(5,5) .* 2)
    weight_params = ScaledSetWeightingParameters(0.5, 2, 1)
    x = @inferred KalmanFilter.mean(χ, weight_params)
    @test x == ones(5) .* -19 .+ ones(5) .* 20 .* 2 .+ ones(5) .* 10 .* 2

    χ = KalmanFilter.SigmaPoints(ones(5), ones(5,5) .* 4, ones(5,5) .* 2)
    weight_params = ScaledSetWeightingParameters(0.5, 2, 1)
    x = @inferred KalmanFilter.mean!(x, χ, weight_params)
    @test x == ones(5) .* -19 .+ ones(5) .* 20 .* 2 .+ ones(5) .* 10 .* 2

    x = zeros(5)
    x = @inferred KalmanFilter._mean!(x, χ, 10)
    @test x == ones(5) .* 20 .* 10 .+ ones(5) .* 10 .* 10
end

@testset "Create pseudo Sigma points" begin
    weighted_P_chol = LowerTriangular(ones(5,5))
    χ_diff_x_pseudo = KalmanFilter.create_pseudo_sigmapoints(weighted_P_chol)
    @test χ_diff_x_pseudo.xi_P_plus == weighted_P_chol
    @test χ_diff_x_pseudo.xi_P_minus == -weighted_P_chol

    χ_diff_x_pseudo = KalmanFilter.create_pseudo_sigmapoints!(χ_diff_x_pseudo, weighted_P_chol)
    @test χ_diff_x_pseudo.xi_P_plus == weighted_P_chol
    @test χ_diff_x_pseudo.xi_P_minus == -weighted_P_chol
end

@testset "UKF covariance" begin
    χ_diff_x = KalmanFilter.SigmaPoints(ones(5), ones(5,5) .* 4, ones(5,5) .* 2)
    weight_params = ScaledSetWeightingParameters(0.5, 2, 1)
    noise = Diagonal(ones(5))
    P = @inferred KalmanFilter.cov(χ_diff_x, noise, weight_params)
    @test P == -16.25 .* ones(5) * ones(5)' .+ 2 .* ones(5,5) * ones(5,5)' .* 16 .+
        2 .* ones(5,5) * ones(5,5)' .* 4 .+ noise

    P = @inferred KalmanFilter.cov!(P, χ_diff_x, noise, weight_params)
    @test P == -16.25 .* ones(5) * ones(5)' .+ 2 .* ones(5,5) * ones(5,5)' .* 16 .+
        2 .* ones(5,5) * ones(5,5)' .* 4 .+ noise

    P_chol = LowerTriangular(ones(5,5))
    χ_diff_x_pseudo = KalmanFilter.PseudoSigmaPoints(P_chol)
    P = @inferred KalmanFilter.cov(χ_diff_x_pseudo, χ_diff_x, weight_params)
    @test P == 2 .* LowerTriangular(ones(5,5)) * ones(5,5)' .* 4 .+
        2 .* LowerTriangular(-ones(5,5)) * ones(5,5)' .* 2

    P = @inferred KalmanFilter.cov!(P, χ_diff_x_pseudo, χ_diff_x, weight_params)
    @test P == 2 .* LowerTriangular(ones(5,5)) * ones(5,5)' .* 4 .+
        2 .* LowerTriangular(-ones(5,5)) * ones(5,5)' .* 2

    𝓨_diff_y = KalmanFilter.SigmaPoints(ones(4), ones(4,5) .* 4, ones(4,5) .* 2)

    P = @inferred KalmanFilter.cov(χ_diff_x, 𝓨_diff_y, weight_params)
    @test P == -16.25 .* ones(5) * ones(4)' .+ 2 .* ones(5,5) * ones(4,5)' .* 16 .+
        2 .* ones(5,5) * ones(4,5)' .* 4

    P = @inferred KalmanFilter.cov!(P, χ_diff_x, 𝓨_diff_y, weight_params)
    @test P == -16.25 .* ones(5) * ones(4)' .+ 2 .* ones(5,5) * ones(4,5)' .* 16 .+
        2 .* ones(5,5) * ones(4,5)' .* 4

    P = zeros(5,4)
    P = @inferred KalmanFilter._cov!(P, χ_diff_x, 𝓨_diff_y, 4)
    @test P == 4 .* ones(5,5) * ones(4,5)' .* 16 .+
        4 .* ones(5,5) * ones(4,5)' .* 4
end

@testset "Weighted cholesky" begin
    weight_params = ScaledSetWeightingParameters(0.5, 2, 1)
    A = [4 0; 0 4]
    P_chol = @inferred KalmanFilter.calc_lower_triangle_cholesky(A, weight_params)
    @test P_chol == cholesky(A).L .* 0.5

    dest = LowerTriangular(zeros(2,2))
    P_chol = @inferred KalmanFilter.calc_lower_triangle_cholesky!(dest, A, weight_params)
    @test P_chol == cholesky(A).L .* 0.5
end

@testset "Apply func to sigma points" begin
    weighted_P_chol = LowerTriangular([2 0; 0 2])
    x = [1, 1]
    F(x) = x .* 2
    χ = @inferred KalmanFilter.apply_func_to_sigma_points(F, x, weighted_P_chol)
    @test χ == [ones(2) .* 2 [6 2; 2 6] [-2 2; 2 -2]]

    F!(x, y) = x .= y .* 2
    χ = @inferred KalmanFilter.apply_func_to_sigma_points!(χ, F!, x, weighted_P_chol)
    @test χ == [ones(2) .* 2 [6 2; 2 6] [-2 2; 2 -2]]
end

@testset "UKF time update" begin
    x = [1., 1.]
    P = [1. 0.; 0. 1.]
    Q = [1. 0.; 0. 1.]

    F(x) = x .* [1., 2.]
    F!(x, y) = x .= y .* [1., 2.]

    tu = time_update(x, P, F, Q)
    @test state(tu) ≈ [1., 2.]
    @test covariance(tu) ≈ [2. 0.; 0. 5.]

    tu_inter = UKFTUIntermediate(2)
    tu = time_update!(tu_inter, x, P, F!, Q)
    @test state(tu) ≈ [1., 2.]
    @test covariance(tu) ≈ [2. 0.; 0. 5.]

end

@testset "KF measurement update" begin

    y = [1., 1.]
    x = [1., 1.]
    P = [1. 0.; 0. 1.]
    R = [1. 0.; 0. 1.]

    H(x) = x .* [1., 1.]
    H!(x, y) = x .= y .* [1., 1.]

    mu = measurement_update(x, P, y, H, R)
    @test state(mu) ≈ [1., 1.]
    @test covariance(mu) ≈ [0.5 0.; 0. 0.5]
    @test innovation(mu) ≈ [0.0, 0.0] atol = 1e-10 #?
    @test innovation_covariance(mu) ≈ [2.0 0.0; 0.0 2.0]
    @test kalman_gain(mu) ≈ [0.5 0.0; 0.0 0.5]

    mu_inter = UKFMUIntermediate(2,2)
    mu = measurement_update!(mu_inter, x, P, y, H!, R)
    @test state(mu) ≈ [1., 1.]
    @test covariance(mu) ≈ [0.5 0.; 0. 0.5]
    @test innovation(mu) ≈ [0.0, 0.0] atol = 1e-10 #?
    @test innovation_covariance(mu) ≈ [2.0 0.0; 0.0 2.0]
    @test kalman_gain(mu) ≈ [0.5 0.0; 0.0 0.5]
end
