@testset "Sigma points" begin
    @testset "Weighting parameters" begin
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

    @testset "Create sigma points" begin
        weight_params = ScaledSetWeightingParameters(0.5, 2, 1)
        A = [4.0 0.0; 0.0 4.0]
        x = randn(2)
        χ = @inferred KalmanFilter.calc_sigma_points(x, A, weight_params)
        @test χ.x0 == x
        @test χ.P_chol == cholesky(A .* 0.25).L

        P_chol_temp = zeros(2,2)
        χ = @inferred KalmanFilter.calc_sigma_points!(P_chol_temp, x, A, weight_params)
        @test χ.x0 == x
        @test χ.P_chol == cholesky(A .* 0.25).L
    end

    @testset "Weighted means" begin
        weight_params = ScaledSetWeightingParameters(0.5, 2, 1)
        χ = KalmanFilter.TransformedSigmaPoints(ones(5), hcat(ones(5,5) .* 4, ones(5,5) .* 2), weight_params)
        x = @inferred KalmanFilter.mean(χ)
        @test x == ones(5) .* -19 .+ ones(5) .* 20 .* 2 .+ ones(5) .* 10 .* 2

        χ = KalmanFilter.TransformedSigmaPoints(ones(5), hcat(ones(5,5) .* 4, ones(5,5) .* 2), weight_params)
        x = zeros(5)
        x = @inferred KalmanFilter.mean!(x, χ)
        @test x == ones(5) .* -19 .+ ones(5) .* 20 .* 2 .+ ones(5) .* 10 .* 2
    end

    @testset "substract mean" begin
        weight_params = ScaledSetWeightingParameters(0.5, 2, 1)
        χ = KalmanFilter.TransformedSigmaPoints(ones(5), hcat(ones(5,5) .* 4, ones(5,5) .* 2), weight_params)
        x = ones(5) .* 2
        unbiased_χ = @inferred KalmanFilter.substract_mean(χ, x)
        @test unbiased_χ == [-ones(5) hcat(ones(5,5) .* 2, zeros(5,5))]
        @test unbiased_χ.weight_params == weight_params

        other_weight_params = MeanSetWeightingParameters(0.0)
        unbiased_χ_temp = KalmanFilter.TransformedSigmaPoints(randn(5), randn(5,10), other_weight_params)
        unbiased_χ = @inferred KalmanFilter.substract_mean!(unbiased_χ_temp, χ, x)
        @test unbiased_χ == [-ones(5) hcat(ones(5,5) .* 2, zeros(5,5))]
        @test unbiased_χ.weight_params == weight_params
    end

    @testset "Covariance" begin
        weight_params = ScaledSetWeightingParameters(0.5, 2, 1)
        𝓨 = KalmanFilter.TransformedSigmaPoints(ones(5), hcat(ones(5,5) .* 4, ones(5,5) .* 2), weight_params)
        noise = Diagonal(ones(5))
        P = @inferred KalmanFilter.cov(𝓨, noise)
        @test P == -16.25 .* ones(5) * ones(5)' .+ 2 .* ones(5,5) * ones(5,5)' .* 16 .+
            2 .* ones(5,5) * ones(5,5)' .* 4 .+ noise

        P = zeros(5,5)
        P = @inferred KalmanFilter.cov!(P, 𝓨, noise)
        @test P == -16.25 .* ones(5) * ones(5)' .+ 2 .* ones(5,5) * ones(5,5)' .* 16 .+
            2 .* ones(5,5) * ones(5,5)' .* 4 .+ noise

        P_chol = LowerTriangular(ones(5,5))
        x = ones(5) .* 4
        χ = KalmanFilter.SigmaPoints(x, P_chol, weight_params)
        P = @inferred KalmanFilter.cov(χ, 𝓨)
        @test P == 2 .* LowerTriangular(ones(5,5)) * ones(5,5)' .* 4 .+
            2 .* LowerTriangular(-ones(5,5)) * ones(5,5)' .* 2

        P = zeros(5,5)
        P = @inferred KalmanFilter.cov!(P, χ, 𝓨)
        @test P == 2 .* LowerTriangular(ones(5,5)) * ones(5,5)' .* 4 .+
            2 .* LowerTriangular(-ones(5,5)) * ones(5,5)' .* 2
    end

    @testset "Transform sigma points" begin
        P = [2.0 0.0; 0.0 2.0]
        x = [1.0, 1.0]
        weight_params = ScaledSetWeightingParameters(0.5, 2, 1)
        χ = KalmanFilter.SigmaPoints(x, LowerTriangular(P), weight_params)
        F(x) = x .* 2
        𝓨 = @inferred KalmanFilter.transform(F, χ)
        @test 𝓨 == [ones(2) .* 2 [6 2; 2 6] [-2 2; 2 -2]]
        @test 𝓨.weight_params == weight_params

        other_weight_params = MeanSetWeightingParameters(0.5)
        𝓨_temp = KalmanFilter.TransformedSigmaPoints(zeros(2), zeros(2,4), other_weight_params)
        F!(x, y) = x .= y .* 2
        xi_temp = zeros(length(x))
        𝓨 = @inferred KalmanFilter.transform!(𝓨_temp, xi_temp, F!, χ)
        @test 𝓨 == [ones(2) .* 2 [6 2; 2 6] [-2 2; 2 -2]]
        @test 𝓨.weight_params == weight_params
    end
end