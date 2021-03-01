@testset "Augmented Sigma points" begin
    @testset "Create sigma points" begin
        weight_params = ScaledSetWeightingParameters(0.5, 2, 1)
        A = [4.0 0.0; 0.0 4.0]
        Q = [4.0 0.0; 0.0 4.0]
        x = randn(2)
        χ = @inferred KalmanFilter.calc_sigma_points(x, KalmanFilter.Augmented(A, Q), weight_params)
        weight = KalmanFilter.calc_cholesky_weight(weight_params, 4)
        @test χ.x0 == x
        @test χ.P_chol == cholesky(A .* weight).L

        P_chol_temp = zeros(2,2)
        Q_chol_temp = zeros(2,2)
        χ = @inferred KalmanFilter.calc_sigma_points!(
            KalmanFilter.Augmented(P_chol_temp, Q_chol_temp),
            x,
            KalmanFilter.Augmented(A, Q),
            weight_params
        )
        @test χ.x0 == x
        @test χ.P_chol == cholesky(A .* weight).L
    end

    @testset "Covariance" begin
        weight_params = ScaledSetWeightingParameters(0.5, 2, 1)
        𝓨 = KalmanFilter.TransformedSigmaPoints(ones(5), hcat(ones(5,5) .* 4, ones(5,5) .* 2), weight_params)
        noise = Diagonal(ones(5))
        P = @inferred KalmanFilter.cov(𝓨, Augment(noise))
        @test P == -16.25 .* ones(5) * ones(5)' .+ 2 .* ones(5,5) * ones(5,5)' .* 16 .+
            2 .* ones(5,5) * ones(5,5)' .* 4

        P = zeros(5,5)
        P = @inferred KalmanFilter.cov!(P, 𝓨, Augment(noise))
        @test P == -16.25 .* ones(5) * ones(5)' .+ 2 .* ones(5,5) * ones(5,5)' .* 16 .+
            2 .* ones(5,5) * ones(5,5)' .* 4

        P_chol = LowerTriangular(ones(5,5))
        x = ones(5) .* 4
        𝓨 = KalmanFilter.TransformedSigmaPoints(ones(5), hcat(ones(5,5) .* 4, ones(5,5) .* 2, ones(5,5) .* 3, ones(5,5) .* 1), weight_params)
        χ = KalmanFilter.AugmentedSigmaPoints(x, P_chol, LowerTriangular(collect(noise)), weight_params)
        P = @inferred KalmanFilter.cov(χ, 𝓨)
        weight_0, weight_i = KalmanFilter.calc_cov_weights(𝓨)
        @test P == weight_i .* LowerTriangular(ones(5,5)) * ones(5,5)' .* 4 .-
#            weight_i .* noise * ones(5,5)' .* 2 .-
            weight_i .* LowerTriangular(ones(5,5)) * ones(5,5)' .* 3
#            weight_i .* noise * ones(5,5)' .* 1

        P = zeros(5,5)
        P = @inferred KalmanFilter.cov!(P, χ, 𝓨)
        @test P == weight_i .* LowerTriangular(ones(5,5)) * ones(5,5)' .* 4 .-
#            weight_i .* noise * ones(5,5)' .* 2 .-
            weight_i .* LowerTriangular(ones(5,5)) * ones(5,5)' .* 3
#            weight_i .* noise * ones(5,5)' .* 1
    end

    @testset "Transform sigma points" begin
        P = [2.0 0.0; 0.0 2.0]
        noise = collect(Diagonal(ones(2) .* 3))
        x = [1.0, 1.0]
        weight_params = ScaledSetWeightingParameters(0.5, 2, 1)
        χ = KalmanFilter.AugmentedSigmaPoints(x, LowerTriangular(P), LowerTriangular(noise), weight_params)
        f(x) = x .* 2
        f(x, n) = x .* 2 .+ n
        𝓨 = @inferred KalmanFilter.transform(f, χ)
        @test 𝓨 == [ones(2) .* 2 [6 2; 2 6] x .* 2 .+ noise [-2 2; 2 -2] x .* 2 .- noise]
        @test 𝓨.weight_params == weight_params

        other_weight_params = MeanSetWeightingParameters(0.5)
        𝓨_temp = KalmanFilter.TransformedSigmaPoints(zeros(2), zeros(2,8), other_weight_params)
        F!(y, x) = y .= x .* 2
        F!(y, x, n) = y .= x .* 2 .+ n
        xi_temp = KalmanFilter.Augmented(zeros(length(x)), zeros(length(x)))
        𝓨 = @inferred KalmanFilter.transform!(𝓨_temp, xi_temp, F!, χ)
        @test 𝓨 == [ones(2) .* 2 [6 2; 2 6] x .* 2 .+ noise [-2 2; 2 -2] x .* 2 .- noise]
        @test 𝓨.weight_params == weight_params
    end
end