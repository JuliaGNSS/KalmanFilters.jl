@testset "Augmented Unscented Kalman filter" begin
    @testset "Weighted augmented cholesky" begin
        weight_params = ScaledSetWeightingParameters(0.5, 2, 1)
        A = KalmanFilter.Augmented([4 0; 0 4], [4 0; 0 4])
        P_chol = @inferred KalmanFilter.calc_weighted_cholesky(A, weight_params)
        @test P_chol.P == cholesky(A.P .* 0.25)
        @test P_chol.noise == cholesky(A.noise .* 0.25)

        dest = KalmanFilter.Augmented(Cholesky(zeros(2,2), 'L', 0), Cholesky(zeros(2,2), 'L', 0))
        P_chol = @inferred KalmanFilter.calc_weighted_cholesky!(dest, A, weight_params)
        @test P_chol.P == cholesky(A.P .* 0.25)
        @test P_chol.noise == cholesky(A.noise .* 0.25)
    end

    @testset "Apply func to augmented sigma points" begin
        weighted_P_chol = KalmanFilter.Augmented(Cholesky([2 0; 0 2], 'L', 0), Cholesky([3 0; 0 3], 'L', 0))
        x = [1, 1]
        F(x) = x .* 2
        F(x, noise) = x .* 2 .+ noise
        χ = @inferred KalmanFilter.apply_func_to_sigma_points(F, x, weighted_P_chol)
        @test χ == [ones(2) .* 2 [6 2; 2 6] [5 2; 2 5] [-2 2; 2 -2] [-1 2; 2 -1]]

        F!(x, y) = x .= y .* 2
        F!(x, y, noise) = x .= y .* 2 .+ noise
        χ = @inferred KalmanFilter.apply_func_to_sigma_points!(χ, F!, x, weighted_P_chol)
        @test χ == [ones(2) .* 2 [6 2; 2 6] [5 2; 2 5] [-2 2; 2 -2] [-1 2; 2 -1]]
    end

    @testset "Weighted means" begin
        χ = KalmanFilter.AugmentedSigmaPoints(ones(5), ones(5,5) .* 4, ones(5,5), ones(5,5) .* 2, ones(5,5) .* 3)
        weight_params = ScaledSetWeightingParameters(0.5, 2, 1)
        x = @inferred KalmanFilter.mean(χ, weight_params)
        @test x == ones(5) .* -39 .+ ones(5) .* 20 .* 2 .+ ones(5) .* 5 .* 2 .+ ones(5) .* 10 .* 2 .+ ones(5) .* 15 .* 2

        χ = KalmanFilter.AugmentedSigmaPoints(ones(5), ones(5,5) .* 4, ones(5,5), ones(5,5) .* 2, ones(5,5) .* 3)
        weight_params = ScaledSetWeightingParameters(0.5, 2, 1)
        x = @inferred KalmanFilter.mean!(x, χ, weight_params)
        @test x == ones(5) .* -39 .+ ones(5) .* 20 .* 2 .+ ones(5) .* 5 .* 2 .+ ones(5) .* 10 .* 2 .+ ones(5) .* 15 .* 2
    end

    @testset "Create augmented pseudo Sigma points" begin
        weighted_P_chol = KalmanFilter.Augmented(Cholesky(ones(5,5), 'L', 0), Cholesky(2 .* ones(5,5), 'L', 0))
        χ_diff_x_pseudo = KalmanFilter.create_pseudo_sigmapoints(weighted_P_chol)
        @test χ_diff_x_pseudo.xi_P_plus == LowerTriangular(ones(5,5))
        @test χ_diff_x_pseudo.xi_P_minus == -LowerTriangular(ones(5,5))

        χ_diff_x_pseudo = KalmanFilter.create_pseudo_sigmapoints!(χ_diff_x_pseudo, weighted_P_chol)
        @test χ_diff_x_pseudo.xi_P_plus == LowerTriangular(ones(5,5))
        @test χ_diff_x_pseudo.xi_P_minus == -LowerTriangular(ones(5,5))
    end

    @testset "Covariance" begin
        χ_diff_x = KalmanFilter.AugmentedSigmaPoints(ones(5), ones(5,5) .* 4, ones(5,5), ones(5,5) .* 2, ones(5,5) .* 3)
        weight_params = ScaledSetWeightingParameters(0.5, 2, 1)
        P = @inferred KalmanFilter.cov(χ_diff_x, nothing, weight_params)
        @test P == -36.25 .* ones(5) * ones(5)' .+ 2 .* ones(5,5) * ones(5,5)' .* 16 .+
            2 .* ones(5,5) * ones(5,5)' .+ 2 .* ones(5,5) * ones(5,5)' .* 4 .+
            2 .* ones(5,5) * ones(5,5)' .* 9

        P_dest = KalmanFilter.Augmented(P, ones(5,5))
        P = @inferred KalmanFilter.cov!(P_dest, χ_diff_x, nothing, weight_params)
        @test P == -36.25 .* ones(5) * ones(5)' .+ 2 .* ones(5,5) * ones(5,5)' .* 16 .+
            2 .* ones(5,5) * ones(5,5)' .+ 2 .* ones(5,5) * ones(5,5)' .* 4 .+
            2 .* ones(5,5) * ones(5,5)' .* 9

        χ_diff_x_pseudo = KalmanFilter.AugmentedPseudoSigmaPoints(KalmanFilter.Augmented(LowerTriangular(ones(5,5)), LowerTriangular(2 .* ones(5,5))))
        𝓨_diff_y = KalmanFilter.AugmentedSigmaPoints(ones(4), ones(4,5) .* 4, ones(4,5), ones(4,5) .* 2, ones(4,5) .* 3)

        P = @inferred KalmanFilter.cov(χ_diff_x_pseudo, 𝓨_diff_y, weight_params)
        @test P == 2 .* LowerTriangular(ones(5,5)) * ones(4,5)' .* 4 .+
            2 .* LowerTriangular(-ones(5,5)) * ones(4,5)' .* 2

        #P_dest = KalmanFilter.Augmented(P, ones(5,5))
        P = @inferred KalmanFilter.cov!(P, χ_diff_x_pseudo, 𝓨_diff_y, weight_params)
        @test P == 2 .* LowerTriangular(ones(5,5)) * ones(4,5)' .* 4 .+
            2 .* LowerTriangular(-ones(5,5)) * ones(4,5)' .* 2

        P = zeros(5,4)
        P = @inferred KalmanFilter._cov!(P, χ_diff_x, 𝓨_diff_y, 4)
        @test P == 4 .* ones(5,5) * ones(4,5)' .* 16 .+
            4 .* ones(5,5) * ones(4,5)' .+ 4 .* ones(5,5) * ones(4,5)' .* 4 .+
            4 .* ones(5,5) * ones(4,5)' .* 9
    end

    @testset "Time update" begin
        x = [1., 1.]
        P = [1. 0.; 0. 1.]
        Q = [1. 0.; 0. 1.]

        F(x) = x .* [1., 2.]
        F(x, noise) = x .* [1., 2.] .+ noise
        F!(x, y) = x .= y .* [1., 2.]
        F!(x, y, noise) = x .= y .* [1., 2.] .+ noise

        tu = time_update(x, P, F, Augment(Q))
        @test get_state(tu) ≈ [1., 2.]
        @test get_covariance(tu) ≈ [2. 0.; 0. 5.]

        tu_inter = UKFTUIntermediate(2, true)
        tu = time_update!(tu_inter, x, P, F!, Augment(Q))
        @test get_state(tu) ≈ [1., 2.]
        @test get_covariance(tu) ≈ [2. 0.; 0. 5.]

        x = randn(6)
        A = randn(6,6)
        P = A'A
        B = randn(6,6)
        Q = B'B
        F = randn(6,6)
        f(x) = F * x
        f(x, noise) = F * x .+ noise

        tu = time_update(x, P, F, Q)
        tu_aug = time_update(x, P, f, Augment(Q))
        @test get_covariance(tu) ≈ get_covariance(tu_aug)
        @test get_state(tu) ≈ get_state(tu_aug)
    end

    @testset "Measurement update" begin

        y = [1., 1.]
        x = [1., 1.]
        P = [1. 0.; 0. 1.]
        R = [1. 0.; 0. 1.]

        H(x) = x .* [1., 1.]
        H(x, noise) = x .* [1., 1.] .+ noise
        H!(x, y) = x .= y .* [1., 1.]
        H!(x, y, noise) = x .= y .* [1., 1.] .+ noise

        mu = measurement_update(x, P, y, H, Augment(R))
        @test get_state(mu) ≈ [1., 1.]
        @test get_covariance(mu) ≈ [0.5 0.; 0. 0.5]
        @test get_innovation(mu) ≈ [0.0, 0.0] atol = 2e-10 #?
        @test get_innovation_covariance(mu) ≈ [2.0 0.0; 0.0 2.0]
        @test get_kalman_gain(mu) ≈ [0.5 0.0; 0.0 0.5]

        mu_inter = UKFMUIntermediate(2,2,true)
        mu = measurement_update!(mu_inter, x, P, y, H!, Augment(R))
        @test get_state(mu) ≈ [1., 1.]
        @test get_covariance(mu) ≈ [0.5 0.; 0. 0.5]
        @test get_innovation(mu) ≈ [0.0, 0.0] atol = 2e-10 #?
        @test get_innovation_covariance(mu) ≈ [2.0 0.0; 0.0 2.0]
        @test get_kalman_gain(mu) ≈ [0.5 0.0; 0.0 0.5]

        x = randn(6)
        A = randn(6,6)
        P = A'A
        B = randn(3,3)
        R = B'B
        y = randn(3)
        H = randn(3,6)
        h(x) = H * x
        h(x, noise) = H * x .+ noise

        tu = measurement_update(x, P, y, H, R)
        tu_aug = measurement_update(x, P, y, h, Augment(R))
        @test get_covariance(tu) ≈ get_covariance(tu_aug)
        @test get_state(tu) ≈ get_state(tu_aug)
    end
end
