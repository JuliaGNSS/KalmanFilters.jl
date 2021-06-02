@testset "Square root Unscented Kalman filter" begin

    @testset "Covariance" begin
        weight_params = ScaledSetWeightingParameters(0.5, 2, 1)
        x = randn(5)
        PL_prior = randn(5, 5)
        P_prior = PL_prior * PL_prior'
        χ = KalmanFilter.calc_sigma_points(x, P_prior, weight_params)
        F = randn(5, 5)
        f(x) = F * x
        𝓨 = KalmanFilter.transform(f, χ)
        QL = randn(5, 5)
        Q = QL * QL'
        Q_chol = cholesky(Q)
        P = @inferred KalmanFilter.cov(𝓨, Q_chol)
        @test P.L * P.U ≈ KalmanFilter.cov(𝓨, Q)
    end

    @testset "Posterior covariance" begin
        weight_params = ScaledSetWeightingParameters(0.5, 2, 1)
        x = randn(5)
        PL = randn(5, 5)
        P = PL * PL'
        P_chol = cholesky(P)
        χ = KalmanFilter.calc_sigma_points(x, P, weight_params)
        F = randn(3, 5)
        h(x) = F * x
        𝓨 = KalmanFilter.transform(h, χ)
        y_est = KalmanFilter.mean(𝓨)
        unbiased_𝓨 = KalmanFilter.substract_mean(𝓨, y_est)
        RL = randn(3, 3)
        R = RL * RL'
        R_chol = cholesky(R)
        S = KalmanFilter.cov(unbiased_𝓨, R)
        S_chol = KalmanFilter.cov(unbiased_𝓨, R_chol)
        Pᵪᵧ = KalmanFilter.cov(χ, unbiased_𝓨)
        K = Pᵪᵧ / S_chol
        K_temp, P_post = @inferred KalmanFilter.calc_kalman_gain_and_posterior_covariance(P_chol, Pᵪᵧ, S_chol, [])
        @test K_temp ≈ K
        @test P_post.L * P_post.U ≈ KalmanFilter.calc_posterior_covariance(P, Pᵪᵧ, K, [])
    end

    @testset "Time update" begin

        x = randn(6)
        A = randn(6,6)
        P = A'A
        P_chol = cholesky(P)
        B = randn(6,6)
        Q = B'B
        Q_chol = cholesky(Q)
        F = randn(6,6)
        f(x) = F * x

        tu = @inferred time_update(x, P, F, Q)
        tu_chol = @inferred time_update(x, P_chol, f, Q_chol)
        @test @inferred(get_covariance(tu_chol)) ≈ get_covariance(tu)
        @test @inferred(get_state(tu_chol)) ≈ get_state(tu)

        f!(y, x) = mul!(y, F, x)
        tu_inter = @inferred SRUKFTUIntermediate(6)
        tu_chol_inplace = @inferred time_update!(tu_inter, x, P_chol, f!, Q_chol)
        @test @inferred(get_covariance(tu_chol_inplace)) ≈ get_covariance(tu)
        @test @inferred(get_state(tu_chol_inplace)) ≈ get_state(tu)
    end

    @testset "Measurement update" begin

        x = randn(6)
        A = randn(6,6)
        P = A'A
        P_chol = cholesky(P)
        B = randn(3,3)
        R = B'B
        y = randn(3)
        R_chol = cholesky(R)
        H = randn(3,6)
        h(x) = H * x

        mu = @inferred measurement_update(x, P, y, H, R)
        mu_chol = @inferred measurement_update(x, P_chol, y, h, R_chol)
        @test @inferred(get_covariance(mu_chol)) ≈ get_covariance(mu)
        @test @inferred(get_state(mu_chol)) ≈ get_state(mu)

        h!(y, x) = mul!(y, H, x)
        mu_inter = @inferred SRUKFMUIntermediate(6, 3)
        mu_chol_inplace = @inferred measurement_update!(mu_inter, x, P_chol, y, h!, R_chol)
        @test @inferred(get_covariance(mu_chol_inplace)) ≈ get_covariance(mu)
        @test @inferred(get_state(mu_chol_inplace)) ≈ get_state(mu)
    end
end
