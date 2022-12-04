@testset "Square root Unscented Kalman filter" begin

    @testset "Covariance" begin
        weight_params = ScaledSetWeightingParameters(0.5, 2, 1)
        x = randn(5)
        PL_prior = randn(5, 5)
        P_prior = PL_prior * PL_prior'
        χ = KalmanFilters.calc_sigma_points(x, P_prior, weight_params)
        F = randn(5, 5)
        f(x) = F * x
        𝓨 = KalmanFilters.transform(f, χ)
        QL = randn(5, 5)
        Q = QL * QL'
        Q_chol = cholesky(Q)
        P = @inferred KalmanFilters.cov(𝓨, Q_chol)
        @test P.L * P.U ≈ KalmanFilters.cov(𝓨, Q)
    end

    @testset "Posterior covariance" begin
        weight_params = ScaledSetWeightingParameters(0.5, 2, 1)
        x = randn(5)
        PL = randn(5, 5)
        P = PL * PL'
        P_chol = cholesky(P)
        χ = KalmanFilters.calc_sigma_points(x, P, weight_params)
        F = randn(3, 5)
        h(x) = F * x
        𝓨 = KalmanFilters.transform(h, χ)
        y_est = KalmanFilters.mean(𝓨)
        unbiased_𝓨 = KalmanFilters.substract_mean(𝓨, y_est)
        RL = randn(3, 3)
        R = RL * RL'
        R_chol = cholesky(R)
        S = KalmanFilters.cov(unbiased_𝓨, R)
        S_chol = KalmanFilters.cov(unbiased_𝓨, R_chol)
        Pᵪᵧ = KalmanFilters.cov(χ, unbiased_𝓨)
        K = Pᵪᵧ / S_chol
        K_temp, P_post = @inferred KalmanFilters.calc_kalman_gain_and_posterior_covariance(P_chol, Pᵪᵧ, S_chol, [])
        @test K_temp ≈ K
        @test P_post.L * P_post.U ≈ KalmanFilters.calc_posterior_covariance(P, Pᵪᵧ, K, [])
    end

    @testset "Time update with $T type $t" for T = (Float64, ComplexF64), t = ((vec = Vector, mat = Matrix), (vec = SVector{3}, mat = SMatrix{3,3}))

        x = t.vec(randn(T, 3))
        A = t.mat(randn(T, 3, 3))
        P = A'A
        P_chol = cholesky(Hermitian(P))
        B = t.mat(randn(T, 3, 3))
        Q = B'B
        Q_chol = cholesky(Hermitian(Q))
        F = t.mat(randn(T, 3, 3))
        f(x) = F * x

        tu = @inferred time_update(x, P, F, Q)
        tu_chol = @inferred time_update(x, P_chol, f, Q_chol)
        @test @inferred(get_covariance(tu_chol)) ≈ get_covariance(tu)
        @test @inferred(get_state(tu_chol)) ≈ get_state(tu)

        if x isa Vector
            f!(y, x) = mul!(y, F, x)
            tu_inter = @inferred SRUKFTUIntermediate(T, 3)
            tu_chol_inplace = @inferred time_update!(tu_inter, x, P_chol, f!, Q_chol)
            @test @inferred(get_covariance(tu_chol_inplace)) ≈ get_covariance(tu)
            @test @inferred(get_state(tu_chol_inplace)) ≈ get_state(tu)
        end
    end

    @testset "Measurement update with $T type $t" for T = (Float64, ComplexF64), t = ((vec = Vector, mat = Matrix), (vec = SVector{3}, mat = SMatrix{3,3}))

        x = t.vec(randn(T, 3))
        A = t.mat(randn(T, 3, 3))
        P = A'A
        P_chol = cholesky(Hermitian(P))
        B = t.mat(randn(T, 3, 3))
        R = B'B
        y = t.vec(randn(T, 3))
        R_chol = cholesky(Hermitian(R))
        H = t.mat(randn(T, 3, 3))
        h(x) = H * x

        mu = @inferred measurement_update(x, P, y, H, R)
        mu_chol = @inferred measurement_update(x, P_chol, y, h, R_chol)
        @test @inferred(get_covariance(mu_chol)) ≈ get_covariance(mu)
        @test @inferred(get_state(mu_chol)) ≈ get_state(mu)

        if x isa Vector
            h!(y, x) = mul!(y, H, x)
            mu_inter = @inferred SRUKFMUIntermediate(T, 3, 3)
            mu_chol_inplace = @inferred measurement_update!(mu_inter, x, P_chol, y, h!, R_chol)
            @test @inferred(get_covariance(mu_chol_inplace)) ≈ get_covariance(mu)
            @test @inferred(get_state(mu_chol_inplace)) ≈ get_state(mu)
        end
    end
end
