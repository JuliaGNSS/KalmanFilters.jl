@testset "Square root Augmented Unscented Kalman filter" begin
    @testset "time update" begin
        x = randn(6)
        A = randn(6,6)
        P = A'A
        P_chol = cholesky(P)
        B = randn(6,6)
        Q = B'B
        Q_chol = cholesky(Q)
        F = randn(6,6)
        f(x) = F * x
        f(x, noise) = F * x .+ noise

        tu = time_update(x, P, F, Q)
        tu_chol = time_update(x, P_chol, f, Augment(Q_chol))
        @test get_covariance(tu_chol) ≈ get_covariance(tu)
        @test get_state(tu_chol) ≈ get_state(tu)

        f!(y, x) = mul!(y, F, x)
        f!(y, x, noise) = y.= @~ F * x .+ noise
        tu_inter = SRAUKFTUIntermediate(6)
        tu_aug_inplace = time_update!(tu_inter, x, P_chol, f!, Augment(Q_chol))
        @test get_covariance(tu_aug_inplace) ≈ get_covariance(tu)
        @test get_state(tu_aug_inplace) ≈ get_state(tu)
    end

    @testset "measurement update" begin
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
        h(x, noise) = H * x .+ noise

        mu = measurement_update(x, P, y, H, R)
        mu_chol = @inferred measurement_update(x, P_chol, y, h, Augment(R_chol))
        @test get_covariance(mu) ≈ get_covariance(mu_chol)
        @test get_state(mu) ≈ get_state(mu_chol)

        h!(y, x) = mul!(y, H, x)
        h!(y, x, noise) = y .= @~ H * x .+ noise
        mu_inter = @inferred SRAUKFMUIntermediate(6, 3)
        mu_ukf_inplace = measurement_update!(mu_inter, x, P_chol, y, h!, Augment(R_chol))
        @test get_covariance(mu_ukf_inplace) ≈ get_covariance(mu)
        @test get_state(mu_ukf_inplace) ≈ get_state(mu)
    end
end
