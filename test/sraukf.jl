@testset "Square root Augmented Unscented Kalman filter" begin

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
        f(x, noise) = F * x .+ noise

        tu = time_update(x, P, F, Q)
        tu_chol = time_update(x, P_chol, f, Augment(Q_chol))
        @test get_covariance(tu_chol) ≈ get_covariance(tu)
        @test get_state(tu_chol) ≈ get_state(tu)

        if x isa Vector
            f!(y, x) = mul!(y, F, x)
            f!(y, x, noise) = y.= @~ F * x .+ noise
            tu_inter = SRAUKFTUIntermediate(T, 3)
            tu_aug_inplace = time_update!(tu_inter, x, P_chol, f!, Augment(Q_chol))
            @test get_covariance(tu_aug_inplace) ≈ get_covariance(tu)
            @test get_state(tu_aug_inplace) ≈ get_state(tu)
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
        h(x, noise) = H * x .+ noise

        mu = measurement_update(x, P, y, H, R)
        mu_chol = @inferred measurement_update(x, P_chol, y, h, Augment(R_chol))
        @test get_covariance(mu) ≈ get_covariance(mu_chol)
        @test get_state(mu) ≈ get_state(mu_chol)

        if x isa Vector
            h!(y, x) = mul!(y, H, x)
            h!(y, x, noise) = y .= @~ H * x .+ noise
            mu_inter = @inferred SRAUKFMUIntermediate(T, 3, 3)
            mu_ukf_inplace = measurement_update!(mu_inter, x, P_chol, y, h!, Augment(R_chol))
            @test get_covariance(mu_ukf_inplace) ≈ get_covariance(mu)
            @test get_state(mu_ukf_inplace) ≈ get_state(mu)
        end
    end

    @testset "Scalar measurement update with $T type $t" for T = (Float64, ComplexF64), t = ((vec = Vector, mat = Matrix), (vec = SVector{3}, mat = SMatrix{3,3}))
        x = t.vec(randn(T, 3))
        PL = t.mat(randn(T, 3, 3))
        P = PL'PL
        P_chol = cholesky(Hermitian(P))
        RL = randn()
        R = RL'RL
        R_chol = cholesky(R)
        y = randn(T)
        H = t.vec(randn(T, 3))'
        h(x) = H * x
        h(x, noise) = H * x .+ noise

        mu = measurement_update(x, P, y, H, R)
        mu_chol = @inferred measurement_update(x, P_chol, y, h, Augment(R_chol))
        @test @inferred(get_covariance(mu_chol)) ≈ get_covariance(mu)
        @test @inferred(get_state(mu_chol)) ≈ get_state(mu)
    end
end
