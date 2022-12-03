@testset "Augmented Unscented Kalman filter" begin
    
    @testset "Time update with $T type $t" for T = (Float64, ComplexF64), t = ((vec = Vector, mat = Matrix), (vec = SVector{3}, mat = SMatrix{3,3}))

        x = t.vec(randn(T, 3))
        A = t.mat(randn(T, 3, 3))
        P = A'A
        B = t.mat(randn(T, 3, 3))
        Q = B'B
        F = t.mat(randn(T, 3, 3))
        f(x) = F * x
        f(x, noise) = F * x .+ noise

        tu = time_update(x, P, F, Q)
        tu_aug = time_update(x, P, f, Augment(Q))
        @test get_covariance(tu_aug) ≈ get_covariance(tu)
        @test get_state(tu_aug) ≈ get_state(tu)

        if x isa Vector
            f!(y, x) = mul!(y, F, x)
            f!(y, x, noise) = y.= @~ F * x .+ noise
            tu_inter = AUKFTUIntermediate(T, 3)
            tu_aug_inplace = time_update!(tu_inter, x, P, f!, Augment(Q))
            @test get_covariance(tu_aug_inplace) ≈ get_covariance(tu)
            @test get_state(tu_aug_inplace) ≈ get_state(tu)
        end
    end

    @testset "Measurement update with $T type $t" for T = (Float64, ComplexF64), t = ((vec = Vector, mat = Matrix), (vec = SVector{3}, mat = SMatrix{3,3}))

        x = t.vec(randn(T, 3))
        A = t.mat(randn(T, 3, 3))
        P = A'A
        B = t.mat(randn(T, 3, 3))
        R = B'B
        y = t.vec(randn(T, 3))
        H = t.mat(randn(T, 3, 3))
        h(x) = H * x
        h(x, noise) = H * x .+ noise

        mu = measurement_update(x, P, y, H, R)
        mu_aug = measurement_update(x, P, y, h, Augment(R))
        @test get_covariance(mu_aug) ≈ get_covariance(mu)
        @test get_state(mu_aug) ≈ get_state(mu)

        h!(y, x) = mul!(y, H, x)
        h!(y, x, noise) = y .= @~ H * x .+ noise
        mu_inter = AUKFMUIntermediate(T, 3, 3)
        mu_ukf_inplace = measurement_update!(mu_inter, x, P, y, h!, Augment(R))
        @test get_covariance(mu_ukf_inplace) ≈ get_covariance(mu)
        @test get_state(mu_ukf_inplace) ≈ get_state(mu)
    end

    @testset "Scalar measurement update with $T type $t" for T = (Float64, ComplexF64), t = ((vec = Vector, mat = Matrix), (vec = SVector{3}, mat = SMatrix{3,3}))
        x = t.vec(randn(T, 3))
        PL = t.mat(randn(T, 3, 3))
        P = PL'PL
        RL = randn()
        R = RL'RL
        y = randn(T)
        H = t.vec(randn(T, 3))'
        h(x) = H * x
        h(x, noise) = H * x .+ noise

        mu = measurement_update(x, P, y, H, R)
        mu_aug = @inferred measurement_update(x, P, y, h, Augment(R))
        @test @inferred(get_covariance(mu_aug)) ≈ get_covariance(mu)
        @test @inferred(get_state(mu_aug)) ≈ get_state(mu)
    end
end
