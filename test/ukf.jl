@testset "Unscented Kalman filter" begin

    @testset "Time update with $T type $t" for T = (Float64, ComplexF64), t = ((vec = Vector, mat = Matrix), (vec = SVector{3}, mat = SMatrix{3,3}))

        x = t.vec(randn(T, 3))
        PL = t.mat(randn(T, 3, 3))
        P = PL'PL
        QL = t.mat(randn(T, 3, 3))
        Q = QL'QL
        F = t.mat(randn(T, 3, 3))
        f(x) = F * x

        tu = time_update(x, P, F, Q)
        tu_ukf = time_update(x, P, f, Q)
        @test get_covariance(tu_ukf) ≈ get_covariance(tu)
        @test get_state(tu_ukf) ≈ get_state(tu)

        if x isa Vector
            f!(y, x) = mul!(y, F, x)
            tu_inter = UKFTUIntermediate(T, 3)
            tu = time_update!(tu_inter, x, P, f!, Q)
            @test get_covariance(tu_ukf) ≈ get_covariance(tu)
            @test get_state(tu_ukf) ≈ get_state(tu)
        end

    end

    @testset "Measurement update with $T type $t" for T = (Float64, ComplexF64), t = ((vec = Vector, mat = Matrix), (vec = SVector{3}, mat = SMatrix{3,3}))

        x = t.vec(randn(T, 3))
        PL = t.mat(randn(T, 3, 3))
        P = PL'PL
        RL = t.mat(randn(T, 3, 3))
        R = RL'RL
        y = t.vec(randn(T, 3))
        H = t.mat(randn(T, 3, 3))
        h(x) = H * x

        mu = measurement_update(x, P, y, H, R)
        mu_ukf = @inferred measurement_update(x, P, y, h, R)
        @test get_covariance(mu_ukf) ≈ get_covariance(mu)
        @test get_state(mu_ukf) ≈ get_state(mu)

        if x isa Vector
            h!(y, x) = mul!(y, H, x)
            mu_inter = @inferred UKFMUIntermediate(T, 3, 3)
            mu_ukf_inplace = @inferred measurement_update!(mu_inter, x, P, y, h!, R)
            @test get_covariance(mu_ukf_inplace) ≈ get_covariance(mu)
            @test get_state(mu_ukf_inplace) ≈ get_state(mu)
        end
    end
end
