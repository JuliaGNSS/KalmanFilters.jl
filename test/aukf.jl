@testset "Augmented Unscented Kalman filter" begin
    
    @testset "Time update" begin

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
        @test get_covariance(tu_aug) ≈ get_covariance(tu)
        @test get_state(tu_aug) ≈ get_state(tu)

        f!(y, x) = mul!(y, F, x)
        f!(y, x, noise) = y.= @~ F * x .+ noise
        tu_inter = AUKFTUIntermediate(6)
        tu_aug_inplace = time_update!(tu_inter, x, P, f!, Augment(Q))
        @test get_covariance(tu_aug_inplace) ≈ get_covariance(tu)
        @test get_state(tu_aug_inplace) ≈ get_state(tu)
    end

    @testset "Measurement update" begin

        x = randn(6)
        A = randn(6,6)
        P = A'A
        B = randn(3,3)
        R = B'B
        y = randn(3)
        H = randn(3,6)
        h(x) = H * x
        h(x, noise) = H * x .+ noise

        mu = measurement_update(x, P, y, H, R)
        mu_aug = measurement_update(x, P, y, h, Augment(R))
        @test get_covariance(mu_aug) ≈ get_covariance(mu)
        @test get_state(mu_aug) ≈ get_state(mu)

        h!(y, x) = mul!(y, H, x)
        h!(y, x, noise) = y .= @~ H * x .+ noise
        mu_inter = AUKFMUIntermediate(6, 3)
        mu_ukf_inplace = measurement_update!(mu_inter, x, P, y, h!, Augment(R))
        @test get_covariance(mu_ukf_inplace) ≈ get_covariance(mu)
        @test get_state(mu_ukf_inplace) ≈ get_state(mu)
    end
end
