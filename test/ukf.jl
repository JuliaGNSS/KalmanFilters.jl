@testset "Unscented Kalman filter" begin

    @testset "Time update" begin

        x = randn(6)
        PL = randn(6,6)
        P = PL'PL
        QL = randn(6,6)
        Q = QL'QL
        F = randn(6,6)
        f(x) = F * x

        tu = time_update(x, P, F, Q)
        tu_ukf = time_update(x, P, f, Q)
        @test get_covariance(tu_ukf) ≈ get_covariance(tu)
        @test get_state(tu_ukf) ≈ get_state(tu)

        f!(y, x) = mul!(y, F, x)
        tu_inter = UKFTUIntermediate(6)
        tu = time_update!(tu_inter, x, P, f!, Q)
        @test get_covariance(tu_ukf) ≈ get_covariance(tu)
        @test get_state(tu_ukf) ≈ get_state(tu)

    end

    @testset "Measurement update" begin

        x = randn(6)
        PL = randn(6,6)
        P = PL'PL
        RL = randn(3,3)
        R = RL'RL
        y = randn(3)
        H = randn(3,6)
        h(x) = H * x

        mu = measurement_update(x, P, y, H, R)
        mu_ukf = measurement_update(x, P, y, h, R)
        @test get_covariance(mu_ukf) ≈ get_covariance(mu)
        @test get_state(mu_ukf) ≈ get_state(mu)

        h!(y, x) = mul!(y, H, x)
        mu_inter = UKFMUIntermediate(6, 3)
        mu_ukf_inplace = measurement_update!(mu_inter, x, P, y, h!, R)
        @test get_covariance(mu_ukf_inplace) ≈ get_covariance(mu)
        @test get_state(mu_ukf_inplace) ≈ get_state(mu)
    end
end
