@testset "Square root Unscented Kalman filter" begin
    @testset "Time update" begin

        x = [1., 1.]
        P = [1. 0.; 0. 1.]
        Q = [1. 0.; 0. 1.]

        F(x) = x .* [1., 2.]

        tu = time_update(x, cholesky(P), F, cholesky(Q))
        @test get_state(tu) ≈ [1., 2.]
        @test get_covariance(tu) ≈ [2. 0.; 0. 5.]
    end

    @testset "Measurement update" begin

    y = [1., 1.]
    x = [1., 1.]
    P = [1. 0.; 0. 1.]
    R = [1. 0.; 0. 1.]

    H(x) = x .* [1., 1.]

    mu = measurement_update(x, cholesky(P), y, H, cholesky(R))
    @test get_state(mu) ≈ [1., 1.]
    @test get_covariance(mu) ≈ [0.5 0.; 0. 0.5]
    @test get_innovation(mu) ≈ [0.0, 0.0] atol = 1e-10 #?
    @test get_innovation_covariance(mu) ≈ [2.0 0.0; 0.0 2.0]
    @test get_kalman_gain(mu) ≈ [0.5 0.0; 0.0 0.5]
    end
end
