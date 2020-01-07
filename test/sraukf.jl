@testset "Square root Augmented Unscented Kalman filter" begin
    @testset "time update" begin
        x = [1., 1.]
        P = [1. 0.; 0. 1.]
        Q = [1. 0.; 0. 1.]

        f(x) = x .* [1., 2.]
        f(x, noise) = x .* [1., 2.] .+ noise

        tu = time_update(x, cholesky(P), f, Augment(cholesky(Q)))
        @test get_state(tu) ≈ [1., 2.]
        @test get_covariance(tu) ≈ [2. 0.; 0. 5.]

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
        @test get_covariance(tu) ≈ get_covariance(tu_chol)
        @test get_state(tu) ≈ get_state(tu_chol)
    end

    @testset "measurement update" begin

        y = [1., 1.]
        x = [1., 1.]
        P = [1. 0.; 0. 1.]
        R = [1. 0.; 0. 1.]

        h(x) = x .* [1., 1.]
        h(x, noise) = x .* [1., 1.] .+ noise

        mu = measurement_update(x, cholesky(P), y, h, Augment(cholesky(R)))
        @test get_state(mu) ≈ [1., 1.]
        @test get_covariance(mu) ≈ [0.5 0.; 0. 0.5]
        @test get_innovation(mu) ≈ [0.0, 0.0] atol = 2e-10 #?
        @test get_innovation_covariance(mu) ≈ [2.0 0.0; 0.0 2.0]
        @test get_kalman_gain(mu) ≈ [0.5 0.0; 0.0 0.5]

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

        tu = measurement_update(x, P, y, H, R)
        tu_chol = measurement_update(x, P_chol, y, h, Augment(R_chol))
        @test get_covariance(tu) ≈ get_covariance(tu_chol)
        @test get_state(tu) ≈ get_state(tu_chol)
    end
end
