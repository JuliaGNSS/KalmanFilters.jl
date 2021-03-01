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

        tu = measurement_update(x, P, y, H, R)
        tu_chol = measurement_update(x, P_chol, y, h, Augment(R_chol))
        @test get_covariance(tu) ≈ get_covariance(tu_chol)
        @test get_state(tu) ≈ get_state(tu_chol)
    end
end
