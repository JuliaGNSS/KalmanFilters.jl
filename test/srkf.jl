@testset "Square root Kalman filter" begin
    @testset "Time update" begin

        x = [1., 1.]
        P = [1. 0.; 0. 1.]
        F = [1. 0.; 0. 2.]
        Q = [1. 0.; 0. 1.]


        tu = time_update(x, cholesky(P), F, cholesky(Q))
        @test get_state(tu) == [1., 2.]
        @test get_covariance(tu) ≈ [2. 0.; 0. 5.]

        x = 1.
        P = 1.
        F = 1.
        Q = 1.


        tu = time_update(x, cholesky(P), F, cholesky(Q))
        @test get_state(tu) == 1.
        @test get_covariance(tu) ≈ [2.]

        x = randn(6)
        PL = randn(6,6)
        P = PL'PL
        P_chol = cholesky(P)
        QL = randn(6,6)
        Q = QL'QL
        Q_chol = cholesky(Q)
        F = randn(6,6)

        tu = time_update(x, P, F, Q)
        tu_chol = time_update(x, P_chol, F, Q_chol)
        @test get_covariance(tu) ≈ get_covariance(tu_chol)
        @test get_state(tu) ≈ get_state(tu_chol)

        tu_interm = SRKFTUIntermediate(6)
        tu_chol_inplace = time_update!(tu_interm, x, P_chol, F, Q_chol)
        @test get_covariance(tu) ≈ get_covariance(tu_chol_inplace)
        @test get_state(tu) ≈ get_state(tu_chol_inplace)
    end

    @testset "Measurement update" begin

        y = [1., 1.]
        x = [1., 1.]
        P = [1. 0.; 0. 1.]
        H = [1. 0.; 0. 1.]
        R = [1. 0.; 0. 1.]


        mu = measurement_update(x, cholesky(P), y, H, cholesky(R))
        @test get_state(mu) == [1., 1.]
        @test get_covariance(mu) ≈ [0.5 0.; 0. 0.5]
        @test get_innovation(mu) == [0.0, 0.0]
        @test get_innovation_covariance(mu) ≈ [2.0 0.0; 0.0 2.0]
        @test get_kalman_gain(mu) ≈ [0.5 0.0; 0.0 0.5]

        x = randn(6)
        PL = randn(6,6)
        P = PL'PL
        P_chol = cholesky(P)
        RL = randn(3,3)
        R = RL'RL
        y = randn(3)
        R_chol = cholesky(R)
        H = randn(3,6)

        mu = measurement_update(x, P, y, H, R)
        mu_chol = measurement_update(x, P_chol, y, H, R_chol)
        @test get_covariance(mu) ≈ get_covariance(mu_chol)
        @test get_state(mu) ≈ get_state(mu_chol)

        mu_interm = SRKFMUIntermediate(6, 3)
        mu_chol_inplace = measurement_update!(mu_interm, x, P_chol, y, H, R_chol)
        @test get_covariance(mu) ≈ get_covariance(mu_chol_inplace)
        @test get_state(mu) ≈ get_state(mu_chol_inplace)
    end
end
