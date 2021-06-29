@testset "Square root Kalman filter" begin

    @testset "Calculate upper triangular of QR" begin
        A = randn(10, 5)
        R_test = @inferred KalmanFilters.calc_upper_triangular_of_qr!(copy(A))

        Q, R = qr(A)
        @test R_test ≈ R

        qr_zeros = zeros(10)
        qr_space_length = @inferred KalmanFilters.calc_gels_working_size(A, qr_zeros)
        qr_space = zeros(qr_space_length)
        R_res = zeros(5, 5)
        R_test_inplace = @inferred KalmanFilters.calc_upper_triangular_of_qr_inplace!(R_res, copy(A), qr_zeros, qr_space)
        @test R_test_inplace ≈ R
    end

    @testset "Time update" begin

        x = [1., 1.]
        P = [1. 0.; 0. 1.]
        F = [1. 0.; 0. 2.]
        Q = [1. 0.; 0. 1.]


        tu = @inferred time_update(x, cholesky(P), F, cholesky(Q))
        @test @inferred(get_state(tu)) == [1., 2.]
        @test @inferred(get_covariance(tu)) ≈ [2. 0.; 0. 5.]

        x = 1.
        P = 1.
        F = 1.
        Q = 1.


        tu = @inferred time_update(x, cholesky(P), F, cholesky(Q))
        @test @inferred(get_state(tu)) == 1.
        @test @inferred(get_covariance(tu)) ≈ [2.]

        x = randn(6)
        PL = randn(6,6)
        P = PL'PL
        P_chol = cholesky(P)
        QL = randn(6,6)
        Q = QL'QL
        Q_chol = cholesky(Q)
        F = randn(6,6)

        tu = @inferred time_update(x, P, F, Q)
        tu_chol = @inferred time_update(x, P_chol, F, Q_chol)
        @test @inferred(get_covariance(tu_chol)) ≈ get_covariance(tu)
        @test @inferred(get_state(tu_chol)) ≈ get_state(tu)

        tu_interm = @inferred SRKFTUIntermediate(6)
        tu_chol_inplace = @inferred time_update!(tu_interm, x, P_chol, F, Q_chol)
        @test @inferred(get_covariance(tu_chol_inplace)) ≈ get_covariance(tu)
        @test @inferred(get_state(tu_chol_inplace)) ≈ get_state(tu)
    end

    @testset "Measurement update" begin

        y = [1., 1.]
        x = [1., 1.]
        P = [1. 0.; 0. 1.]
        H = [1. 0.; 0. 1.]
        R = [1. 0.; 0. 1.]


        mu = @inferred measurement_update(x, cholesky(P), y, H, cholesky(R))
        @test @inferred(get_state(mu)) == [1., 1.]
        @test @inferred(get_covariance(mu)) ≈ [0.5 0.; 0. 0.5]
        @test @inferred(get_innovation(mu)) == [0.0, 0.0]
        @test @inferred(get_innovation_covariance(mu)) ≈ [2.0 0.0; 0.0 2.0]
        @test @inferred(get_kalman_gain(mu)) ≈ [0.5 0.0; 0.0 0.5]

        x = randn(6)
        PL = randn(6,6)
        P = PL'PL
        P_chol = cholesky(P)
        RL = randn(3,3)
        R = RL'RL
        y = randn(3)
        R_chol = cholesky(R)
        H = randn(3,6)

        mu = @inferred measurement_update(x, P, y, H, R)
        mu_chol = @inferred measurement_update(x, P_chol, y, H, R_chol)
        @test @inferred(get_covariance(mu_chol)) ≈ get_covariance(mu)
        @test @inferred(get_state(mu_chol)) ≈ get_state(mu)

        mu_interm = SRKFMUIntermediate(6, 3)
        mu_chol_inplace = @inferred measurement_update!(mu_interm, x, P_chol, y, H, R_chol)
        @test @inferred(get_covariance(mu)) ≈ get_covariance(mu_chol_inplace)
        @test @inferred(get_state(mu)) ≈ get_state(mu_chol_inplace)
    end
end
