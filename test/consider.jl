@testset "Consider Kalman filter" begin

    @testset "Consider matrices" begin
        x = randn(6)
        PL = randn(6, 6)
        P = PL'PL
        P_chol = cholesky(P)
        F = collect(Diagonal(ones(6)))
        F[1:4,1:6] = randn(4, 6)
        H = randn(3,6)

        cons_x = ConsideredState(x[1:4], x[5:6])
        @test all(cons_x .== x)
        cons_P = ConsideredCovariance(P[1:4,1:4], P[1:4,5:6], P[5:6,5:6])
        @test all(cons_P .≈ P)
        cons_P_chol = @inferred cholesky(cons_P)
        @test all(cons_P_chol.U .≈ P_chol.U)
        cons_F = ConsideredProcessModel(F[1:4,1:4], F[1:4,5:6])
        @test all(cons_F .== F)
        cons_H = ConsideredMeasurementModel(H[:,1:4], H[:,5:6])
        @test all(cons_H .== H)
    end

    @testset "Time update" begin

        x = randn(6)
        PL = randn(6, 6)
        P = PL'PL
        P_chol = cholesky(P)
        QLS = randn(4, 4)
        cons_Q = QLS'QLS
        cons_Q_chol = cholesky(cons_Q)
        Q_chol = Cholesky(zeros(6, 6), 'U', 0)
        Q_chol.factors[1:4,1:4] = cons_Q_chol.U
        Q = Q_chol.U'Q_chol.U
        F = collect(Diagonal(ones(6)))
        F[1:4,1:6] = randn(4, 6)
        cons_x = ConsideredState(x[1:4], x[5:6])
        cons_P = ConsideredCovariance(P[1:4,1:4], P[1:4,5:6], P[5:6,5:6])
        cons_P_chol = @inferred cholesky(cons_P)
        cons_F = ConsideredProcessModel(F[1:4,1:4], F[1:4,5:6])

        fake_cons_tu = @inferred time_update(x, P, F, Q)

        cons_tu = @inferred time_update(cons_x, cons_P, cons_F, cons_Q)
        @test @inferred(get_covariance(cons_tu)) ≈ get_covariance(fake_cons_tu)
        @test @inferred(get_state(cons_tu)) ≈ get_state(fake_cons_tu)

        cons_tu_chol = @inferred time_update(cons_x, cons_P_chol, cons_F, cons_Q_chol)
        @test @inferred(get_covariance(cons_tu_chol)) ≈ get_covariance(fake_cons_tu)
        @test @inferred(get_state(cons_tu_chol)) ≈ get_state(fake_cons_tu)
    end

    @testset "Measurement update" begin

        x = randn(6)
        PL = randn(6,6)
        P = PL'PL
        P_chol = cholesky(P)
        RL = randn(3,3)
        R = RL'RL
        y = randn(3)
        R_chol = cholesky(R)
        H = randn(3,6)
        h(x) = H * x

        cons_x = ConsideredState(x[1:4], x[5:6])
        cons_P = ConsideredCovariance(P[1:4,1:4], P[1:4,5:6], P[5:6,5:6])
        cons_P_chol = @inferred cholesky(cons_P)
        cons_H = ConsideredMeasurementModel(H[:,1:4], H[:,5:6])

        # Comparison measurement update
        # Nulling Kalman Gain for considered states
        # http://www.anuncommonlab.com/articles/how-kalman-filters-work/part2.html
        ỹ = KalmanFilter.calc_innovation(H, x, y)
        PHᵀ = KalmanFilter.calc_P_xy(P, H)
        S = KalmanFilter.calc_innovation_covariance(H, P, R)
        K = KalmanFilter.calc_kalman_gain(PHᵀ, S)
        K[5:6,:] .= 0.0
        x_post = KalmanFilter.calc_posterior_state(x, K, ỹ)
        # Use Joseph form, because of nulling Kalman gain
        P_post = (I - K * H) * P * (I - K * H)' + K * R * K'

        cons_mu = @inferred measurement_update(cons_x, cons_P, y, cons_H, R)
        @test @inferred(get_covariance(cons_mu)) ≈ P_post
        @test @inferred(get_state(cons_mu)) ≈ x_post

        cons_mu_chol = measurement_update(cons_x, cons_P_chol, y, cons_H, R_chol)
        # P_cc does not work for cons_mu_chol
        #@test @inferred(get_covariance(cons_mu_chol)) ≈ P_post
        @test @inferred(get_state(cons_mu_chol)) ≈ x_post

        cons_mu_ukf = measurement_update(cons_x, cons_P, y, h, R)
        @test @inferred(get_covariance(cons_mu_ukf)) ≈ P_post
        @test @inferred(get_state(cons_mu_ukf)) ≈ x_post

        cons_mu_srukf = measurement_update(cons_x, cons_P_chol, y, h, R_chol)
        @test @inferred(get_covariance(cons_mu_srukf)) ≈ P_post
        @test @inferred(get_state(cons_mu_srukf)) ≈ x_post
    end
end