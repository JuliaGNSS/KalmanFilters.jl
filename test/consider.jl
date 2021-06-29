@testset "Consider Kalman filter" begin

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

        # Comparison measurement update
        # Nulling Kalman Gain for considered states
        # http://www.anuncommonlab.com/articles/how-kalman-filters-work/part2.html
        ỹ = KalmanFilters.calc_innovation(H, x, y)
        PHᵀ = KalmanFilters.calc_P_xy(P, H)
        S = KalmanFilters.calc_innovation_covariance(H, P, R)
        K = KalmanFilters.calc_kalman_gain(PHᵀ, S, [])
        K[5:6,:] .= 0.0
        x_post = KalmanFilters.calc_posterior_state(x, K, ỹ, [])
        # Use Joseph form, because of nulling Kalman gain
        P_post = (I - K * H) * P * (I - K * H)' + K * R * K'

        cons_mu = @inferred measurement_update(x, P, y, H, R, consider = 5:6)
        @test @inferred(get_covariance(cons_mu)) ≈ P_post
        @test @inferred(get_state(cons_mu)) ≈ x_post
        @test @inferred(get_kalman_gain(cons_mu)) ≈ K[1:4,:]

        cons_mu_chol = measurement_update(x, P_chol, y, H, R_chol, consider = 5:6)
        @test @inferred(get_covariance(cons_mu_chol)) ≈ P_post
        @test @inferred(get_state(cons_mu_chol)) ≈ x_post
        @test @inferred(get_kalman_gain(cons_mu_chol)) ≈ K[1:4,:]

        cons_mu_ukf = measurement_update(x, P, y, h, R, consider = 5:6)
        @test @inferred(get_covariance(cons_mu_ukf)) ≈ P_post
        @test @inferred(get_state(cons_mu_ukf)) ≈ x_post
        @test @inferred(get_kalman_gain(cons_mu_ukf)) ≈ K[1:4,:]

        cons_mu_srukf = measurement_update(x, P_chol, y, h, R_chol, consider = 5:6)
        @test @inferred(get_covariance(cons_mu_srukf)) ≈ P_post
        @test @inferred(get_state(cons_mu_srukf)) ≈ x_post
        @test @inferred(get_kalman_gain(cons_mu_srukf)) ≈ K[1:4,:]
    end
end