@testset "Kalman filter" begin
    @testset "Time update inplace functions" begin
        x = randn(2)
        F = randn(2,2)
        x_apri = similar(x)
        @test KalmanFilter.calc_apriori_state!(x_apri, x, F) ≈ F * x

        Q = randn(2,2)
        P = randn(2,2)
        FP = similar(P)
        P_apri = similar(P)
        @test KalmanFilter.calc_apriori_covariance!(P_apri, FP, P, F, Q) ≈ F * P * F' .+ Q
    end

    @testset "Measurement update inplace functions" begin

        x = randn(2)
        y = randn(2)
        H = randn(2,2)
        ỹ = similar(y)
        @test KalmanFilter.calc_innovation!(ỹ, H, x, y) ≈ y .- H * x

        H = randn(2,2)
        R = randn(2,2)
        PHᵀ = randn(2,2)
        S = similar(R)
        @test KalmanFilter.calc_innovation_covariance!(S, H, PHᵀ, R) ≈ H * PHᵀ .+ R

        PHᵀ = randn(2,2)
        SL = randn(2,2)
        S = SL*SL'
        S_lu = similar(S)
        K = similar(PHᵀ)
        @test KalmanFilter.calc_kalman_gain!(S_lu, K, PHᵀ, S) ≈ PHᵀ / S

        x = randn(2)
        x_posterior = similar(x)
        K = randn(2,2)
        ỹ = randn(2)
        @test KalmanFilter.calc_posterior_state!(x_posterior, x, K, ỹ) ≈ x .+ K * ỹ

        P = randn(2,2)
        P_posterior = similar(P)
        PHᵀ = randn(2,2)
        K = randn(2,2)
        @test KalmanFilter.calc_posterior_covariance!(P_posterior, P, PHᵀ, K) ≈ P .- PHᵀ * K'

    end

    @testset "Time update" begin

        x = [1., 1.]
        P = [1. 0.; 0. 1.]
        F = [1. 0.; 0. 2.]
        Q = [1. 0.; 0. 1.]


        tu = time_update(x, P, F, Q)
        @test get_state(tu) == [1., 2.]
        @test get_covariance(tu) == [2. 0.; 0. 5.]

        tu_inter = KFTUIntermediate(2)
        tu = time_update!(tu_inter, x, P, F, Q)
        @test get_state(tu) == [1., 2.]
        @test get_covariance(tu) == [2. 0.; 0. 5.]
    end

    @testset "Measurement update" begin

        y = [1., 1.]
        x = [1., 1.]
        P = [1. 0.; 0. 1.]
        H = [1. 0.; 0. 1.]
        R = [1. 0.; 0. 1.]


        mu = measurement_update(x, P, y, H, R)
        @test get_state(mu) == [1., 1.]
        @test get_covariance(mu) == [0.5 0.; 0. 0.5]
        @test get_innovation(mu) == [0.0, 0.0]
        @test get_innovation_covariance(mu) == [2.0 0.0; 0.0 2.0]
        @test get_kalman_gain(mu) == [0.5 0.0; 0.0 0.5]

        mu_inter = KFMUIntermediate(2,2)
        mu = measurement_update!(mu_inter, x, P, y, H, R)
        @test get_state(mu) == [1., 1.]
        @test get_covariance(mu) ≈ [0.5 0.; 0. 0.5]
        @test get_innovation(mu) == [0.0, 0.0]
        @test get_innovation_covariance(mu) == [2.0 0.0; 0.0 2.0]
        @test get_kalman_gain(mu) ≈ [0.5 0.0; 0.0 0.5]

        y = 1.
        x = [1., 1.]
        P = [1. 0.; 0. 1.]
        H = [1., 0.]'
        R = 1.


        mu = measurement_update(x, P, y, H, R)
        @test get_state(mu) == [1., 1.]
        @test get_covariance(mu) == [0.5 0.; 0. 1.]
        @test get_innovation(mu) == 0.
        @test get_innovation_covariance(mu) == 2.
        @test get_kalman_gain(mu) == [0.5, 0.]

        H_temp = Matrix(H)
        y_temp = [y]
        R_temp = [R]
        mu_inter = KFMUIntermediate(2,1)
        mu = measurement_update!(mu_inter, x, P, y_temp, H_temp, R_temp)
        @test get_state(mu) == [1., 1.]
        @test get_covariance(mu) ≈ [0.5 0.; 0. 1.]
        @test get_innovation(mu) == [0.]
        @test get_innovation_covariance(mu) ≈ [2.]
        @test get_kalman_gain(mu) ≈ [0.5, 0.]

        y = [1., 1.]
        x = 1.
        P = 1.
        H = [1., 0.]
        R = [1. 0.; 0. 1.]

        mu = measurement_update(x, P, y, H, R)
        @test get_state(mu) == 1.
        @test get_covariance(mu) == 0.5
        @test get_innovation(mu) == [0.0, 1.0]
        @test get_innovation_covariance(mu) == [2.0 0.0; 0.0 1.0]
        @test get_kalman_gain(mu) == [0.5 0.0]

        x_temp = [x]
        P_temp = Matrix(adjoint([P]))
        H_temp = Matrix{eltype(H)}(undef, 2, 1)
        H_temp[1] = H[1]
        H_temp[2] = H[2]
        mu_inter = KFMUIntermediate(1,2)
        mu = measurement_update!(mu_inter, x_temp, P_temp, y, H_temp, R)
        @test get_state(mu) == [1.]
        @test get_covariance(mu) ≈ [0.5]
        @test get_innovation(mu) == [0.0, 1.0]
        @test get_innovation_covariance(mu) == [2.0 0.0; 0.0 1.0]
        @test get_kalman_gain(mu) ≈ [0.5 0.0]
    end
end
