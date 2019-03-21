@testset "KF time update inplace functions" begin
    x = randn(2)
    F = randn(2,2)
    x_temp = similar(x)
    @test F * x ≈ KalmanFilter.calc_apriori_state!(x_temp, x, F)

    Q = randn(2,2)
    P = randn(2,2)
    FP = similar(P)
    @test F * P * F' .+ Q ≈ KalmanFilter.calc_apriori_covariance!(FP, P, F, Q)
end

@testset "KF measurement update inplace functions" begin

    x = randn(2)
    y = randn(2)
    H = randn(2,2)
    ỹ = similar(y)
    @test y - H * x ≈ KalmanFilter.calc_innovation!(ỹ, H, x, y)

    x = randn()
    H = randn(2)
    @test y - H * x ≈ KalmanFilter.calc_innovation!(ỹ, H, x, y)

    x = randn(2)
    y = randn()
    H = transpose(randn(2))
    @test y - H * x ≈ KalmanFilter.calc_innovation!(ỹ, H, x, y)

    H = randn(2,2)
    R = randn(2,2)
    PHᵀ = randn(2,2)
    S = similar(R)
    @test H * PHᵀ .+ R ≈ KalmanFilter.calc_innovation_covariance!(S, H, PHᵀ, R)

    H = randn(2)
    PHᵀ = transpose(randn(2))
    @test H * PHᵀ .+ R ≈ KalmanFilter.calc_innovation_covariance!(S, H, PHᵀ, R)

    H = adjoint(randn(2))
    PHᵀ = randn(2)
    R = randn()
    S = 0.
    @test H * PHᵀ .+ R ≈ KalmanFilter.calc_innovation_covariance!(S, H, PHᵀ, R)

    PHᵀ = randn(2,2)
    S = randn(2,2)
    S_lu = similar(S)
    K = similar(PHᵀ)
    @test PHᵀ / S ≈ KalmanFilter.calc_kalman_gain!(S_lu, K, PHᵀ, S)

    PHᵀ = randn(2,2)
    S = randn()
    K = similar(PHᵀ)
    @test PHᵀ ./ S ≈ KalmanFilter.calc_kalman_gain!(S_lu, K, PHᵀ, S)

    x = randn(2)
    K = randn(2,2)
    ỹ = randn(2)
    @test x .+ K * ỹ ≈ KalmanFilter.calc_posterior_state!(x, K, ỹ)

    x = randn(2)
    K = randn(2)
    ỹ = randn()
    @test x .+ K .* ỹ ≈ KalmanFilter.calc_posterior_state!(x, K, ỹ)

    x = randn()
    K = transpose(randn(2))
    ỹ = randn(2)
    @test x .+ K * ỹ ≈ KalmanFilter.calc_posterior_state!(x, K, ỹ)

    P = randn(2,2)
    PHᵀ = randn(2,2)
    K = randn(2,2)
    @test P - PHᵀ * K' ≈ KalmanFilter.calc_posterior_covariance!(P, PHᵀ, K)

    P = randn(2,2)
    PHᵀ = randn(2)
    K = randn(2)
    @test P - PHᵀ * K' ≈ KalmanFilter.calc_posterior_covariance!(P, PHᵀ, K)

    P = randn()
    PHᵀ = transpose(randn(2))
    K = transpose(randn(2))
    @test P - PHᵀ * K' ≈ KalmanFilter.calc_posterior_covariance!(P, PHᵀ, K)

    X = complex.(randn(3,3), randn(3,3))
    X_temp = copy(X)
    @test X' == KalmanFilter.adjoint!(X_temp)

    A = randn(2,2)
    B = randn(2,2)
    @test A / B ≈ KalmanFilter.rdiv!(A, B)

    A = adjoint(randn(2,2))
    B = randn(2,2)
    @test A / B ≈ KalmanFilter.rdiv!(A, B)

    A = transpose(randn(2,2))
    B = randn(2,2)
    @test A / B ≈ KalmanFilter.rdiv!(A, B)
end

@testset "KF time update" begin

    x = [1., 1.]
    P = [1. 0.; 0. 1.]
    F = [1. 0.; 0. 2.]
    Q = [1. 0.; 0. 1.]


    tu = time_update(x, P, F, Q)
    @test state(tu) == [1., 2.]
    @test covariance(tu) == [2. 0.; 0. 5.]

    tu_inter = KFTUIntermediate(2)
    tu = time_update!(tu_inter, x, P, F, Q)
    @test state(tu) == [1., 2.]
    @test covariance(tu) == [2. 0.; 0. 5.]

    x = 1.
    P = 1.
    F = 1.
    Q = 1.


    tu = time_update(x, P, F, Q)
    @test state(tu) == 1.
    @test covariance(tu) == 2.
end

@testset "KF measurement update" begin

    y = [1., 1.]
    x = [1., 1.]
    P = [1. 0.; 0. 1.]
    H = [1. 0.; 0. 1.]
    R = [1. 0.; 0. 1.]


    mu = measurement_update(x, P, y, H, R)
    @test state(mu) == [1., 1.]
    @test covariance(mu) == [0.5 0.; 0. 0.5]
    @test innovation(mu) == [0.0, 0.0]
    @test innovation_covariance(mu) == [2.0 0.0; 0.0 2.0]
    @test kalman_gain(mu) == [0.5 0.0; 0.0 0.5]

    mu_inter = KFMUIntermediate(2,2)
    mu = measurement_update!(mu_inter, x, P, y, H, R)
    @test state(mu) == [1., 1.]
    @test covariance(mu) == [0.5 0.; 0. 0.5]
    @test innovation(mu) == [0.0, 0.0]
    @test innovation_covariance(mu) == [2.0 0.0; 0.0 2.0]
    @test kalman_gain(mu) == [0.5 0.0; 0.0 0.5]

    y = 1.
    x = [1., 1.]
    P = [1. 0.; 0. 1.]
    H = [1., 0.]'
    R = 1.


    mu = measurement_update(x, P, y, H, R)
    @test state(mu) == [1., 1.]
    @test covariance(mu) == [0.5 0.; 0. 1.]
    @test innovation(mu) == 0.
    @test innovation_covariance(mu) == 2.
    @test kalman_gain(mu) == [0.5, 0.]

    mu_inter = KFMUIntermediate(2,1)
    mu = measurement_update!(mu_inter, x, P, y, H, R)
    @test state(mu) == [1., 1.]
    @test covariance(mu) == [0.5 0.; 0. 1.]
    @test innovation(mu) == 0.
    @test innovation_covariance(mu) == 2.
    @test kalman_gain(mu) == [0.5, 0.]

    y = [1., 1.]
    x = 1.
    P = 1.
    H = [1., 0.]
    R = [1. 0.; 0. 1.]

    mu = measurement_update(x, P, y, H, R)
    @test state(mu) == 1.
    @test covariance(mu) == 0.5
    @test innovation(mu) == [0.0, 1.0]
    @test innovation_covariance(mu) == [2.0 0.0; 0.0 1.0]
    @test kalman_gain(mu) == [0.5 0.0]

    mu_inter = KFMUIntermediate(1,2)
    mu = measurement_update!(mu_inter, x, P, y, H, R)
    @test state(mu) == 1.
    @test covariance(mu) == 0.5
    @test innovation(mu) == [0.0, 1.0]
    @test innovation_covariance(mu) == [2.0 0.0; 0.0 1.0]
    @test kalman_gain(mu) == [0.5 0.0]
end
