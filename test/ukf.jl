@testset "Sigma points" begin
    𝐱 = [0, 1]
    𝐏 = diagm([2, 3])
    scales = ScalingParameters(1e-3, 2, 0)
    χ = KalmanFilter.calc_sigma_points(𝐱, 𝐏, scales)
    @test χ * KalmanFilter.mean_weights(scales, 2) ≈ 𝐱
    @test (χ .- 𝐱) .* KalmanFilter.cov_weights(scales, 2)' * (χ .- 𝐱)' ≈ 𝐏
end

@testset "UKF time update" begin
    scales = ScalingParameters(1e-3, 2, 0)
    𝐱 = [0, 1]
    𝐏 = diagm([2, 3])
    χ = KalmanFilter.calc_sigma_points(𝐱, 𝐏, scales)
    𝐟(x) = x
    χ_next, 𝐱_next, 𝐏_next = KalmanFilter._time_update(χ, scales, 𝐟, eye(2))
    @test χ_next == χ
    @test 𝐱_next ≈ 𝐱
    @test 𝐏_next ≈ 𝐏 + eye(2)
end

@testset "UKF measurement update" begin
    scales = ScalingParameters(1e-3, 2, 0)
    𝐱 = [0, 1]
    𝐲 = 𝐱
    𝐏 = diagm([2, 3])
    χ = KalmanFilter.calc_sigma_points(𝐱, 𝐏, scales)
    h(x) = x
    𝐱_next, 𝐏_next, 𝐲̂, 𝐏yy = KalmanFilter._measurement_update(χ, 𝐱, 𝐏, scales, 𝐲, h, 0)
    @test 𝐲̂ ≈ zeros(2) rtol = 1
    @test 𝐏yy ≈ 𝐏
    @test 𝐱_next ≈ 𝐱
    @test 𝐏_next ≈ zeros(2,2) rtol = 1
end

@testset "Unscented Kalman filter: system test" begin

    𝐱 = [0, 1]
    𝐏 = diagm([2, 3])
    𝐐 = diagm([0.25, 0.25])
    time_update = KalmanFilter.init_kalman(𝐱, 𝐏)
    measurement_update = time_update(x -> [x[1] + 0.1 * x[2]; x[2]], 𝐐)
    time_update, 𝐱, 𝐏 = measurement_update(5, x -> x[1], 0.1)
    measurement_update = time_update(x -> [x[1] + 0.1 * x[2]; x[2]], 𝐐)
    time_update, 𝐱, 𝐏 = measurement_update(5, x -> x[1], 0.1)
    @test 𝐱[1] ≈ 5 atol = 0.02
end