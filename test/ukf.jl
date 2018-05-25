@testset "Sigma points" begin
    χ = KalmanFilter.calc_sigma_points(𝐱, 𝐏, scales)
    @test χ * KalmanFilter.mean_weights(scales, 2) ≈ 𝐱
    @test (χ .- 𝐱) .* KalmanFilter.cov_weights(scales, 2)' * (χ .- 𝐱)' ≈ 𝐏
end

@testset "UKF time update" begin
    χ = KalmanFilter.calc_sigma_points(𝐱, 𝐏, scales)
    χ_next, 𝐱_next, 𝐏_next = KalmanFilter._time_update(χ, scales, 𝐟, eye(2))
    @test χ_next == χ
    @test 𝐱_next ≈ 𝐱
    @test 𝐏_next ≈ 𝐏 + eye(2)
end

@testset "UKF measurement update" begin
    χ = KalmanFilter.calc_sigma_points(𝐱, 𝐏, scales)
    𝐱_next, 𝐏_next, 𝐲̃, 𝐏yy = KalmanFilter._measurement_update(χ, 𝐱, 𝐏, scales, 𝐲, 𝐡, 0)
    @test 𝐲̃ ≈ zeros(2) rtol = 1
    @test 𝐏yy ≈ 𝐏
    @test 𝐱_next ≈ 𝐱
    @test 𝐏_next ≈ zeros(2,2) rtol = 1
end

@testset "UKF time update without augmentation" begin
    used_states = trues(2)
    measurement_update = KalmanFilter.time_update(𝐱, 𝐏, 𝐱, 𝐏, scales, 𝐟, 𝐐, used_states, false)
    time_update, 𝐱_next, 𝐏_next, 𝐲̃, 𝐏yy = measurement_update(𝐲, 𝐇, 𝐑)
    @test 𝐱_next ≈ 𝐱 
end

@testset "Unscented Kalman filter: system test" begin
    time_update = KalmanFilter.init_kalman(𝐱, 𝐏)
    measurement_update = time_update(x -> [x[1] + 0.1 * x[2]; x[2]], 𝐐)
    time_update, 𝐱_next, 𝐏_next = measurement_update(5, x -> x[1], 0.1)
    measurement_update = time_update(x -> [x[1] + 0.1 * x[2]; x[2]], 𝐐)
    time_update, 𝐱_next, 𝐏_next = measurement_update(5, x -> x[1], 0.1)
    @test 𝐱_next[1] ≈ 5 atol = 0.05
end