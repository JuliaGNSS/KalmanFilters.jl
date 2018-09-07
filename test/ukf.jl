@testset "Sigma points" begin
    χ = @inferred KalmanFilter.calc_sigma_points(𝐱, 𝐏, scales)
    @test χ * KalmanFilter.mean_weights(scales, 2) ≈ 𝐱
    @test (χ .- 𝐱) .* KalmanFilter.cov_weights(scales, 2)' * (χ .- 𝐱)' ≈ 𝐏
end

@testset "UKF time update" begin
    χ = @inferred KalmanFilter.calc_sigma_points(𝐱, 𝐏, scales)
    χ_next, 𝐱_next, 𝐏_next = @inferred KalmanFilter._time_update(χ, scales, 𝐟, Matrix{Float64}(I, 2,2))
    @test χ_next == χ
    @test 𝐱_next ≈ 𝐱
    @test 𝐏_next ≈ 𝐏 + I
end

@testset "UKF measurement update" begin
    χ = @inferred KalmanFilter.calc_sigma_points(𝐱, 𝐏, scales)
    𝐱_next, 𝐏_next, 𝐲̃, 𝐏yy = @inferred KalmanFilter._measurement_update(χ, 𝐱, 𝐏, scales, 𝐲, 𝐡, 0)
    @test 𝐲̃ ≈ zeros(2) rtol = 1
    @test 𝐏yy ≈ 𝐏
    @test 𝐱_next ≈ 𝐱
    @test 𝐏_next ≈ zeros(2,2) rtol = 1
end

@testset "UKF time update UKF measurement update without augmentation" begin
    measurement_update = @inferred KalmanFilter.time_update(𝐱, 𝐏, 𝐱, 𝐏, scales, 𝐟, 𝐐, used_states, false)
    time_update, 𝐱_next, 𝐏_next, 𝐲̃, 𝐏yy = @inferred measurement_update(𝐲, 𝐡, 𝐑)
    @test 𝐏_next ≈ Matrix(Diagonal([5/3, 7/4])) # ??
    @test 𝐱_next ≈ 𝐱
end

@testset "UKF time update UKF measurement update with augmented 𝐐" begin
    measurement_update = @inferred KalmanFilter.time_update(𝐱, 𝐏, 𝐱, 𝐏, scales, x -> [x[1] + x[3]; x[2] + x[4]], Augment(𝐐), used_states, false)
    time_update, 𝐱_next, 𝐏_next, 𝐲̃, 𝐏yy = @inferred measurement_update(𝐲, 𝐡, 𝐑)
    @test 𝐏_next ≈ Matrix(Diagonal([3/4, 4/5]))
    @test 𝐱_next ≈ 𝐱
end

@testset "UKF time update UKF measurement update with augmented 𝐐 and 𝐑" begin
    measurement_update = @inferred KalmanFilter.time_update(𝐱, 𝐏, 𝐱, 𝐏, scales, x -> [x[1] + x[3]; x[2] + x[4]; x[5]; x[6]], Augment(𝐐), Augment(𝐑), used_states, false)
    time_update, 𝐱_next, 𝐏_next, 𝐲̃, 𝐏yy = @inferred measurement_update(𝐲, x -> [x[1] + x[3]; x[2] + x[4]])
    @test 𝐏_next ≈ Matrix(Diagonal([3/4, 4/5]))
    @test 𝐱_next ≈ 𝐱
end

@testset "KF time update UKF measurement update without augmentation" begin
    measurement_update = @inferred KalmanFilter.time_update(𝐱, 𝐏, 𝐱, 𝐏, scales, 𝐅, 𝐐, used_states, false)
    time_update, 𝐱_next, 𝐏_next, 𝐲̃, 𝐏yy = @inferred measurement_update(𝐲, 𝐡, 𝐑)
    @test 𝐏_next ≈ Matrix(Diagonal([3/4, 4/5]))
    @test 𝐱_next ≈ 𝐱
end

@testset "KF time update UKF measurement update with augmented 𝐐" begin
    measurement_update = @inferred KalmanFilter.time_update(𝐱, 𝐏, 𝐱, 𝐏, scales, [𝐅 I], Augment(𝐐), used_states, false)
    time_update, 𝐱_next, 𝐏_next, 𝐲̃, 𝐏yy = @inferred measurement_update(𝐲, 𝐡, 𝐑)
    @test 𝐏_next ≈ Matrix(Diagonal([3/4, 4/5]))
    @test 𝐱_next ≈ 𝐱
end

@testset "KF time update UKF measurement update with augmented 𝐑" begin
    measurement_update = @inferred KalmanFilter.time_update(𝐱, 𝐏, 𝐱, 𝐏, scales, 𝐅, 𝐐, used_states, false)
    time_update, 𝐱_next, 𝐏_next, 𝐲̃, 𝐏yy = @inferred measurement_update(𝐲, x -> [x[1] + x[3]; x[2] + x[4]], Augment(𝐑))
    @test 𝐏_next ≈ Matrix(Diagonal([3/4, 4/5])) # ??
    @test 𝐱_next ≈ 𝐱
end

@testset "KF time update UKF measurement update with augmented 𝐐 and 𝐑" begin
    measurement_update = @inferred KalmanFilter.time_update(𝐱, 𝐏, 𝐱, 𝐏, scales, [𝐅 I zeros(2,2); zeros(2,4) I], Augment(𝐐), Augment(𝐑), used_states, false)
    time_update, 𝐱_next, 𝐏_next, 𝐲̃, 𝐏yy = @inferred measurement_update(𝐲, x -> [x[1] + x[3]; x[2] + x[4]])
    @test 𝐏_next ≈ Matrix(Diagonal([3/4, 4/5]))
    @test 𝐱_next ≈ 𝐱
end

@testset "UKF time update KF measurement update without augmentation" begin
    measurement_update = @inferred KalmanFilter.time_update(𝐱, 𝐏, 𝐱, 𝐏, scales, 𝐟, 𝐐, used_states, false)
    time_update, 𝐱_next, 𝐏_next, 𝐲̃, 𝐏yy = @inferred measurement_update(𝐲, 𝐇, 𝐑)
    @test 𝐏_next ≈ Matrix(Diagonal([3/4, 4/5]))
    @test 𝐱_next ≈ 𝐱
end

@testset "UKF time update KF measurement update with augmented 𝐐" begin
    measurement_update = @inferred KalmanFilter.time_update(𝐱, 𝐏, 𝐱, 𝐏, scales, x -> [x[1] + x[3]; x[2] + x[4]], Augment(𝐐), used_states, false)
    time_update, 𝐱_next, 𝐏_next, 𝐲̃, 𝐏yy = @inferred measurement_update(𝐲, 𝐇, 𝐑)
    @test 𝐏_next ≈ Matrix(Diagonal([3/4, 4/5]))
    @test 𝐱_next ≈ 𝐱
end

@testset "UKF time update KF measurement update with augmented 𝐑" begin
    measurement_update = @inferred KalmanFilter.time_update(𝐱, 𝐏, 𝐱, 𝐏, scales, 𝐟, 𝐐, used_states, false)
    time_update, 𝐱_next, 𝐏_next, 𝐲̃, 𝐏yy = @inferred measurement_update(𝐲, [𝐇 I], Augment(𝐑))
    @test 𝐏_next ≈ Matrix(Diagonal([3/4, 4/5]))
    @test 𝐱_next ≈ 𝐱
end

@testset "UKF time update KF measurement update with augmented 𝐐 and 𝐑" begin
    measurement_update = @inferred KalmanFilter.time_update(𝐱, 𝐏, 𝐱, 𝐏, scales, x -> [x[1] + x[3]; x[2] + x[4]; x[5]; x[6]], Augment(𝐐), Augment(𝐑), used_states, false)
    time_update, 𝐱_next, 𝐏_next, 𝐲̃, 𝐏yy = @inferred measurement_update(𝐲, [𝐇 I])
    @test 𝐏_next ≈ Matrix(Diagonal([3/4, 4/5]))
    @test 𝐱_next ≈ 𝐱
end

@testset "Unscented Kalman filter: system test" begin
    time_update = @inferred KalmanFilter.init_kalman(𝐱, 𝐏)
    measurement_update = @inferred time_update(x -> [x[1] + 0.1 * x[2]; x[2]], 𝐐)
    time_update, 𝐱_next, 𝐏_next = @inferred measurement_update(5, x -> x[1], 0.1)
    measurement_update = @inferred time_update(x -> [x[1] + 0.1 * x[2]; x[2]], 𝐐)
    time_update, 𝐱_next, 𝐏_next = @inferred measurement_update(5, x -> x[1], 0.1)
    @test 𝐱_next[1] ≈ 5 atol = 0.05
end
