@testset "Kalman Filter: system test" begin

    time_update = @inferred KalmanFilter.init_kalman(𝐱, 𝐏)
    measurement_update = @inferred time_update(𝐅, 𝐐)
    time_update, 𝐱_next, 𝐏_next = @inferred measurement_update(5, 𝐇, 0.1)
    measurement_update = @inferred time_update(𝐅, 𝐐)
    time_update, 𝐱_next, 𝐏_next = @inferred measurement_update(5, 𝐇, 0.1)
    @test 𝐱_next[1] ≈ 5 atol = 0.05
end

@testset "KF time update KF measurement update without augmentation" begin
    measurement_update = @inferred KalmanFilter.time_update(𝐱, 𝐏, 𝐱, 𝐏, scales, 𝐅, 𝐐, used_states, false)
    time_update, 𝐱_next, 𝐏_next, 𝐲̃, 𝐏yy = @inferred measurement_update(𝐲, 𝐇, 𝐑)
    @test 𝐏_next ≈ Matrix(Diagonal([3/4, 4/5]))
    @test 𝐱_next ≈ 𝐱
end

@testset "KF time update KF measurement update with augmented 𝐐" begin
    measurement_update = @inferred KalmanFilter.time_update(𝐱, 𝐏, 𝐱, 𝐏, scales, [𝐅 I], Augment(𝐐), used_states, false)
    time_update, 𝐱_next, 𝐏_next, 𝐲̃, 𝐏yy = @inferred measurement_update(𝐲, 𝐇, 𝐑)
    @test 𝐏_next ≈ Matrix(Diagonal([3/4, 4/5]))
    @test 𝐱_next ≈ 𝐱
end

@testset "KF time update KF measurement update with augmented 𝐑" begin
    measurement_update = @inferred KalmanFilter.time_update(𝐱, 𝐏, 𝐱, 𝐏, scales, 𝐅, 𝐐, used_states, false)
    time_update, 𝐱_next, 𝐏_next, 𝐲̃, 𝐏yy = @inferred measurement_update(𝐲, [𝐇 I], Augment(𝐑))
    @test 𝐏_next ≈ Matrix(Diagonal([3/4, 4/5]))
    @test 𝐱_next ≈ 𝐱
end

@testset "KF time update KF measurement update with augmented 𝐐 and 𝐑" begin
    measurement_update = @inferred KalmanFilter.time_update(𝐱, 𝐏, 𝐱, 𝐏, scales, [𝐅 I zeros(2,2); zeros(2,4) I], Augment(𝐐), Augment(𝐑), used_states, false)
    time_update, 𝐱_next, 𝐏_next, 𝐲̃, 𝐏yy = @inferred measurement_update(𝐲, [𝐇 I])
    @test 𝐏_next ≈ Matrix(Diagonal([3/4, 4/5]))
    @test 𝐱_next ≈ 𝐱
end
