using Base.Test, KalmanFilter

srand(1234)

@testset "Sigma points" begin

    x = [0, 1]
    P = diagm([2, 3])
    scales = ScalingParameters(1, 2, 0)
    χ = KalmanFilter.calc_sigma_points(x, P, scales)
    @test mean(χ, 2) ≈ x
    @test var(χ, 2) ≈ diag(P)
end

@testset "Kalman system" begin

    𝐱 = [0, 1]
    𝐏 = diagm([2, 3])
    𝐓 = [1 0.1; 0 1]
    𝐐 = diagm([0.25, 0.25])
    time_update = KalmanFilter.init_kalman(𝐱, 𝐏)
    measurement_update = time_update(𝐓, 𝐐)
    time_update, 𝐱, 𝐏 = measurement_update(5, x -> x[1], 0.1)
    measurement_update = time_update(𝐓, 𝐐)
    time_update, 𝐱, 𝐏 = measurement_update(5, x -> x[1], 0.1)
    @test 𝐱[1] ≈ 5 atol = 0.01
end

@testset "Measurement augmentation" begin

    𝐱 = [0, 1]
    𝐏 = diagm([2, 3])
    𝐓 = [1 0.1; 0 1]
    𝐐 = diagm([0.25, 0.25])
    time_update = KalmanFilter.init_kalman(𝐱, 𝐏)
    measurement_update = time_update(𝐓, 𝐐)
    time_update1, 𝐱, 𝐏 = measurement_update(5, x -> x[1], 0.1)
    time_update2, 𝐱_aug, 𝐏_aug = measurement_update(5, x -> x[1] + x[3], Augment(0.1))
    @test 𝐱 ≈ 𝐱_aug
    @test 𝐏 ≈ 𝐏_aug
end

@testset "Transition augmentation" begin

    𝐱 = [0, 1]
    𝐏 = diagm([2, 3])
    𝐓 = [1 0.1; 0 1]
    𝐐 = diagm([0.25, 0.25])
    time_update = KalmanFilter.init_kalman(𝐱, 𝐏)
    measurement_update = time_update(𝐓, 𝐐)
    measurement_update_aug = time_update([𝐓 eye(2)], Augment(𝐐))
    time_update1, 𝐱, 𝐏 = measurement_update(5, x -> x[1], 0.1)
    time_update2, 𝐱_aug, 𝐏_aug = measurement_update_aug(5, x -> x[1], 0.1)
    @test 𝐱 ≈ 𝐱_aug
    @test 𝐏 ≈ 𝐏_aug
end

@testset "Transition and measurement augmentation" begin

    𝐱 = [0, 1]
    𝐏 = diagm([2, 3])
    𝐓 = [1 0.1; 0 1]
    𝐐 = diagm([0.25, 0.25])
    time_update = KalmanFilter.init_kalman(𝐱, 𝐏)
    measurement_update = time_update(𝐓, 𝐐)
    measurement_update_aug = time_update([𝐓 eye(2) zeros(2); 0 0 0 0 1], Augment(𝐐), Augment(0.1))
    time_update1, 𝐱, 𝐏 = measurement_update(5, x -> x[1], 0.1)
    time_update2, 𝐱_aug, 𝐏_aug = measurement_update_aug(5, x -> x[1] + x[3])
    @test 𝐱 ≈ 𝐱_aug
    @test 𝐏 ≈ 𝐏_aug
end