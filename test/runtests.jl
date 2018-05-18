using Base.Test, KalmanFilter

srand(1234)

@testset "Sigma points" begin

    x = [0, 1]
    P = diagm([2, 3])
    weights = Weights(1, 2, 0)
    iweights = KalmanFilter.InternalWeights(weights, length(x))
    χ = KalmanFilter.calc_sigma_points(x, P, iweights)
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
    time_update, 𝐱, 𝐏 = measurement_update(5 + randn() * 0.1, x -> x[1], 0.1)
    measurement_update = time_update(𝐓, 𝐐)
    time_update, 𝐱, 𝐏 = measurement_update(5 + randn() * 0.1, x -> x[1], 0.1)
    @test 𝐱[1] ≈ 5 atol = 0.2
end