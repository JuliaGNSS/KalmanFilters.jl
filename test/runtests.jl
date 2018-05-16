using Base.Test, KalmanFilter

@testset "Sigma points" begin

    x = [0, 1]
    P = diagm([2, 3])
    weights = Weights(1, 2, 0)
    iweights = KalmanFilter.InternalWeights(weights, length(x))
    χ = KalmanFilter.calc_sigma_points(x, P, iweights)
    @test mean(χ, 2) ≈ x
    @test var(χ, 2) ≈ diag(P)
end
