@testset "Kalman Filter: system test" begin

    𝐱 = [0, 1]
    𝐏 = diagm([2, 3])
    𝐓 = [1 0.1; 0 1]
    𝐇 = [1 0]
    𝐐 = diagm([0.25, 0.25])
    time_update = KalmanFilter.init_kalman(𝐱, 𝐏)
    measurement_update = time_update(𝐓, 𝐐)
    time_update, 𝐱, 𝐏 = measurement_update(5, 𝐇, 0.1)
    measurement_update = time_update(𝐓, 𝐐)
    time_update, 𝐱, 𝐏 = measurement_update(5, 𝐇, 0.1)
    @test 𝐱[1] ≈ 5 atol = 0.01
end