@testset "Kalman Filter: system test" begin

    time_update = KalmanFilter.init_kalman(𝐱, 𝐏)
    measurement_update = time_update(𝐅, 𝐐)
    time_update, 𝐱_next, 𝐏_next = measurement_update(5, 𝐇, 0.1)
    measurement_update = time_update(𝐅, 𝐐)
    time_update, 𝐱_next, 𝐏_next = measurement_update(5, 𝐇, 0.1)
    @test 𝐱_next[1] ≈ 5 atol = 0.05
end