srand(1234)

@testset "Kalman Filter System Test" begin
    start_pt = 19
    start_vel = 2
    start_acc = 1
    σ_acc_noise = 1.0
    σ_meas_noise = 1.25

    function init_measurement(start_acc, start_vel, start_pt, Δt, σ_meas_noise, σ_acc_noise)
        𝐱 = start_pt
        𝐯 = start_vel
        𝐚 = start_acc
        () -> begin
            noise_acc = randn() * σ_acc_noise
            # # without random walk behaviour
            # 𝐱 = 0.5 * (𝐚 + noise_acc) * Δt^2 + (𝐯 + noise_acc * Δt) * Δt + 𝐱
            # 𝐯 = 𝐯 + Δt * 𝐚
            # 𝐚 = 𝐚

            # incl. random walk behaviour
            𝐱 = 0.5 * (𝐚 + noise_acc) * Δt^2 + 𝐯 * Δt + 𝐱
            𝐯 = 𝐯 + Δt * (𝐚 + noise_acc)
            𝐚 = 𝐚 + noise_acc

            𝐱, 𝐱 + randn() * σ_meas_noise
        end
    end

    Δt = 0.1
    measurement = init_measurement(start_acc, start_vel, start_pt, Δt, σ_meas_noise, σ_acc_noise)

    # State space matrices of discretized white noise acceleration model
    𝐅_sys = [1 Δt 0.5*Δt^2; 0 1 Δt; 0 0 1]
    𝐇_sys = [1 0 0]
    𝐐_sys = [Δt^2/2; Δt; 1] * [Δt^2/2 Δt 1] * σ_acc_noise^2
    𝐑_sys = σ_meas_noise^2

    # Initialization
    maxiter = 20000
    range = 1:Δt:floor(maxiter * Δt) + (1 - Δt)
    counter = 1
    𝐱_init = [0 0 0]'
    𝐏_init = [2.5 0.25 0.1; 0.25 2.5 0.2; 0.1 0.2 2.5]
    𝐲̃_over_time = Vector(length(range))
    𝐒_over_time = Vector{Matrix{Float64}}(length(range))
    time_update = KalmanFilter.init_kalman(𝐱_init, 𝐏_init)

    # run Kalman Filter
    for i = range
        measurement_update = time_update(𝐅_sys, 𝐐_sys)
        𝐲_sys, 𝐳_sys = measurement()
        time_update, 𝐱_next, 𝐏_next, 𝐲̃, 𝐒 = measurement_update(𝐳_sys, 𝐇_sys, 𝐑_sys)

        𝐲̃_over_time[counter] = 𝐲̃
        𝐒_over_time[counter] = 𝐒
        counter += 1
    end

    # Statistical consistency testing
    @test sigma_bound_test(𝐲̃_over_time[4:end], 𝐒_over_time[4:end]) == [true]
    @test two_sigma_bound_test(𝐲̃_over_time[4:end], 4 .* 𝐒_over_time[4:end]) == [true]

    window_start = 4
    window_length = 400
    window = window_start:window_start + window_length - 1
    dof = length(window) * size(𝐲̃_over_time[window_start], 1)
    nis_over_time_sys = map((𝐱, σ²) -> nis(𝐱, σ²), 𝐲̃_over_time[window], 𝐒_over_time[window])
    result_nis_test = nis_test(nis_over_time_sys, dof)
    @test result_nis_test == true
end
