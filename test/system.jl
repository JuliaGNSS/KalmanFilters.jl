@testset "Kalman filter system test" begin
    start_pt = 19
    start_vel = 2
    start_acc = 1
    σ_acc_noise = 0.0001
    σ_meas_noise = 0.25
    Δt = 0.1

    function measure(state, Δt, σ_meas_noise, σ_acc_noise)
        s, v, a = state[1], state[2], state[3]
        a_next = a + randn() * σ_acc_noise
        v_next = v + a * Δt
        s_next = a * Δt^2 / 2 + v * Δt + s
        s_next + randn() * σ_meas_noise, (s_next, v_next, a_next)
    end

    # State space matrices of discretized white noise acceleration model
    F = [1 Δt Δt^2/2; 0 1 Δt; 0 0 1]
    H = [1, 0, 0]'
    Q = [Δt^2/2; Δt; 1] * [Δt^2/2 Δt 1] * σ_acc_noise^2
    R = σ_meas_noise^2

    # Initialization
    maxiter = 20000
    counter = 1
    x_init = [0.0, 0.0, 0.0]
    P_init = [2.5 0.25 0.1; 0.25 2.5 0.2; 0.1 0.2 2.5]
    ỹ_over_time = Vector{Float64}(undef, maxiter)
    S_over_time = Vector{Float64}(undef, maxiter)
    kalman_inits = KalmanInits(x_init, P_init)
    states = (start_pt, start_vel, start_acc)
    s_over_time = Vector{Float64}(undef, maxiter)
    z_over_time = Vector{Float64}(undef, maxiter)
    s̃_over_time = Vector{Float64}(undef, maxiter)
    # run Kalman Filter
    time_update_results = time_update(kalman_inits, F, Q)
    for i = 1:maxiter
        measurement, states = measure(states, Δt, σ_meas_noise, σ_acc_noise)
        measurement_update_results = measurement_update(time_update_results, measurement, H, R)
        time_update_results = time_update(measurement_update_results, F, Q)

        #s̃_over_time[counter] = state(measurement_update_results)[1]
        #s_over_time[counter] = states[1]
        #z_over_time[counter] = measurement
        ỹ_over_time[counter] = innovation(measurement_update_results)
        S_over_time[counter] = innovation_covariance(measurement_update_results)
        counter += 1
    end
    #using PyPlot
    #figure()
    #plot(s_over_time)
    #plot(z_over_time)
    #plot(s̃_over_time)
    #plot(ỹ_over_time)
    #plot(s_over_time .- s̃_over_time)
    #legend(("Ground truth", "Messung", "Schätzung", "Innovation", "Error"))

    # Statistical consistency testing
    @test sigma_bound_test(ỹ_over_time[4:end], S_over_time[4:end]) == true
    @test two_sigma_bound_test(ỹ_over_time[4:end], S_over_time[4:end]) == true

    window_start = 4
    window_length = 400
    window = window_start:window_start + window_length - 1
    dof = length(window) * size(ỹ_over_time[window_start], 1)
    nis_over_time_sys = map((x, σ²) -> nis(x, σ²), ỹ_over_time[window], S_over_time[window])
    result_nis_test = nis_test(nis_over_time_sys, dof)
    @test result_nis_test == true
end
