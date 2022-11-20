function get_process(T)
    [
        1 T T^2/2
        0 1 T
        0 0 1
    ]
end

function get_process_covariance(T)
    [
        T^5/20 T^4/8 T^3/6
        T^4/8 T^3/3 T^2/2
        T^3/6 T^2/2 T
    ]
end

function update_state(pos_vec_acc_state, process, process_covariance)
    num_states = length(pos_vec_acc_state)
    C = cholesky(process_covariance).L
#    U, s = svd(process_covariance)
#    C = U * Diagonal(sqrt.(s))
    process * pos_vec_acc_state + C * randn(eltype(pos_vec_acc_state), num_states)
end

function create_measurement(pos_vec_acc_state, H, R)
    H * pos_vec_acc_state + randn(eltype(pos_vec_acc_state)) * sqrt(R) # Scalar
end

function simulate_kalman_filter_over_time(;
    x, P, F, Q, R, H, num_iterations = 1000,
)
    true_states_over_time = Matrix{eltype(x)}(undef, num_iterations, length(x))
    est_states_over_time = Matrix{eltype(x)}(undef, num_iterations, length(x))
    est_state_vars_over_time = Matrix{Float64}(undef, num_iterations, length(x))
    nis_over_time = Vector{Float64}(undef, num_iterations)
    innovation_over_time = Vector{eltype(x)}(undef, num_iterations)
    innovation_vars_over_time = Vector{Float64}(undef, num_iterations)

    tu = KalmanFilters.KFTimeUpdate(x, P)
    for i = 1:num_iterations
        # Simulate measurement
        measurement = create_measurement(x, H, R)

        # Filter with Kalman Filter
        mu = measurement_update(get_state(tu), get_covariance(tu), measurement, H, R)

        # Save values
        true_states_over_time[i, :] = x
        est_states_over_time[i, :] = get_state(mu)
        est_state_vars_over_time[i, :] = diag(get_covariance(mu))
        nis_over_time[i] = calc_nis(mu)
        innovation_over_time[i] = get_innovation(mu)
        innovation_vars_over_time[i] = get_innovation_covariance(mu)

        tu = time_update(get_state(mu), get_covariance(mu), F, Q)

        # Update simulation states
        x = update_state(x, F, Q)
    end
    true_states_over_time, est_states_over_time, est_state_vars_over_time, nis_over_time, innovation_over_time, innovation_vars_over_time
end

function test_innovation(niss, innovations, innovation_vars)
    @test calc_nis_test(niss, num_measurements = size(innovations, 2))
    @test innovation_correlation_test(innovations)
    @test all(isapprox.(mean(innovations, dims = 1), 0.0, atol = 5e-3)) # 5e-3?
    if eltype(innovations) <: Complex
        @test all(calc_sigma_bound_test(real.(innovations), innovation_vars / 2; atol = 0.02))
        @test all(calc_sigma_bound_test(imag.(innovations), innovation_vars / 2; atol = 0.02))
    else
        @test all(calc_sigma_bound_test(innovations, innovation_vars; atol = 0.02))
    end
end

function test_state_errors(state_errors, est_state_vars)
    @test all(isapprox.(mean(state_errors, dims = 1), 0.0, atol = 5e-3)) # 5e-3?
    if eltype(state_errors) <: Complex
        @test all(calc_sigma_bound_test(real.(state_errors), est_state_vars / 2; atol = 0.08))
        @test all(calc_sigma_bound_test(imag.(state_errors), est_state_vars / 2; atol = 0.08))
    else
        @test all(calc_sigma_bound_test(state_errors, est_state_vars; atol = 0.08))
    end
end

@testset "Kalman filter system test" begin

    acc_std = 1e-2
    noise_std = 0.01

    T = 1e-2
    F = get_process(T)
    Q = get_process_covariance(T) * acc_std^2
    R = noise_std^2
    H = [1, 0, 0]'
    num_iterations = 4000

    x = zeros(3)
    P = collect(Diagonal([8.0e1^2, 2.0e-1^2, 2.0e-1^2]))
    
    true_states_over_time, est_states_over_time, est_state_vars_over_time, nis_over_time, innovation_over_time, innovation_vars_over_time =
        simulate_kalman_filter_over_time(; x, P, F, Q, R, H, num_iterations)

    # System is ergodic -> Monte Carlo evaluation isn't needed
    # Normalized innovation squared test
    test_innovation(nis_over_time, innovation_over_time, innovation_vars_over_time)
    
    # State error should be zero mean and 68% should be within sigma bound
    state_errors_over_time = est_states_over_time - true_states_over_time
    test_state_errors(state_errors_over_time, est_state_vars_over_time)

#=
    using Plots
    plot(true_states_over_time, layout = 3, lab = "True", ylabel = ["Position" "Velocity" "Acceleration"])
    plot!(est_states_over_time, lab = "Est.")

    plot(true_states_over_time - est_states_over_time, layout = 3, legend = false, ylabel = ["Position error" "Velocity error" "Acceleration error"])
    plot!(sqrt.(est_state_vars_over_time), color = :red)
    plot!(-sqrt.(est_state_vars_over_time), color = :red)

    plot(innovation_over_time[2:end])
    plot!(sqrt.(innovation_cov_over_time[2:end]), color = :red)
    plot!(-sqrt.(innovation_cov_over_time[2:end]), color = :red)
=#
end

@testset "Complex Kalman filter system test" begin

    acc_std = 1e-2
    noise_std = 0.01

    T = 1e-2
    F = get_process(T)
    Q = get_process_covariance(T) * acc_std^2
    R = noise_std^2
    H = [1, 0, 0]'
    num_iterations = 4000

    x = zeros(ComplexF64, 3)
    P = collect(Diagonal([8.0e1^2, 2.0e-1^2, 2.0e-1^2]))
    
    true_states_over_time, est_states_over_time, est_state_vars_over_time, nis_over_time, innovation_over_time, innovation_vars_over_time =
        simulate_kalman_filter_over_time(; x, P, F, Q, R, H, num_iterations)

    # System is ergodic -> Monte Carlo evaluation isn't needed
    # Normalized innovation squared test
    test_innovation(nis_over_time, innovation_over_time, innovation_vars_over_time)
    
    # State error should be zero mean and 68% should be within sigma bound
    state_errors_over_time = est_states_over_time - true_states_over_time
    test_state_errors(state_errors_over_time, est_state_vars_over_time)

#=
    using Plots
    plot(real.(true_states_over_time), layout = 3, lab = "True real", ylabel = ["Position" "Velocity" "Acceleration"])
    plot!(real.(est_states_over_time), lab = "Est. real")
    plot!(imag.(true_states_over_time), lab = "True imag")
    plot!(imag.(est_states_over_time), lab = "Est. imag")

    plot(real.(true_states_over_time - est_states_over_time), layout = 3, legend = false, ylabel = ["Position error" "Velocity error" "Acceleration error"])
    plot!(sqrt.(real.(est_state_vars_over_time)), color = :red)
    plot!(-sqrt.(real.(est_state_vars_over_time)), color = :red)

    plot(real.(innovation_over_time[2:end]))
    plot!(sqrt.(real.(innovation_cov_over_time[2:end])), color = :red)
    plot!(-sqrt.(real.(innovation_cov_over_time[2:end])), color = :red)
=#
end