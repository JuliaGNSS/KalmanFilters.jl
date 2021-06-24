using
    BenchmarkTools,
    LinearAlgebra,
    KalmanFilter,
    Plots,
    LazyArrays,
    Colors

function init_mu(num_states, num_measures)
    x = randn(num_states)
    PL = randn(num_states, num_states)
    P = PL'PL
    RL = randn(num_measures, num_measures)
    R = RL'RL
    y = randn(num_measures)
    H = randn(num_measures, num_states)
    P_chol = cholesky(P)
    R_chol = cholesky(R)
    h(x) = H * x
    h!(y, x) = mul!(y, H, x)
    h(x, noise) = H * x .+ noise
    h!(y, x, noise) = y .= @~ H * x .+ noise
    return x, y, P, H, R, P_chol, R_chol, h, h!
end

function init_tu(num_states)
    x = randn(num_states)
    A = randn(num_states, num_states)
    P = A'A
    B = randn(num_states, num_states)
    Q = B'B
    F = randn(num_states, num_states)
    P_chol = cholesky(P)
    Q_chol = cholesky(Q)
    f(x) = F * x
    f!(y, x) = mul!(y, F, x)
    f(x, noise) = F * x .+ noise
    f!(y, x, noise) = y.= @~ F * x .+ noise
    return x, P, Q, F, P_chol, Q_chol, f, f!
end

function run_measurement_update_benchmarks(num_state_tests, num_measurement_tests; allocation = false)

    num_measurements = length(num_measurement_tests)
    kf_types = (:kf, :srkf, :ukf, :srukf, :aukf, :sraukf)
    buffers = [(inplace = zeros(length(num_state_tests), num_measurements), allocating = zeros(length(num_state_tests), num_measurements)) for _ = 1:length(kf_types)]
    results = NamedTuple{kf_types}(buffers)

    for (i, num_states) in enumerate(num_state_tests)
        for (j, num_measures) in enumerate(num_measurement_tests)
            x, y, P, H, R, P_chol, R_chol, h, h! = init_mu(num_states, num_measures)

            results.kf.allocating[i,j] = if allocation
                @allocated measurement_update(x, P, y, H, R)
            else
                @belapsed measurement_update($x, $P, $y, $H, $R)
            end
            kf_inter = KFMUIntermediate(num_states, num_measures)
            results.kf.inplace[i,j] = if allocation
                @allocated measurement_update!(kf_inter, x, P, y, H, R)
            else
                @belapsed measurement_update!($kf_inter, $x, $P, $y, $H, $R)
            end

            results.srkf.allocating[i,j] = if allocation
                @allocated measurement_update(x, P_chol, y, H, R_chol)
            else
                @belapsed measurement_update($x, $P_chol, $y, $H, $R_chol)
            end
            srkf_inter = SRKFMUIntermediate(num_states, num_measures)
            results.srkf.inplace[i,j] = if allocation
                @allocated measurement_update!(srkf_inter, x, P_chol, y, H, R_chol)
            else 
                @belapsed measurement_update!($srkf_inter, $x, $P_chol, $y, $H, $R_chol)
            end

            results.ukf.allocating[i,j] = if allocation
                @allocated measurement_update(x, P, y, h, R)
            else
                @belapsed measurement_update($x, $P, $y, $h, $R)
            end
            ukf_inter = UKFMUIntermediate(num_states, num_measures)
            results.ukf.inplace[i,j] = if allocation
                @allocated measurement_update!(ukf_inter, x, P, y, h!, R)
            else
                @belapsed measurement_update!($ukf_inter, $x, $P, $y, $h!, $R)
            end

            results.srukf.allocating[i,j] = if allocation
                @allocated measurement_update(x, P_chol, y, h, R_chol)
            else
                @belapsed measurement_update($x, $P_chol, $y, $h, $R_chol)
            end
            srukf_inter = SRUKFMUIntermediate(num_states, num_measures)
            results.srukf.inplace[i,j] = if allocation
                @allocated measurement_update!(srukf_inter, x, P_chol, y, h!, R_chol)
            else
                @belapsed measurement_update!($srukf_inter, $x, $P_chol, $y, $h!, $R_chol)
            end

            results.aukf.allocating[i,j] = if allocation
                @allocated measurement_update(x, P, y, h, Augment(R))
            else
                @belapsed measurement_update($x, $P, $y, $h, $(Augment(R)))
            end
            aukf_inter = AUKFMUIntermediate(num_states, num_measures)
            results.aukf.inplace[i,j] = if allocation
                @allocated measurement_update!(aukf_inter, x, P, y, h!, Augment(R))
            else
                @belapsed measurement_update!($aukf_inter, $x, $P, $y, $h!, $(Augment(R)))
            end

            results.sraukf.allocating[i,j] = if allocation 
                @allocated measurement_update(x, P_chol, y, h, Augment(R_chol))
            else
                @belapsed measurement_update($x, $P_chol, $y, $h, $(Augment(R_chol)))
            end
            sraukf_inter = SRAUKFMUIntermediate(num_states, num_measures)
            results.sraukf.inplace[i,j] = if allocation
                @allocated measurement_update!(sraukf_inter, x, P_chol, y, h!, Augment(R_chol))
            else
                @belapsed measurement_update!($sraukf_inter, $x, $P_chol, $y, $h!, $(Augment(R_chol)))
            end
        end
    end
    results
end

function run_time_update_benchmarks(num_state_tests; allocation = false)

    kf_types = (:kf, :srkf, :ukf, :srukf, :aukf, :sraukf)
    buffers = [(inplace = zeros(length(num_state_tests)), allocating = zeros(length(num_state_tests))) for _ = 1:length(kf_types)]
    results = NamedTuple{kf_types}(buffers)

    for (i, num_states) in enumerate(num_state_tests)
        x, P, Q, F, P_chol, Q_chol, f, f! = init_tu(num_states)

        results.kf.allocating[i] = if allocation
            @allocated time_update(x, P, F, Q)
        else
            @belapsed time_update($x, $P, $F, $Q)
        end
        kf_inter = KFTUIntermediate(num_states)
        results.kf.inplace[i] = if allocation
            @allocated time_update!(kf_inter, x, P, F, Q)
        else
            @belapsed time_update!($kf_inter, $x, $P, $F, $Q)
        end
    
        results.srkf.allocating[i] = if allocation
            @allocated time_update(x, P_chol, F, Q_chol)
        else
            @belapsed time_update($x, $P_chol, $F, $Q_chol)
        end
        srkf_inter = SRKFTUIntermediate(num_states)
        results.srkf.inplace[i] = if allocation
            @allocated time_update!(srkf_inter, x, P_chol, F, Q_chol)
        else
            @belapsed time_update!($srkf_inter, $x, $P_chol, $F, $Q_chol)
        end
    
        results.ukf.allocating[i] = if allocation
            @allocated time_update(x, P, f, Q)
        else
            @belapsed time_update($x, $P, $f, $Q)
        end
        ukf_inter = UKFTUIntermediate(num_states)
        results.ukf.inplace[i] = if allocation
            @allocated time_update!(ukf_inter, x, P, f!, Q)
        else
            @belapsed time_update!($ukf_inter, $x, $P, $f!, $Q)
        end
    
        results.srukf.allocating[i] = if allocation
            @allocated time_update(x, P_chol, f, Q_chol)
        else
            @belapsed time_update($x, $P_chol, $f, $Q_chol)
        end
        srukf_inter = SRUKFTUIntermediate(num_states)
        results.srukf.inplace[i] = if allocation
            @allocated time_update!(srukf_inter, x, P_chol, f!, Q_chol)
        else
            @belapsed time_update!($srukf_inter, $x, $P_chol, $f!, $Q_chol)
        end
    
        results.aukf.allocating[i] = if allocation
            @allocated time_update(x, P, f, (Augment(Q)))
        else
            @belapsed time_update($x, $P, $f, $(Augment(Q)))
        end
        aukf_inter = AUKFTUIntermediate(num_states)
        results.aukf.inplace[i] = if allocation
            @allocated time_update!(aukf_inter, x, P, f!, Augment(Q))
        else
            @belapsed time_update!($aukf_inter, $x, $P, $f!, $(Augment(Q)))
        end

        results.sraukf.allocating[i] = if allocation
            @allocated time_update(x, P_chol, f, Augment(Q_chol))
        else
            @belapsed time_update($x, $P_chol, $f, $(Augment(Q_chol)))
        end
        sraukf_inter = SRAUKFTUIntermediate(num_states)
        results.sraukf.inplace[i] = if allocation
            @allocated time_update!(sraukf_inter, x, P_chol, f!, Augment(Q_chol))
        else
            @belapsed time_update!($sraukf_inter, $x, $P_chol, $f!, $(Augment(Q_chol)))
        end
    end
    results
end

function plot_benchmarks(results, num_state_tests; num_measurement_tests = 1:1, ylabel = "Time (μs)", scale = 10^-6, logscale = false)
    colors = distinguishable_colors(length(results), [RGB(1,1,1), RGB(0,0,0)], dropseed = true)
    num_rows = ceil(Int, length(num_measurement_tests) / 2)
    p = plot(
        num_state_tests,
        results.kf.allocating / scale,
        layout = length(num_measurement_tests) > 1 ? (num_rows, 2) : 1,
        seriescolor = colors[1],
        legend = false,
        title = length(num_measurement_tests) > 1 ? permutedims(map(x -> "$x measurements", num_measurement_tests)) : "Time update",
        xlabel = "# States",
        ylabel = ylabel,
        yscale = logscale ? :log10 : :identity
    )
    plot!(num_state_tests, results.kf.inplace / scale, layout = length(num_measurement_tests), seriescolor = colors[1], linestyle = :dash)

    plot!(num_state_tests, results.srkf.allocating / scale, layout = length(num_measurement_tests), seriescolor = colors[2])
    plot!(num_state_tests, results.srkf.inplace / scale, layout = length(num_measurement_tests), seriescolor = colors[2], linestyle = :dash)

    plot!(num_state_tests, results.ukf.allocating / scale, layout = length(num_measurement_tests), seriescolor = colors[3])
    plot!(num_state_tests, results.ukf.inplace / scale, layout = length(num_measurement_tests), seriescolor = colors[3], linestyle = :dash)

    plot!(num_state_tests, results.srukf.allocating / scale, layout = length(num_measurement_tests), seriescolor = colors[4])
    plot!(num_state_tests, results.srukf.inplace / scale, layout = length(num_measurement_tests), seriescolor = colors[4], linestyle = :dash)

    plot!(num_state_tests, results.aukf.allocating / scale, layout = length(num_measurement_tests), seriescolor = colors[5])
    plot!(num_state_tests, results.aukf.inplace / scale, layout = length(num_measurement_tests), seriescolor = colors[5], linestyle = :dash)

    plot!(num_state_tests, results.sraukf.allocating / scale, layout = length(num_measurement_tests), seriescolor = colors[6], label = "SRAUKF")
    plot!(num_state_tests, results.sraukf.inplace / scale, layout = length(num_measurement_tests), seriescolor = colors[6], linestyle = :dash, label = "SRAUKF inplace")

    # Hack for single legend
    legend_plot = plot([1], framestyle = :none, seriescolor = colors[1], label = "KF")
    plot!([1], seriescolor = colors[1], linestyle = :dash, label = "KF inplace")

    plot!([1], seriescolor = colors[2], label = "SRKF")
    plot!([1], seriescolor = colors[2], linestyle = :dash, label = "SRKF inplace")

    plot!([1], seriescolor = colors[3], label = "UKF")
    plot!([1], seriescolor = colors[3], linestyle = :dash, label = "UKF inplace")

    plot!([1], seriescolor = colors[4], label = "SRUKF")
    plot!([1], seriescolor = colors[4], linestyle = :dash, label = "SRUKF inplace")

    plot!([1], seriescolor = colors[5], label = "AUKF")
    plot!([1], seriescolor = colors[5], linestyle = :dash, label = "AUKF inplace")

    plot!([1], seriescolor = colors[6], label = "SRAUKF")
    plot!([1], seriescolor = colors[6], linestyle = :dash, label = "SRAUKF inplace")
    plot(p, legend_plot, layout = grid(2, 1, heights = [num_rows, 1.2] / (num_rows + 1.2)), size = (800, num_rows * 440))
end

num_state_tests = [1, 5, 10, 20, 30, 40, 50, 60]
num_measurement_tests = [2, 4, 8, 16, 32, 64]
tu_time = run_time_update_benchmarks(num_state_tests)
tu_allocations = run_time_update_benchmarks(num_state_tests, allocation = true)

tu_time_plot = plot_benchmarks(tu_time, num_state_tests)
tu_alloc_plot = plot_benchmarks(tu_allocations, num_state_tests, ylabel = "Allocations (kB)", scale = 10^3)

png(tu_time_plot, "tu_time")
png(tu_alloc_plot, "tu_alloc")

mu_time = run_measurement_update_benchmarks(num_state_tests, num_measurement_tests)
mu_allocations = run_measurement_update_benchmarks(num_state_tests, num_measurement_tests, allocation = true)

mu_time_plot = plot_benchmarks(mu_time, num_state_tests, num_measurement_tests = num_measurement_tests)
mu_alloc_plot = plot_benchmarks(mu_allocations, num_state_tests, num_measurement_tests = num_measurement_tests, ylabel = "Allocations (kB)", scale = 10^3)

png(mu_time_plot, "mu_time")
png(mu_alloc_plot, "mu_alloc")