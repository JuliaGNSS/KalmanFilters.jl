# Benchmarks all filters and generates the plots of the README. Run it with an
# environment that provides KalmanFilters, BenchmarkTools, ForwardDiff and CairoMakie,
# e.g.
#   julia --project=<env> benchmark/run_benchmarks.jl
using BenchmarkTools, LinearAlgebra, KalmanFilters, CairoMakie
using ForwardDiff # loads the DifferentiationInterface backend used by the EKF
# The in-place EKF needs ForwardDiff's vector mode to not allocate, see the README.
using KalmanFilters: AutoForwardDiff

# The plots show the minimum time, which settles within far fewer samples than the
# default 5 s budget per benchmark collects (see also `benchmarks.jl`). With the default,
# the several hundred benchmarks take about an hour.
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.25

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
    h!(y, x, noise) = (mul!(y, H, x); y .+= noise)
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
    f!(y, x, noise) = (mul!(y, F, x); y .+= noise)
    return x, P, Q, F, P_chol, Q_chol, f, f!
end

function run_measurement_update_benchmarks(
    num_state_tests,
    num_measurement_tests;
    allocation = false,
)
    num_measurements = length(num_measurement_tests)
    kf_types = (:kf, :srkf, :ekf, :srekf, :ukf, :srukf, :aukf, :sraukf)
    buffers = [
        (
            inplace = zeros(length(num_state_tests), num_measurements),
            allocating = zeros(length(num_state_tests), num_measurements),
        ) for _ = 1:length(kf_types)
    ]
    results = NamedTuple{kf_types}(buffers)

    for (i, num_states) in enumerate(num_state_tests)
        for (j, num_measures) in enumerate(num_measurement_tests)
            x, y, P, H, R, P_chol, R_chol, h, h! = init_mu(num_states, num_measures)

            results.kf.allocating[i, j] = if allocation
                @allocated measurement_update(x, P, y, H, R)
            else
                @belapsed measurement_update($x, $P, $y, $H, $R)
            end
            kf_inter = KFMUIntermediate(num_states, num_measures)
            results.kf.inplace[i, j] = if allocation
                @allocated measurement_update!(kf_inter, x, P, y, H, R)
            else
                @belapsed measurement_update!($kf_inter, $x, $P, $y, $H, $R)
            end

            results.srkf.allocating[i, j] = if allocation
                @allocated measurement_update(x, P_chol, y, H, R_chol)
            else
                @belapsed measurement_update($x, $P_chol, $y, $H, $R_chol)
            end
            srkf_inter = SRKFMUIntermediate(num_states, num_measures)
            results.srkf.inplace[i, j] = if allocation
                @allocated measurement_update!(srkf_inter, x, P_chol, y, H, R_chol)
            else
                @belapsed measurement_update!($srkf_inter, $x, $P_chol, $y, $H, $R_chol)
            end

            ekf_h = JacobianPreparation(h, zero(x))
            results.ekf.allocating[i, j] = if allocation
                @allocated measurement_update(x, P, y, ekf_h, R)
            else
                @belapsed measurement_update($x, $P, $y, $ekf_h, $R)
            end
            ekf_inter = EKFMUIntermediate(num_states, num_measures)
            ekf_h! = JacobianPreparation(
                h!,
                zero(y),
                zero(x);
                backend = AutoForwardDiff(; chunksize = num_states),
            )
            results.ekf.inplace[i, j] = if allocation
                @allocated measurement_update!(ekf_inter, x, P, y, ekf_h!, R)
            else
                @belapsed measurement_update!($ekf_inter, $x, $P, $y, $ekf_h!, $R)
            end

            results.srekf.allocating[i, j] = if allocation
                @allocated measurement_update(x, P_chol, y, ekf_h, R_chol)
            else
                @belapsed measurement_update($x, $P_chol, $y, $ekf_h, $R_chol)
            end
            srekf_inter = SREKFMUIntermediate(num_states, num_measures)
            results.srekf.inplace[i, j] = if allocation
                @allocated measurement_update!(srekf_inter, x, P_chol, y, ekf_h!, R_chol)
            else
                @belapsed measurement_update!(
                    $srekf_inter,
                    $x,
                    $P_chol,
                    $y,
                    $ekf_h!,
                    $R_chol,
                )
            end

            results.ukf.allocating[i, j] = if allocation
                @allocated measurement_update(x, P, y, h, R)
            else
                @belapsed measurement_update($x, $P, $y, $h, $R)
            end
            ukf_inter = UKFMUIntermediate(num_states, num_measures)
            results.ukf.inplace[i, j] = if allocation
                @allocated measurement_update!(ukf_inter, x, P, y, h!, R)
            else
                @belapsed measurement_update!($ukf_inter, $x, $P, $y, $h!, $R)
            end

            results.srukf.allocating[i, j] = if allocation
                @allocated measurement_update(x, P_chol, y, h, R_chol)
            else
                @belapsed measurement_update($x, $P_chol, $y, $h, $R_chol)
            end
            srukf_inter = SRUKFMUIntermediate(num_states, num_measures)
            results.srukf.inplace[i, j] = if allocation
                @allocated measurement_update!(srukf_inter, x, P_chol, y, h!, R_chol)
            else
                @belapsed measurement_update!($srukf_inter, $x, $P_chol, $y, $h!, $R_chol)
            end

            results.aukf.allocating[i, j] = if allocation
                @allocated measurement_update(x, P, y, h, Augment(R))
            else
                @belapsed measurement_update($x, $P, $y, $h, $(Augment(R)))
            end
            aukf_inter = AUKFMUIntermediate(num_states, num_measures)
            results.aukf.inplace[i, j] = if allocation
                @allocated measurement_update!(aukf_inter, x, P, y, h!, Augment(R))
            else
                @belapsed measurement_update!($aukf_inter, $x, $P, $y, $h!, $(Augment(R)))
            end

            results.sraukf.allocating[i, j] = if allocation
                @allocated measurement_update(x, P_chol, y, h, Augment(R_chol))
            else
                @belapsed measurement_update($x, $P_chol, $y, $h, $(Augment(R_chol)))
            end
            sraukf_inter = SRAUKFMUIntermediate(num_states, num_measures)
            results.sraukf.inplace[i, j] = if allocation
                @allocated measurement_update!(sraukf_inter, x, P_chol, y, h!, Augment(R_chol))
            else
                @belapsed measurement_update!(
                    $sraukf_inter,
                    $x,
                    $P_chol,
                    $y,
                    $h!,
                    $(Augment(R_chol)),
                )
            end
        end
    end
    results
end

function run_time_update_benchmarks(num_state_tests; allocation = false)
    kf_types = (:kf, :srkf, :ekf, :srekf, :ukf, :srukf, :aukf, :sraukf)
    buffers = [
        (
            inplace = zeros(length(num_state_tests)),
            allocating = zeros(length(num_state_tests)),
        ) for _ = 1:length(kf_types)
    ]
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

        ekf_f = JacobianPreparation(f, zero(x))
        results.ekf.allocating[i] = if allocation
            @allocated time_update(x, P, ekf_f, Q)
        else
            @belapsed time_update($x, $P, $ekf_f, $Q)
        end
        ekf_inter = EKFTUIntermediate(num_states)
        ekf_f! = JacobianPreparation(
            f!,
            zero(x),
            zero(x);
            backend = AutoForwardDiff(; chunksize = num_states),
        )
        results.ekf.inplace[i] = if allocation
            @allocated time_update!(ekf_inter, x, P, ekf_f!, Q)
        else
            @belapsed time_update!($ekf_inter, $x, $P, $ekf_f!, $Q)
        end

        results.srekf.allocating[i] = if allocation
            @allocated time_update(x, P_chol, ekf_f, Q_chol)
        else
            @belapsed time_update($x, $P_chol, $ekf_f, $Q_chol)
        end
        srekf_inter = SREKFTUIntermediate(num_states)
        results.srekf.inplace[i] = if allocation
            @allocated time_update!(srekf_inter, x, P_chol, ekf_f!, Q_chol)
        else
            @belapsed time_update!($srekf_inter, $x, $P_chol, $ekf_f!, $Q_chol)
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

# Each filter and its square root variant share a color and are distinguished by the
# marker (circles for the standard, triangles for the square root variant), the
# allocating and in-place updates by the line style.
const FILTER_FAMILIES = (
    (name = "KF", standard = :kf, square_root = :srkf),
    (name = "EKF", standard = :ekf, square_root = :srekf),
    (name = "UKF", standard = :ukf, square_root = :srukf),
    (name = "AUKF", standard = :aukf, square_root = :sraukf),
)
const FAMILY_COLORS = Makie.wong_colors()[1:4]
const VARIANT_MARKERS = (standard = (:circle, 7), square_root = (:utriangle, 11))
const UPDATE_LINESTYLES = (allocating = :solid, inplace = :dash)

# The README is at most about 830 pixels wide on GitHub. The figures are laid out for
# that width and saved with twice the resolution, so that they stay sharp.
const FIGURE_WIDTH = 800
const PX_PER_UNIT = 2

function plot_series!(ax, num_state_tests, results, column; scale)
    for (family, color) in zip(FILTER_FAMILIES, FAMILY_COLORS),
        (update, linestyle) in pairs(UPDATE_LINESTYLES),
        (variant, (marker, markersize)) in pairs(VARIANT_MARKERS)

        filter_type = getproperty(family, variant)
        isnothing(filter_type) && continue
        values = getproperty(results[filter_type], update)[:, column] ./ scale
        scatterlines!(
            ax,
            num_state_tests,
            values;
            color,
            linestyle,
            linewidth = 2,
            marker,
            markersize,
        )
    end
end

# Ticks at 1, 2 and 5 times the powers of ten, or only at the powers of ten if the
# values span several of them, labeled as plain numbers instead of as `10^x`.
function log_ticks(values)
    finite_values = filter(v -> isfinite(v) && v > 0, values)
    low, high =
        floor(Int, log10(minimum(finite_values))), ceil(Int, log10(maximum(finite_values)))
    mantissas = high - low > 3 ? (1,) : (1, 2, 5)
    ticks = [m * 10.0^e for e = low:high for m in mantissas]
    ticks, map(tick -> string(tick >= 1 ? round(Int, tick) : tick), ticks)
end

function add_legend!(position)
    gray = RGBf(0.35, 0.35, 0.35)
    Legend(
        position,
        [
            [LineElement(; color, linewidth = 3) for color in FAMILY_COLORS],
            [
                MarkerElement(; color = gray, marker, markersize) for
                (marker, markersize) in values(VARIANT_MARKERS)
            ],
            [
                LineElement(; color = gray, linestyle, linewidth = 2) for
                linestyle in values(UPDATE_LINESTYLES)
            ],
        ],
        [
            [family.name for family in FILTER_FAMILIES],
            ["Standard", "Square root"],
            ["Allocating", "In-place"],
        ],
        ["Filter", "Variant", "Update"];
        orientation = :horizontal,
        titleposition = :top,
        framevisible = false,
        tellheight = true,
        tellwidth = false,
    )
end

panel_values(results, column) = reduce(
    vcat,
    [
        getproperty(r, update)[:, column] for r in results for
        update in keys(UPDATE_LINESTYLES)
    ],
)

function plot_benchmarks(
    results,
    num_state_tests;
    num_measurement_tests = nothing,
    title = "Time update",
    ylabel = "Time (μs)",
    scale = 10^-6,
    logscale = false,
)
    columns = isnothing(num_measurement_tests) ? (1:1) : eachindex(num_measurement_tests)
    num_cols = length(columns) > 1 ? 2 : 1
    num_rows = cld(length(columns), num_cols)
    panel_height = num_cols > 1 ? 260 : 340
    fig = Figure(; size = (FIGURE_WIDTH, num_rows * panel_height + 110), fontsize = 14)
    axes = map(enumerate(columns)) do (i, column)
        row, col = fldmod1(i, num_cols)
        Axis(
            fig[row, col];
            title = isnothing(num_measurement_tests) ? title :
                    "$(num_measurement_tests[column]) measurements",
            xlabel = row == num_rows ? "Number of states" : "",
            ylabel = col == 1 ? ylabel : "",
            yscale = logscale ? log10 : identity,
            yticks = logscale ? log_ticks(panel_values(results, column) ./ scale) :
                     Makie.automatic,
            xticks = 0:10:maximum(num_state_tests),
        )
    end
    for (ax, column) in zip(axes, columns)
        plot_series!(ax, num_state_tests, results, column; scale)
    end
    linkxaxes!(axes...)
    add_legend!(fig[num_rows+1, 1:num_cols])
    fig
end

save_plot(name, fig) = save(joinpath(@__DIR__, "$name.png"), fig; px_per_unit = PX_PER_UNIT)

num_state_tests = [1, 5, 10, 20, 30, 40, 50, 60]
num_measurement_tests = [2, 4, 8, 16, 32, 64]
tu_time = run_time_update_benchmarks(num_state_tests)
tu_allocations = run_time_update_benchmarks(num_state_tests; allocation = true)

save_plot("tu_time", plot_benchmarks(tu_time, num_state_tests; logscale = true))
save_plot(
    "tu_alloc",
    plot_benchmarks(
        tu_allocations,
        num_state_tests;
        title = "Time update allocations",
        ylabel = "Allocations (kB)",
        scale = 10^3,
    ),
)

mu_time = run_measurement_update_benchmarks(num_state_tests, num_measurement_tests)
mu_allocations = run_measurement_update_benchmarks(
    num_state_tests,
    num_measurement_tests;
    allocation = true,
)

save_plot(
    "mu_time",
    plot_benchmarks(mu_time, num_state_tests; num_measurement_tests, logscale = true),
)
save_plot(
    "mu_alloc",
    plot_benchmarks(
        mu_allocations,
        num_state_tests;
        num_measurement_tests,
        ylabel = "Allocations (kB)",
        scale = 10^3,
    ),
)
