# Benchmark suite for AirspeedVelocity (`benchpkg`), run on every PR by
# `.github/workflows/benchmark_pr.yml`. AirspeedVelocity runs the PR head's copy of this
# file against both the base and the head revision, so it must only use API that exists
# on both.
#
# Run locally with:
#   julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
#   julia --project=benchmark -e 'include("benchmark/benchmarks.jl"); run(SUITE; verbose = true)'
# or compare revisions with AirspeedVelocity, e.g.
#   benchpkg KalmanFilters --path=. --rev=master,dirty --add=ForwardDiff,StaticArrays
using BenchmarkTools
using ForwardDiff # loads the DifferentiationInterface backend used by the EKF
# The in-place EKF needs ForwardDiff's vector mode to not allocate, see the README.
using KalmanFilters: AutoForwardDiff
using KalmanFilters
using LinearAlgebra
using Random
using StaticArrays

const SUITE = BenchmarkGroup()

# Keep the suite affordable on CI: every benchmark is tuned and run for both the base and
# head rev. The PR comment reports the minimum time, which settles within far fewer samples
# than the default 5 s budget collects; 0.25 s is as stable as 1 s run to run, and still
# gives the slowest benchmarks (~0.3 ms) hundreds of samples.
const SECONDS = 0.25

# (number of states, number of measurements)
const SIZES = ((2, 2), (10, 4), (50, 16))

# The in-place updates write into `x` and `P`, so every sample of theirs starts from a
# fresh copy of the prior, made in the untimed `setup`, and runs once (`evals = 1`).

random_pos_def(n) = (A = randn(n, n); A'A + n * I)
vector_mode(num_states) = AutoForwardDiff(; chunksize = num_states)

size_label(num_states) = "$num_states states"
size_label(num_states, num_measures) = "$num_states states, $num_measures measurements"

function init_tu(num_states)
    x = randn(num_states)
    P = random_pos_def(num_states)
    Q = random_pos_def(num_states)
    F = randn(num_states, num_states)
    f(x) = F * x
    f!(y, x) = mul!(y, F, x)
    # Augmented variants pass the noise sigma points as an extra argument
    f(x, noise) = F * x .+ noise
    f!(y, x, noise) = (mul!(y, F, x); y .+= noise)
    return x, P, Q, F, f, f!
end

function init_mu(num_states, num_measures)
    x = randn(num_states)
    P = random_pos_def(num_states)
    R = random_pos_def(num_measures)
    y = randn(num_measures)
    H = randn(num_measures, num_states)
    h(x) = H * x
    h!(y, x) = mul!(y, H, x)
    h(x, noise) = H * x .+ noise
    h!(y, x, noise) = (mul!(y, H, x); y .+= noise)
    return x, P, R, y, H, h, h!
end

Random.seed!(1234)

tu = SUITE["time update"] = BenchmarkGroup()
for (num_states, _) in SIZES
    x, P, Q, F, f, f! = init_tu(num_states)
    P_chol = cholesky(P)
    Q_chol = cholesky(Q)
    label = size_label(num_states)

    for name in ("KF", "SRKF", "UKF", "SRUKF", "AUKF", "SRAUKF", "EKF")
        haskey(tu, name) || (tu[name] = BenchmarkGroup())
        tu[name][label] = BenchmarkGroup()
    end

    tu["KF"][label]["allocating"] =
        @benchmarkable time_update($x, $P, $F, $Q) seconds = SECONDS
    tu["KF"][label]["inplace"] = @benchmarkable time_update!(
        $(KFTUIntermediate(num_states)),
        x,
        P,
        $F,
        $Q,
    ) setup = (x = copy($x); P = copy($P)) evals = 1 seconds = SECONDS

    tu["SRKF"][label]["allocating"] =
        @benchmarkable time_update($x, $P_chol, $F, $Q_chol) seconds = SECONDS
    tu["SRKF"][label]["inplace"] = @benchmarkable time_update!(
        $(SRKFTUIntermediate(num_states)),
        x,
        P_chol,
        $F,
        $Q_chol,
    ) setup = (x = copy($x); P_chol = copy($P_chol)) evals = 1 seconds = SECONDS

    tu["UKF"][label]["allocating"] =
        @benchmarkable time_update($x, $P, $f, $Q) seconds = SECONDS
    tu["UKF"][label]["inplace"] = @benchmarkable time_update!(
        $(UKFTUIntermediate(num_states)),
        x,
        P,
        $f!,
        $Q,
    ) setup = (x = copy($x); P = copy($P)) evals = 1 seconds = SECONDS

    tu["SRUKF"][label]["allocating"] =
        @benchmarkable time_update($x, $P_chol, $f, $Q_chol) seconds = SECONDS
    tu["SRUKF"][label]["inplace"] = @benchmarkable time_update!(
        $(SRUKFTUIntermediate(num_states)),
        x,
        P_chol,
        $f!,
        $Q_chol,
    ) setup = (x = copy($x); P_chol = copy($P_chol)) evals = 1 seconds = SECONDS

    tu["AUKF"][label]["allocating"] =
        @benchmarkable time_update($x, $P, $f, $(Augment(Q))) seconds = SECONDS
    tu["AUKF"][label]["inplace"] = @benchmarkable time_update!(
        $(AUKFTUIntermediate(num_states)),
        x,
        P,
        $f!,
        $(Augment(Q)),
    ) setup = (x = copy($x); P = copy($P)) evals = 1 seconds = SECONDS

    tu["SRAUKF"][label]["allocating"] =
        @benchmarkable time_update($x, $P_chol, $f, $(Augment(Q_chol))) seconds = SECONDS
    tu["SRAUKF"][label]["inplace"] = @benchmarkable time_update!(
        $(SRAUKFTUIntermediate(num_states)),
        x,
        P_chol,
        $f!,
        $(Augment(Q_chol)),
    ) setup = (x = copy($x); P_chol = copy($P_chol)) evals = 1 seconds = SECONDS

    tu["EKF"][label]["allocating"] = @benchmarkable time_update(
        $x,
        $P,
        $(JacobianPreparation(f, zero(x))),
        $Q,
    ) seconds = SECONDS
    # The in-place EKF is newer than some base revisions this suite runs against.
    if isdefined(KalmanFilters, :EKFTUIntermediate)
        tu["EKF"][label]["inplace"] = @benchmarkable time_update!(
            $(EKFTUIntermediate(num_states)),
            x,
            P,
            $(JacobianPreparation(f!, zero(x), zero(x); backend = vector_mode(num_states))),
            $Q,
        ) setup = (x = copy($x); P = copy($P)) evals = 1 seconds = SECONDS
    end
end

mu = SUITE["measurement update"] = BenchmarkGroup()
for (num_states, num_measures) in SIZES
    x, P, R, y, H, h, h! = init_mu(num_states, num_measures)
    P_chol = cholesky(P)
    R_chol = cholesky(R)
    label = size_label(num_states, num_measures)

    for name in ("KF", "SRKF", "UKF", "SRUKF", "AUKF", "SRAUKF", "EKF")
        haskey(mu, name) || (mu[name] = BenchmarkGroup())
        mu[name][label] = BenchmarkGroup()
    end

    mu["KF"][label]["allocating"] =
        @benchmarkable measurement_update($x, $P, $y, $H, $R) seconds = SECONDS
    mu["KF"][label]["inplace"] = @benchmarkable measurement_update!(
        $(KFMUIntermediate(num_states, num_measures)),
        x,
        P,
        $y,
        $H,
        $R,
    ) setup = (x = copy($x); P = copy($P)) evals = 1 seconds = SECONDS

    mu["SRKF"][label]["allocating"] =
        @benchmarkable measurement_update($x, $P_chol, $y, $H, $R_chol) seconds = SECONDS
    mu["SRKF"][label]["inplace"] = @benchmarkable measurement_update!(
        $(SRKFMUIntermediate(num_states, num_measures)),
        x,
        P_chol,
        $y,
        $H,
        $R_chol,
    ) setup = (x = copy($x); P_chol = copy($P_chol)) evals = 1 seconds = SECONDS

    mu["UKF"][label]["allocating"] =
        @benchmarkable measurement_update($x, $P, $y, $h, $R) seconds = SECONDS
    mu["UKF"][label]["inplace"] = @benchmarkable measurement_update!(
        $(UKFMUIntermediate(num_states, num_measures)),
        x,
        P,
        $y,
        $h!,
        $R,
    ) setup = (x = copy($x); P = copy($P)) evals = 1 seconds = SECONDS

    mu["SRUKF"][label]["allocating"] =
        @benchmarkable measurement_update($x, $P_chol, $y, $h, $R_chol) seconds = SECONDS
    mu["SRUKF"][label]["inplace"] = @benchmarkable measurement_update!(
        $(SRUKFMUIntermediate(num_states, num_measures)),
        x,
        P_chol,
        $y,
        $h!,
        $R_chol,
    ) setup = (x = copy($x); P_chol = copy($P_chol)) evals = 1 seconds = SECONDS

    mu["AUKF"][label]["allocating"] =
        @benchmarkable measurement_update($x, $P, $y, $h, $(Augment(R))) seconds = SECONDS
    mu["AUKF"][label]["inplace"] = @benchmarkable measurement_update!(
        $(AUKFMUIntermediate(num_states, num_measures)),
        x,
        P,
        $y,
        $h!,
        $(Augment(R)),
    ) setup = (x = copy($x); P = copy($P)) evals = 1 seconds = SECONDS

    mu["SRAUKF"][label]["allocating"] = @benchmarkable measurement_update(
        $x,
        $P_chol,
        $y,
        $h,
        $(Augment(R_chol)),
    ) seconds = SECONDS
    mu["SRAUKF"][label]["inplace"] = @benchmarkable measurement_update!(
        $(SRAUKFMUIntermediate(num_states, num_measures)),
        x,
        P_chol,
        $y,
        $h!,
        $(Augment(R_chol)),
    ) setup = (x = copy($x); P_chol = copy($P_chol)) evals = 1 seconds = SECONDS

    mu["EKF"][label]["allocating"] = @benchmarkable measurement_update(
        $x,
        $P,
        $y,
        $(JacobianPreparation(h, zero(x))),
        $R,
    ) seconds = SECONDS
    if isdefined(KalmanFilters, :EKFMUIntermediate)
        mu["EKF"][label]["inplace"] = @benchmarkable measurement_update!(
            $(EKFMUIntermediate(num_states, num_measures)),
            x,
            P,
            $y,
            $(JacobianPreparation(h!, zero(y), zero(x); backend = vector_mode(num_states))),
            $R,
        ) setup = (x = copy($x); P = copy($P)) evals = 1 seconds = SECONDS
    end
end

# A full filter loop (time update followed by measurement update) with StaticArrays, see
# also `static_arrays_benchmark.jl`.
function run_static_srkf(x, P_chol, F, Q_chol, y, H, R_chol, num_iterations)
    for _ = 1:num_iterations
        tu = time_update(x, P_chol, F, Q_chol)
        mu = measurement_update(get_state(tu), get_sqrt_covariance(tu), y, H, R_chol)
        x, P_chol = get_state(mu), get_sqrt_covariance(mu)
    end
    return x, P_chol
end

function run_static_kf(x, P, F, Q, y, H, R, num_iterations)
    for _ = 1:num_iterations
        tu = time_update(x, P, F, Q)
        mu = measurement_update(get_state(tu), get_covariance(tu), y, H, R)
        x, P = get_state(mu), get_covariance(mu)
    end
    return x, P
end

# The models are passed on, so `::F ... where {F}` makes Julia specialise on them.
function run_static_ukf(x, P, f::F, Q, y, h::H, R, num_iterations) where {F,H}
    for _ = 1:num_iterations
        tu = time_update(x, P, f, Q)
        mu = measurement_update(get_state(tu), get_covariance(tu), y, h, R)
        x, P = get_state(mu), get_covariance(mu)
    end
    return x, P
end

function run_static_srukf(x, P_chol, f::F, Q_chol, y, h::H, R_chol, num_iterations) where {F,H}
    for _ = 1:num_iterations
        tu = time_update(x, P_chol, f, Q_chol)
        mu = measurement_update(get_state(tu), get_sqrt_covariance(tu), y, h, R_chol)
        x, P_chol = get_state(mu), get_sqrt_covariance(mu)
    end
    return x, P_chol
end

static = SUITE["StaticArrays"] = BenchmarkGroup()
let Dx = 2, Dy = 2
    F = @SMatrix randn(Dx, Dx)
    Q = SMatrix{Dx,Dx}(random_pos_def(Dx))
    H = @SMatrix randn(Dy, Dx)
    R = SMatrix{Dy,Dy}(random_pos_def(Dy))
    x = @SVector randn(Dx)
    P = SMatrix{Dx,Dx}(random_pos_def(Dx))
    y = @SVector randn(Dy)
    static["KF 100 iterations"] =
        @benchmarkable run_static_kf($x, $P, $F, $Q, $y, $H, $R, 100) seconds = SECONDS
    static["SRKF 100 iterations"] = @benchmarkable run_static_srkf(
        $x,
        $(cholesky(P)),
        $F,
        $(cholesky(Q)),
        $y,
        $H,
        $(cholesky(R)),
        100,
    ) seconds = SECONDS
    f(x) = F * x
    h(x) = H * x
    static["UKF 100 iterations"] =
        @benchmarkable run_static_ukf($x, $P, $f, $Q, $y, $h, $R, 100) seconds = SECONDS
    static["SRUKF 100 iterations"] = @benchmarkable run_static_srukf(
        $x,
        $(cholesky(P)),
        $f,
        $(cholesky(Q)),
        $y,
        $h,
        $(cholesky(R)),
        100,
    ) seconds = SECONDS
end
