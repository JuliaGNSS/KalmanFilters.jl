# Entry point of the `juliac --trim=safe` check (see `check.jl`): one time update and
# one measurement update through every filter, in-place and allocating, printed so
# the trimmed executable's output can be compared against a regular Julia session.
# The in-place updates write into their `x` and `P`, so each gets its own copy of
# the prior and its measurement update continues from what it wrote.
using KalmanFilters, LinearAlgebra
using LazyArrays: @~
# The EKF's default `AutoForwardDiff` backend is a DifferentiationInterface extension
# that only loads with ForwardDiff itself.
using ForwardDiff
using DifferentiationInterface: AutoForwardDiff

# Callable structs rather than closures: the trimmed build needs every call site
# resolvable, and a closure over a non-constant global is not.
struct LinearModel
    F::Matrix{Float64}
end
(m::LinearModel)(x) = m.F * x
(m::LinearModel)(x, noise) = m.F * x .+ noise

# The in-place counterpart, a separate type because `f!(y, x)` and the augmented
# `f(x, noise)` share an arity.
struct InPlaceLinearModel
    F::Matrix{Float64}
end
(m::InPlaceLinearModel)(y, x) = mul!(y, m.F, x)
(m::InPlaceLinearModel)(y, x, noise) = y .= @~ m.F * x .+ noise

deterministic_matrix(rows, cols, seed) =
    [sin(seed + 1.3 * i + 0.7 * j) for i in 1:rows, j in 1:cols]
function positive_definite(n, seed)
    A = deterministic_matrix(n, n, seed)
    A' * A + I
end

# One `print` per value: a long `print(io, xs...)` is not specialised on its
# arguments' types, which leaves the call dynamic.
function report(io, name, update)
    print(io, name, ":")
    for v in (get_state(update)[1], get_state(update)[end],
        get_covariance(update)[1, 1], get_covariance(update)[end, end])
        print(io, " ", v)
    end
    println(io)
end

function (@main)(args::Vector{String})::Cint
    io = Core.stdout
    n, m = 4, 3
    x = [0.1 * i for i in 1:n]
    P = positive_definite(n, 1)
    Q = 0.1 * positive_definite(n, 2)
    R = 0.2 * positive_definite(m, 3)
    y = [cos(i) for i in 1:m]
    F = I + 0.1 * deterministic_matrix(n, n, 4)
    H = deterministic_matrix(m, n, 5)
    f = LinearModel(F)
    h = LinearModel(H)
    f! = InPlaceLinearModel(F)
    h! = InPlaceLinearModel(H)
    P_chol = cholesky(Hermitian(P))
    Q_chol = cholesky(Hermitian(Q))
    R_chol = cholesky(Hermitian(R))

    # Kalman filter
    tu = time_update(x, P, F, Q)
    report(io, "KF time update", tu)
    report(io, "KF measurement update", measurement_update(get_state(tu), get_covariance(tu), y, H, R))
    tu! = time_update!(KFTUIntermediate(Float64, n), copy(x), copy(P), F, Q)
    report(io, "KF! time update", tu!)
    report(io, "KF! measurement update",
        measurement_update!(KFMUIntermediate(Float64, n, m), get_state(tu!), get_covariance(tu!), y, H, R))
    report(io, "KF scalar measurement update", measurement_update(x, P, y[1], H[1, :]', R[1, 1]))

    # Square-root Kalman filter
    tu = time_update(x, P_chol, F, Q_chol)
    report(io, "SRKF time update", tu)
    report(io, "SRKF measurement update", measurement_update(get_state(tu), tu.covariance, y, H, R_chol))
    tu! = time_update!(SRKFTUIntermediate(Float64, n), copy(x), copy(P_chol), F, Q_chol)
    report(io, "SRKF! time update", tu!)
    report(io, "SRKF! measurement update",
        measurement_update!(SRKFMUIntermediate(Float64, n, m), get_state(tu!), tu!.covariance, y, H, R_chol))

    # Unscented Kalman filter
    tu = time_update(x, P, f, Q)
    report(io, "UKF time update", tu)
    report(io, "UKF measurement update", measurement_update(get_state(tu), get_covariance(tu), y, h, R))
    tu! = time_update!(UKFTUIntermediate(Float64, n), copy(x), copy(P), f!, Q)
    report(io, "UKF! time update", tu!)
    report(io, "UKF! measurement update",
        measurement_update!(UKFMUIntermediate(Float64, n, m), get_state(tu!), get_covariance(tu!), y, h!, R))

    # Square-root unscented Kalman filter
    tu = time_update(x, P_chol, f, Q_chol)
    report(io, "SRUKF time update", tu)
    report(io, "SRUKF measurement update", measurement_update(get_state(tu), tu.covariance, y, h, R_chol))
    tu! = time_update!(SRUKFTUIntermediate(Float64, n), copy(x), copy(P_chol), f!, Q_chol)
    report(io, "SRUKF! time update", tu!)
    report(io, "SRUKF! measurement update",
        measurement_update!(SRUKFMUIntermediate(Float64, n, m), get_state(tu!), tu!.covariance, y, h!, R_chol))

    # Augmented unscented Kalman filter
    tu = time_update(x, P, f, Augment(Q))
    report(io, "AUKF time update", tu)
    report(io, "AUKF measurement update", measurement_update(get_state(tu), get_covariance(tu), y, h, Augment(R)))
    tu! = time_update!(AUKFTUIntermediate(Float64, n), copy(x), copy(P), f!, Augment(Q))
    report(io, "AUKF! time update", tu!)
    report(io, "AUKF! measurement update",
        measurement_update!(AUKFMUIntermediate(Float64, n, m), get_state(tu!), get_covariance(tu!), y, h!, Augment(R)))

    # Square-root augmented unscented Kalman filter
    tu = time_update(x, P_chol, f, Augment(Q_chol))
    report(io, "SRAUKF time update", tu)
    report(io, "SRAUKF measurement update",
        measurement_update(get_state(tu), tu.covariance, y, h, Augment(R_chol)))
    tu! = time_update!(
        SRAUKFTUIntermediate(Float64, n), copy(x), copy(P_chol), f!, Augment(Q_chol))
    report(io, "SRAUKF! time update", tu!)
    report(io, "SRAUKF! measurement update",
        measurement_update!(SRAUKFMUIntermediate(Float64, n, m), get_state(tu!), tu!.covariance, y, h!, Augment(R_chol)))

    # Extended Kalman filter. The chunk size is fixed: the default `AutoForwardDiff()`
    # picks it from `length(x)` at run time, and the chunk is a type parameter, so the
    # Jacobian's configuration would not be inferable.
    backend = AutoForwardDiff(; chunksize = 4)
    F_jacobian = JacobianPreparation(LinearModel(F), zeros(n); backend)
    H_jacobian = JacobianPreparation(LinearModel(H), zeros(n); backend)
    tu = time_update(x, P, F_jacobian, Q)
    report(io, "EKF time update", tu)
    report(io, "EKF measurement update",
        measurement_update(get_state(tu), get_covariance(tu), y, H_jacobian, R))
    F_jacobian! = JacobianPreparation(f!, zeros(n), zeros(n); backend)
    H_jacobian! = JacobianPreparation(h!, zeros(m), zeros(n); backend)
    tu! = time_update!(EKFTUIntermediate(Float64, n), copy(x), copy(P), F_jacobian!, Q)
    report(io, "EKF! time update", tu!)
    report(io, "EKF! measurement update",
        measurement_update!(EKFMUIntermediate(Float64, n, m), get_state(tu!), get_covariance(tu!), y, H_jacobian!, R))

    # Square-root extended Kalman filter
    tu = time_update(x, P_chol, F_jacobian, Q_chol)
    report(io, "SREKF time update", tu)
    report(io, "SREKF measurement update",
        measurement_update(get_state(tu), tu.covariance, y, H_jacobian, R_chol))
    tu! = time_update!(SREKFTUIntermediate(Float64, n), copy(x), copy(P_chol), F_jacobian!, Q_chol)
    report(io, "SREKF! time update", tu!)
    report(io, "SREKF! measurement update",
        measurement_update!(SREKFMUIntermediate(Float64, n, m), get_state(tu!), tu!.covariance, y, H_jacobian!, R_chol))
    return 0
end
