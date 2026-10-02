struct GradientOrJacobianPreparation{P<:DifferentiationInterface.Prep,F,B,C}
    f::F
    preparation::P
    backend::B
    contexts::C
end

"""
$(SIGNATURES)

JacobianPreparation calculates the Jacobian matrix automatically.
Simply pass the function that you'd like to calculate the Jacobian matrix
for and it will do so automatically. You can use different kinds of
automatic differentiator. By default it will use AutoForwardDiff.
You must also pass the state vector x in order to optimize the process.
It doesn't need to hold actual values. You can pass e.g. `zeros(num_x)`.
The type must match with the type of your state vector. With contexts,
parameters can be provided that will be passed alongside the state vector
to the function f.

For a `Vector` state the default `AutoForwardDiff()` picks its chunk size from
`length(x)` at run time, which leaves the prepared Jacobian's type uninferable.
Pass `backend = AutoForwardDiff(; chunksize = length(x))` with a constant chunk
size where that matters, e.g. for a `juliac --trim` build.
"""
function JacobianPreparation(
    f,
    x::AbstractVector,
    contexts::Vararg{DifferentiationInterface.Context,C};
    backend = AutoForwardDiff(),
) where {C}
    GradientOrJacobianPreparation(
        f,
        prepare_jacobian(f, backend, x, contexts...),
        backend,
        contexts,
    )
end

"""
$(SIGNATURES)

GradientPreparation calculates the gradient automatically. In contrast to 
JacobianPreparation the function `f` needs to be a scalar instead of a vector.
See JacobianPreparation for more information.
"""
function GradientPreparation(
    f,
    x::AbstractVector,
    contexts::Vararg{DifferentiationInterface.Context,C};
    backend = AutoForwardDiff(),
) where {C}
    GradientOrJacobianPreparation(
        f,
        prepare_gradient(f, backend, x, contexts...),
        backend,
        contexts,
    )
end

"""
$(SIGNATURES)

GradientOrJacobianContextUpdate allows to change the context parameters.
The type of contexts must match with the context parameter provided for
JacobianPreparation.
"""
function GradientOrJacobianContextUpdate(
    F::GradientOrJacobianPreparation,
    contexts::Vararg{DifferentiationInterface.Context,C},
) where {C}
    GradientOrJacobianPreparation(F.f, F.preparation, F.backend, contexts)
end

"""
$(SIGNATURES)

Extended Kalman Filter time update.
F is the GradientOrJacobianPreparation object.
"""
function time_update(x, P, F::GradientOrJacobianPreparation, Q)
    F.preparation isa DifferentiationInterface.GradientPrep &&
        error("Gradient is currently not supported for the time update.")
    x_apri, jacobian = value_and_jacobian(F.f, F.preparation, F.backend, x, F.contexts...)
    P_apri = calc_apriori_covariance(P, jacobian, Q)
    KFTimeUpdate(x_apri, P_apri)
end

function value_and_gradient_or_jacobian(
    F::GradientOrJacobianPreparation{<:DifferentiationInterface.JacobianPrep},
    x,
)
    value_and_jacobian(F.f, F.preparation, F.backend, x, F.contexts...)
end

function value_and_gradient_or_jacobian(
    F::GradientOrJacobianPreparation{<:DifferentiationInterface.GradientPrep},
    x,
)
    value, gradient = value_and_gradient(F.f, F.preparation, F.backend, x, F.contexts...)
    return value, transpose(gradient)
end

"""
$(SIGNATURES)

Extended Kalman Filter measurement update.
H is the GradientOrJacobianPreparation object.
"""
function measurement_update(
    x,
    P,
    y,
    H::GradientOrJacobianPreparation,
    R;
    consider = nothing,
)
    y_pre, gradient_or_jacobian = value_and_gradient_or_jacobian(H, x)
    ỹ = calc_innovation(y_pre, y)
    PHᵀ = calc_P_xy(P, gradient_or_jacobian)
    S = calc_innovation_covariance(gradient_or_jacobian, P, R)
    K = calc_kalman_gain(PHᵀ, S, consider)
    x_post = calc_posterior_state(x, K, ỹ, consider)
    P_post = calc_posterior_covariance(P, PHᵀ, K, consider)
    KFMeasurementUpdate(x_post, P_post, ỹ, S, K)
end

calc_innovation(y_pre, y) = y - y_pre
# ── In-place updates ─────────────────────────────────────────────────────────

# A Jacobian prepared for an in-place model `f!(y, x, contexts...)`, as the in-place
# updates take it.
struct InPlaceJacobianPreparation{P<:DifferentiationInterface.JacobianPrep,F,B,C}
    f!::F
    preparation::P
    backend::B
    contexts::C
end

"""
$(SIGNATURES)

JacobianPreparation for an in-place function `f!(y, x, contexts...)`, which writes its
value into `y`, as the in-place `time_update!` and `measurement_update!` of the Extended
Kalman Filter take it. Like `x`, `y` doesn't need to hold actual values, but its type
and length must match the function's output, e.g. `zeros(num_y)`.

ForwardDiff only computes the Jacobian without allocating in its vector mode, i.e. if
the chunk size is the length of `x`. The default `AutoForwardDiff()` picks that for up
to 12 states; for more, pass `backend = AutoForwardDiff(; chunksize = length(x))`.
"""
function JacobianPreparation(
    f!,
    y::AbstractVector,
    x::AbstractVector,
    contexts::Vararg{DifferentiationInterface.Context,C};
    backend = AutoForwardDiff(),
) where {C}
    InPlaceJacobianPreparation(
        f!,
        prepare_jacobian(f!, y, backend, x, contexts...),
        backend,
        contexts,
    )
end

function GradientOrJacobianContextUpdate(
    F::InPlaceJacobianPreparation,
    contexts::Vararg{DifferentiationInterface.Context,C},
) where {C}
    InPlaceJacobianPreparation(F.f!, F.preparation, F.backend, contexts)
end

# Writes the value of `F.f!` at `x` into `y` and its Jacobian into `J`. DI's
# `value_and_jacobian!` would allocate a `DiffResult` for ForwardDiff; this is its
# generic fallback, which evaluates the model once more for the value.
function value_and_jacobian_inplace!(y, J, F::InPlaceJacobianPreparation, x)
    jacobian!(F.f!, y, J, F.preparation, F.backend, x, F.contexts...)
    F.f!(y, x, map(DifferentiationInterface.unwrap, F.contexts)...)
    return y, J
end

struct EKFTUIntermediate{T}
    x_apri::Vector{T}
    jacobian::Matrix{T}
    fp::Matrix{T}
end

EKFTUIntermediate(::Type{T}, num_x::Number) where {T} = EKFTUIntermediate(
    Vector{T}(undef, num_x),
    Matrix{T}(undef, num_x, num_x),
    Matrix{T}(undef, num_x, num_x),
)

EKFTUIntermediate(num_x::Number) = EKFTUIntermediate(Float64, num_x)

struct EKFMUIntermediate{T}
    innovation::Vector{T}
    jacobian::Matrix{T}
    innovation_covariance::Matrix{T}
    kalman_gain::Matrix{T}
    pht::Matrix{T}
    s_chol::Matrix{T}
end

EKFMUIntermediate(::Type{T}, num_x::Number, num_y::Number) where {T} = EKFMUIntermediate(
    Vector{T}(undef, num_y),
    Matrix{T}(undef, num_y, num_x),
    Matrix{T}(undef, num_y, num_y),
    Matrix{T}(undef, num_x, num_y),
    Matrix{T}(undef, num_x, num_y),
    Matrix{T}(undef, num_y, num_y),
)

EKFMUIntermediate(num_x::Number, num_y::Number) = EKFMUIntermediate(Float64, num_x, num_y)

"""
$(SIGNATURES)

Extended Kalman Filter time update in place: `x` and `P` are overwritten with the a
priori state and covariance. `F` is the JacobianPreparation of an in-place model
`f!(y, x)`.
"""
function time_update!(tu::EKFTUIntermediate, x, P, F::InPlaceJacobianPreparation, Q)
    value_and_jacobian_inplace!(tu.x_apri, tu.jacobian, F, x)
    copyto!(x, tu.x_apri)
    calc_apriori_covariance!(P, tu.fp, tu.jacobian, Q)
    KFTimeUpdate(x, P)
end

"""
$(SIGNATURES)

Extended Kalman Filter measurement update in place: `x` and `P` are overwritten with
the posterior state and covariance. `H` is the JacobianPreparation of an in-place model
`h!(y, x)`.
"""
function measurement_update!(
    mu::EKFMUIntermediate,
    x,
    P,
    y,
    H::InPlaceJacobianPreparation,
    R,
)
    y_pre, jacobian = value_and_jacobian_inplace!(mu.innovation, mu.jacobian, H, x)
    ỹ = y_pre .= y .- y_pre
    PHᵀ = calc_P_xy!(mu.pht, P, jacobian)
    S = calc_innovation_covariance!(mu.innovation_covariance, jacobian, PHᵀ, R)
    K = calc_kalman_gain!(mu.s_chol, mu.kalman_gain, PHᵀ, S)
    calc_posterior_state!(x, K, ỹ)
    calc_posterior_covariance!(P, PHᵀ, K)
    KFMeasurementUpdate(x, P, ỹ, S, K)
end
