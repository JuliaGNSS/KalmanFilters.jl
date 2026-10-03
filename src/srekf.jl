# The Square Root Extended Kalman Filter is the Square Root Kalman Filter on the
# Jacobians of the models. Its allocating time update needs no method of its own: the
# Extended Kalman Filter's dispatches `calc_apriori_covariance` on the `Cholesky`
# covariances to the square root one.

"""
$(SIGNATURES)

Square Root Extended Kalman Filter measurement update.
H is the GradientOrJacobianPreparation object.
"""
function measurement_update(
    x,
    P::Cholesky,
    y,
    H::GradientOrJacobianPreparation,
    R::Cholesky;
    consider = nothing,
)
    y_pre, gradient_or_jacobian = value_and_gradient_or_jacobian(H, x)
    ỹ = calc_innovation(y_pre, y)
    PHᵀ, S, P_post =
        calc_cross_cov_innovation_posterior(P, gradient_or_jacobian, R, consider)
    K = calc_kalman_gain(PHᵀ, S.L, consider)
    x_post = calc_posterior_state(x, K, ỹ, consider)
    KFMeasurementUpdate(x_post, P_post, ỹ, S, K)
end

# ── In-place updates ─────────────────────────────────────────────────────────

struct SREKFTUIntermediate{T}
    jacobian::Matrix{T}
    srkf::SRKFTUIntermediate{T}
end

SREKFTUIntermediate(::Type{T}, num_x::Number) where {T} =
    SREKFTUIntermediate(Matrix{T}(undef, num_x, num_x), SRKFTUIntermediate(T, num_x))

SREKFTUIntermediate(num_x::Number) = SREKFTUIntermediate(Float64, num_x)

struct SREKFMUIntermediate{T}
    jacobian::Matrix{T}
    srkf::SRKFMUIntermediate{T,Matrix{T}}
end

function SREKFMUIntermediate(::Type{T}, num_x::Number, num_y::Number) where {T}
    SREKFMUIntermediate(Matrix{T}(undef, num_y, num_x), SRKFMUIntermediate(T, num_x, num_y))
end

SREKFMUIntermediate(num_x::Number, num_y::Number) =
    SREKFMUIntermediate(Float64, num_x, num_y)

"""
$(SIGNATURES)

Square Root Extended Kalman Filter time update in place: `x` and `P` are overwritten
with the a priori state and covariance. `F` is the JacobianPreparation of an in-place
model `f!(y, x)`.
"""
function time_update!(
    tu::SREKFTUIntermediate,
    x,
    P::Cholesky,
    F::InPlaceJacobianPreparation,
    Q::Cholesky,
)
    value_and_jacobian_inplace!(tu.srkf.x_apri, tu.jacobian, F, x)
    copyto!(x, tu.srkf.x_apri)
    calc_apriori_covariance!(tu.srkf, P, tu.jacobian, Q)
    KFTimeUpdate(x, P)
end

"""
$(SIGNATURES)

Square Root Extended Kalman Filter measurement update in place: `x` and `P` are
overwritten with the posterior state and covariance. `H` is the JacobianPreparation of
an in-place model `h!(y, x)`.
"""
function measurement_update!(
    mu::SREKFMUIntermediate,
    x,
    P::Cholesky,
    y,
    H::InPlaceJacobianPreparation,
    R::Cholesky,
)
    y_pre, jacobian = value_and_jacobian_inplace!(mu.srkf.innovation, mu.jacobian, H, x)
    ỹ = y_pre .= y .- y_pre
    calc_sqrt_posterior!(mu.srkf, x, P, ỹ, jacobian, R)
end
