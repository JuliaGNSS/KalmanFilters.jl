struct KFTimeUpdate{X,P} <: AbstractTimeUpdate{X,P}
    state::X
    covariance::P
end

struct KFMeasurementUpdate{X,P,R,S,K} <: AbstractMeasurementUpdate{X,P}
    state::X
    covariance::P
    innovation::R
    innovation_covariance::S
    kalman_gain::K
end

"""
$(SIGNATURES)

Kalman Filter time update.
"""
function time_update(x, P, F::Union{Number,AbstractMatrix}, Q)
    x_apri = calc_apriori_state(x, F)
    P_apri = calc_apriori_covariance(P, F, Q)
    KFTimeUpdate(x_apri, P_apri)
end

"""
$(SIGNATURES)

Kalman Filter measurement update.
"""
function measurement_update(
    x,
    P,
    y,
    H::Union{Number,AbstractVector,AbstractMatrix},
    R;
    consider = nothing,
)
    ỹ = calc_innovation(H, x, y)
    PHᵀ = calc_P_xy(P, H)
    S = calc_innovation_covariance(H, P, R)
    K = calc_kalman_gain(PHᵀ, S, consider)
    x_post = calc_posterior_state(x, K, ỹ, consider)
    P_post = calc_posterior_covariance(P, PHᵀ, K, consider)
    KFMeasurementUpdate(x_post, P_post, ỹ, S, K)
end

calc_apriori_state(x, F) = F * x
calc_apriori_covariance(P, F, Q) = F * P * F' + Q

calc_P_xy(P, H) = P * H'
calc_innovation(H, x, y) = y - H * x
calc_innovation_covariance(H, P, R) = H * P * H' + R
calc_kalman_gain(PHᵀ, S, consider::Nothing) = PHᵀ / S
calc_posterior_state(x, K, ỹ, consider::Nothing) = x + K * ỹ
calc_posterior_covariance(P, PHᵀ, K, consider::Nothing) = P - PHᵀ * K' # (I - K * H) * P ?

# ── In-place updates ─────────────────────────────────────────────────────────
#
# The `!` variants write the new state into `x` and the new covariance into `P` (for a
# `Cholesky`, into the factor it holds, whichever triangle that is) and keep everything
# else in a preallocated intermediate. They return the same update object as the
# allocating variants, whose state and covariance then alias `x` and `P`.

struct KFTUIntermediate{T}
    x_apri::Vector{T}
    fp::Matrix{T}
end

KFTUIntermediate(::Type{T}, num_x::Number) where {T} =
    KFTUIntermediate(Vector{T}(undef, num_x), Matrix{T}(undef, num_x, num_x))

KFTUIntermediate(num_x::Number) = KFTUIntermediate(Float64, num_x)

struct KFMUIntermediate{T,K<:Union{<:AbstractVector{T},<:AbstractMatrix{T}}}
    innovation::Vector{T}
    innovation_covariance::Matrix{T}
    kalman_gain::K
    pht::K
    s_chol::Matrix{T}
end

function KFMUIntermediate(::Type{T}, num_x::Number, num_y::Number) where {T}
    return KFMUIntermediate(
        Vector{T}(undef, num_y),
        Matrix{T}(undef, num_y, num_y),
        Matrix{T}(undef, num_x, num_y),
        Matrix{T}(undef, num_x, num_y),
        Matrix{T}(undef, num_y, num_y),
    )
end

KFMUIntermediate(num_x::Number, num_y::Number) = KFMUIntermediate(Float64, num_x, num_y)

"""
$(SIGNATURES)

Kalman Filter measurement update in place: `x` and `P` are overwritten with the
posterior state and covariance.
"""
function measurement_update!(mu::KFMUIntermediate, x, P, y, H::AbstractMatrix, R)
    ỹ = calc_innovation!(mu.innovation, H, x, y)
    PHᵀ = calc_P_xy!(mu.pht, P, H)
    S = calc_innovation_covariance!(mu.innovation_covariance, H, PHᵀ, R)
    K = calc_kalman_gain!(mu.s_chol, mu.kalman_gain, PHᵀ, S)
    calc_posterior_state!(x, K, ỹ)
    calc_posterior_covariance!(P, PHᵀ, K)
    KFMeasurementUpdate(x, P, ỹ, S, K)
end

"""
$(SIGNATURES)

Kalman Filter time update in place: `x` and `P` are overwritten with the a priori
state and covariance.
"""
function time_update!(tu::KFTUIntermediate, x, P, F::AbstractMatrix, Q)
    calc_apriori_state!(tu.x_apri, x, F)
    copyto!(x, tu.x_apri)
    calc_apriori_covariance!(P, tu.fp, F, Q)
    KFTimeUpdate(x, P)
end

function calc_P_xy!(PHᵀ, P, H)
    mul!(PHᵀ, P, H')
    PHᵀ
end

calc_apriori_state!(x_apri, x, F) = mul!(x_apri, F, x)

# P ← F P Fᵀ + Q, with `FP` as scratch for the first product
function calc_apriori_covariance!(P, FP, F, Q)
    mul!(FP, F, P)
    mul!(P, FP, F')
    P .+= Q
end

function calc_innovation!(ỹ, H, x, y)
    ỹ .= @~ -1 * H * x + y # Order is important to trigger BLAS
end

function calc_innovation_covariance!(S, H, PHᵀ, R)
    S .= @~ H * PHᵀ + R
end

function calc_kalman_gain!(S_chol, K, PHᵀ, S)
    S_chol .= S
    K .= PHᵀ
    rdiv!(K, cholesky!(Hermitian(S_chol)))
end

# x ← x + K ỹ
calc_posterior_state!(x, K, ỹ) = mul!(x, K, ỹ, true, true)

# P ← P − PHᵀ Kᵀ
calc_posterior_covariance!(P, PHᵀ, K) = mul!(P, PHᵀ, K', -1, true)
