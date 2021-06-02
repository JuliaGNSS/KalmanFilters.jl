struct KFTimeUpdate{X,P} <: AbstractTimeUpdate{X,P}
    state::X
    covariance::P
end

struct KFTUIntermediate{T}
    x_apri::Vector{T}
    p_apri::Matrix{T}
    fp::Matrix{T}
end

KFTUIntermediate(T::Type, num_x::Number) =
    KFTUIntermediate(
        Vector{T}(undef, num_x),
        Matrix{T}(undef, num_x, num_x),
        Matrix{T}(undef, num_x, num_x)
    )

KFTUIntermediate(num_x::Number) = KFTUIntermediate(Float64, num_x)

struct KFMeasurementUpdate{X,P,R,S,K} <: AbstractMeasurementUpdate{X,P}
    state::X
    covariance::P
    innovation::R
    innovation_covariance::S
    kalman_gain::K
end

struct KFMUIntermediate{T,K<:Union{<:AbstractVector{T},<:AbstractMatrix{T}}}
    innovation::Vector{T}
    innovation_covariance::Matrix{T}
    kalman_gain::K
    pht::K
    s_chol::Matrix{T}
    x_posterior::Vector{T}
    p_posterior::Matrix{T}
end

function KFMUIntermediate(T::Type, num_x::Number, num_y::Number)
    return KFMUIntermediate(
        Vector{T}(undef, num_y),
        Matrix{T}(undef, num_y, num_y),
        Matrix{T}(undef, num_x, num_y),
        Matrix{T}(undef, num_x, num_y),
        Matrix{T}(undef, num_y, num_y),
        Vector{T}(undef, num_x),
        Matrix{T}(undef, num_x, num_x)
    )
end

KFMUIntermediate(num_x::Number, num_y::Number) = KFMUIntermediate(Float64, num_x, num_y)

calc_apriori_state(x, F) = F * x
calc_apriori_covariance(P, F, Q) = F * P * F' .+ Q

function time_update(x, P, F::Union{Number, AbstractMatrix}, Q)
    x_apri = calc_apriori_state(x, F)
    P_apri = calc_apriori_covariance(P, F, Q)
    KFTimeUpdate(x_apri, P_apri)
end

function time_update!(tu::KFTUIntermediate, x, P, F::AbstractMatrix, Q)
    x_apri = calc_apriori_state!(tu.x_apri, x, F)
    P_apri = calc_apriori_covariance!(tu.p_apri, tu.fp, P, F, Q)
    KFTimeUpdate(x_apri, P_apri)
end

function measurement_update(x, P, y, H::Union{Number, AbstractVector, AbstractMatrix}, R)
    ỹ = calc_innovation(H, x, y)
    PHᵀ = calc_P_xy(P, H)
    S = calc_innovation_covariance(H, P, R)
    K = calc_kalman_gain(PHᵀ, S)
    x_post = calc_posterior_state(x, K, ỹ)
    P_post = calc_posterior_covariance(P, PHᵀ, K)
    KFMeasurementUpdate(x_post, P_post, ỹ, S, K)
end

@inline calc_P_xy(P, H) = P * H'
@inline calc_innovation(H, x, y) = y .- H * x
@inline calc_innovation_covariance(H, P, R) = H * P * H' .+ R
@inline calc_kalman_gain(PHᵀ, S) = PHᵀ / S
@inline calc_posterior_state(x, K, ỹ) = x .+ K * ỹ
@inline calc_posterior_covariance(P, PHᵀ, K) = P .- PHᵀ * K' # (I - K * H) * P ?

function measurement_update!(mu::KFMUIntermediate, x, P, y, H::AbstractMatrix, R)
    ỹ = calc_innovation!(mu.innovation, H, x, y)
    PHᵀ = calc_P_xy!(mu.pht, P, H)
    S = calc_innovation_covariance!(mu.innovation_covariance, H, PHᵀ, R)
    K = calc_kalman_gain!(mu.s_chol, mu.kalman_gain, PHᵀ, S)
    x_post = calc_posterior_state!(mu.x_posterior, x, K, ỹ)
    P_post = calc_posterior_covariance!(mu.p_posterior, P, PHᵀ, K)
    KFMeasurementUpdate(x_post, P_post, ỹ, S, K)
end

function calc_P_xy!(PHᵀ, P, H)
    mul!(PHᵀ, P, H')
    PHᵀ
end

function calc_apriori_state!(x_apri, x, F)
    x_apri .= @~ F * x
end

function calc_apriori_covariance!(P_apri, FP, P, F, Q)
    FP .= @~ F * P
    P_apri .= @~ FP * F' + Q
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

function calc_posterior_state!(x_posterior, x, K, ỹ)
    x_posterior .= @~ K * ỹ + x
end

function calc_posterior_covariance!(P_posterior, P, PHᵀ, K)
    P_posterior .= @~ -1 * PHᵀ * K' + P # Order is important to trigger BLAS
end