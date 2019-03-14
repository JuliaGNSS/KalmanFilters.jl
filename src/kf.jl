struct KFTimeUpdate{X,P} <: AbstractTimeUpdate
    state::X
    covariance::P
end

struct KFTUIntermediate{T}
    state_temp::Vector{T}
    fp::Matrix{T}
end

KFTUIntermediate(T::Type, num_x::Number) =
    KFTUIntermediate(
        Vector{T}(undef, num_x),
        Matrix{T}(undef, num_x, num_x)
    )

KFTUIntermediate(num_x::Number) = KFTUIntermediate(Float64, num_x)

struct KFMeasurementUpdate{X,P,R,S,K} <: AbstractMeasurementUpdate
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
    s_lu::Matrix{T}
end

function KFMUIntermediate(T::Type, num_x::Number, num_y::Number)
    if num_y == 1
        return KFMUIntermediate(
            Vector{T}(undef, num_y),
            Matrix{T}(undef, num_y, num_y),
            Vector{T}(undef, num_x),
            Vector{T}(undef, num_x),
            Matrix{T}(undef, num_y, num_y)
        )
    elseif num_x == 1
        return KFMUIntermediate(
            Vector{T}(undef, num_y),
            Matrix{T}(undef, num_y, num_y),
            adjoint(Vector{T}(undef, num_y)),
            adjoint(Vector{T}(undef, num_y)),
            Matrix{T}(undef, num_y, num_y)
        )
    else
        return KFMUIntermediate(
            Vector{T}(undef, num_y),
            Matrix{T}(undef, num_y, num_y),
            Matrix{T}(undef, num_x, num_y),
            Matrix{T}(undef, num_x, num_y),
            Matrix{T}(undef, num_y, num_y)
        )
    end
end

KFMUIntermediate(num_x::Number, num_y::Number) = KFMUIntermediate(Float64, num_x, num_y)

function time_update(mu::T, F, Q) where T <: Union{KalmanInits, <:AbstractMeasurementUpdate}
    x, P = state(mu), covariance(mu)
    x_apri = F * x
    P_apri = F * P * F' .+ Q
    KFTimeUpdate(x_apri, P_apri)
end

function time_update!(tu::KFTUIntermediate, mu::T, F, Q) where T <: Union{KalmanInits, <:AbstractMeasurementUpdate}
    x, P = state(mu), covariance(mu)
    x_apri = calc_apriori_state!(tu.state_temp, x, F)
    P_apri = calc_apriori_covariance!(tu.fp, P, F, Q)
    KFTimeUpdate(x_apri, P_apri)
end

function measurement_update(y, tu::T, H, R) where T <: Union{KalmanInits, <:AbstractTimeUpdate}
    x, P = state(tu), covariance(tu)
    ỹ = y .- H * x
    PHᵀ = P * H'
    S = H * PHᵀ .+ R
    K = PHᵀ / S
    x_post = x .+ K * ỹ
    P_post = P - PHᵀ * K' # (I - K * H) * P ?
    KFMeasurementUpdate(x_post, P_post, ỹ, S, K)
end

function measurement_update!(mu::KFMUIntermediate, y, tu::T, H, R) where T <: Union{KalmanInits, <:AbstractTimeUpdate}
    x, P = state(tu), covariance(tu)
    PHᵀ = mu.pht
    ỹ = calc_innovation!(mu.innovation, H, x, y)
    mul!(PHᵀ, P, H')
    S = calc_innovation_covariance!(mu.innovation_covariance, H, PHᵀ, R)
    K = calc_kalman_gain!(mu.s_lu, mu.kalman_gain, PHᵀ, S)
    x_post = calc_posterior_state!(x, K, ỹ)
    P_post = calc_posterior_covariance!(P, PHᵀ, K)
    KFMeasurementUpdate(x_post, P_post, ỹ, S, K)
end

function calc_apriori_state!(x_temp, x, F)
    mul!(x_temp, F, x)
end

function calc_apriori_covariance!(FP, P, F, Q)
    mul!(FP, F, P)
    P .= Mul(FP, F') .+ Q
end

function calc_innovation!(ỹ, H, x::AbstractVector, y::AbstractVector)
    ỹ .= (-1.) .* Mul(H, x) .+ y # Order is important to trigger BLAS
end

function calc_innovation!(ỹ, H, x::AbstractVector, y)
    y - H * x
end

function calc_innovation!(ỹ, H, x, y::AbstractVector)
    ỹ .= y .- H .* x
end

function calc_innovation_covariance!(S, H, PHᵀ, R::AbstractMatrix)
    S .= Mul(H, PHᵀ) .+ R
end

# Can be removed once https://github.com/JuliaArrays/LazyArrays.jl/issues/27 is fixed
function calc_innovation_covariance!(S, H::AbstractVector, PHᵀ, R::AbstractMatrix)
    mul!(S, H, PHᵀ)
    S .+= R
end

function calc_innovation_covariance!(S, H, PHᵀ, R)
    H * PHᵀ + R
end

function calc_kalman_gain!(S_lu, K, PHᵀ, S::AbstractMatrix)
    S_lu .= S
    K .= PHᵀ
    rdiv!(K, S_lu)
end

function calc_kalman_gain!(S_lu, K, PHᵀ, S)
    PHᵀ ./ S
end

function calc_posterior_state!(x::AbstractVector, K, ỹ::AbstractVector)
    x .= Mul(K, ỹ) .+ x
end

function calc_posterior_state!(x::AbstractVector, K, ỹ)
    x .= K .* ỹ .+ x
end

function calc_posterior_state!(x, K, ỹ::AbstractVector)
    K * ỹ + x
end

function calc_posterior_covariance!(P::AbstractMatrix, PHᵀ, K)
    P .= (-1.) .* Mul(PHᵀ, K') .+ P
end

# Can be removed once https://github.com/JuliaArrays/LazyArrays.jl/issues/27 is fixed
function calc_posterior_covariance!(P::AbstractMatrix, PHᵀ::AbstractVector, K)
    P .-= PHᵀ * K'
end

function calc_posterior_covariance!(P, PHᵀ, K)
    P - PHᵀ * K'
end

function adjoint!(X)
    for i = 1:size(X,1), j = i:size(X,2)
        @inbounds X[i,j], X[j,i] = X[j,i]', X[i,j]'
    end
    return X
end

function rdiv!(A::StridedVecOrMat, B::StridedMatrix)
    adjoint(ldiv!(adjoint(lu!(B)), adjoint!(A)))
end

function rdiv!(adjA::Adjoint{<:Any, <:StridedVecOrMat}, B::StridedMatrix)
    A = adjA.parent
    adjoint(ldiv!(adjoint(lu!(B)), A))
end

function rdiv!(transA::Transpose{<:Any, <:StridedVecOrMat}, B::StridedMatrix)
    A = transA.parent
    adjoint(ldiv!(adjoint(lu!(B)), A))
end
