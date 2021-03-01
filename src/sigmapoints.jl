struct WanMerweWeightingParameters <: AbstractWeightingParameters
    α::Float64
    β::Float64
    κ::Float64
end

struct MeanSetWeightingParameters <: AbstractWeightingParameters
    ω₀::Float64
end

struct GaussSetWeightingParameters <: AbstractWeightingParameters
    κ::Float64
end

struct ScaledSetWeightingParameters <: AbstractWeightingParameters
    α::Float64
    β::Float64
    κ::Float64
end

struct SPTimeUpdate{X,P,O} <: AbstractTimeUpdate{X,P}
    state::X
    covariance::P
    χ::O
end

struct SPMeasurementUpdate{X,P,O,I,IC,K} <: AbstractMeasurementUpdate{X,P}
    state::X
    covariance::P
    𝓨::O
    innovation::I
    innovation_covariance::IC
    kalman_gain::K
end

abstract type AbstractSigmaPoints{T} <: AbstractMatrix{T} end

struct SigmaPoints{T, W <: AbstractWeightingParameters} <: AbstractSigmaPoints{T}
    x0::Vector{T}
    P_chol::Matrix{T}
    weight_params::W
    SigmaPoints{T, W}(x0, P_chol, weight_params) where {T<:Real, W<:AbstractWeightingParameters} =
        size(x0, 1) == size(P_chol, 1) == size(P_chol, 2) && P_chol == LowerTriangular(P_chol) ?
        new{T, W}(x0, P_chol, weight_params) :
        error("The length of the first dimension must be equal to the size of P_chol and P_chol must have a LowerTriangular structure")
end

SigmaPoints(x0::Vector{T}, P_chol::LowerTriangular{T}, weight_params::W) where {T<:Real, W<:AbstractWeightingParameters} =
    SigmaPoints{T, W}(x0, collect(P_chol), weight_params)
SigmaPoints(x0::Vector{T}, P_chol::Cholesky{T}, weight_params::W) where {T<:Real, W<:AbstractWeightingParameters} =
    SigmaPoints{T, W}(x0, collect(P_chol.L), weight_params)

function calc_sigma_points(
    x::AbstractVector{T},
    P::AbstractMatrix{T},
    weight_params::W
) where {T, W<:AbstractWeightingParameters}
    weight = calc_cholesky_weight(weight_params, P)
    P_chol = cholesky(Hermitian(P .* weight))
    SigmaPoints{T, W}(x, P_chol.L, weight_params)
end

function calc_sigma_points(
    x::AbstractVector{T},
    P::Cholesky{T},
    weight_params::W
) where {T, W<:AbstractWeightingParameters}
    weight = calc_cholesky_weight(weight_params, P)
    SigmaPoints{T, W}(x, P.L * sqrt(weight), weight_params)
end

function calc_sigma_points!(
    P_chol_temp::AbstractMatrix{T},
    x::AbstractVector{T},
    P::AbstractMatrix{T},
    weight_params::W
) where {T, W<:AbstractWeightingParameters}
    weight = calc_cholesky_weight(weight_params, P)
    P_chol_temp .= P .* weight
    P_chol = cholesky!(Hermitian(P_chol_temp))
    P_chol_temp .= P_chol.uplo === 'U' ? transpose(P_chol.U) : P_chol.L
    SigmaPoints{T, W}(x, P_chol_temp, weight_params)
end

function calc_sigma_points!(
    P_chol_temp::AbstractMatrix{T},
    x::AbstractVector{T},
    P::Cholesky{T},
    weight_params::W
) where {T, W<:AbstractWeightingParameters}
    weight = calc_cholesky_weight(weight_params, P)
    P_chol_temp .= (P.uplo === 'U' ? transpose(P.U) : P.L) .* sqrt(weight)
    SigmaPoints{T, W}(x, P_chol_temp, weight_params)
end

struct TransformedSigmaPoints{T, W <: AbstractWeightingParameters} <: AbstractSigmaPoints{T}
    x0::Vector{T}
    xi::Matrix{T}
    weight_params::W
    TransformedSigmaPoints{T, W}(x0, xi, weight_params) where {T<:Real, W<:AbstractWeightingParameters} =
        size(x0, 1) == size(xi, 1) ?
        new{T, W}(x0, xi, weight_params) :
        error("The length of the first dimension must be the same for all inputs")
end

TransformedSigmaPoints(x0::Vector{T}, xi::Matrix{T}, weight_params::W) where {T<:Real, W<:AbstractWeightingParameters} =
    TransformedSigmaPoints{T, W}(x0, xi, weight_params)

function transform(F, χ::SigmaPoints{T}) where T
    𝓨_x0 = F(χ.x0)
    𝓨_xi = Matrix{T}(undef, length(𝓨_x0), 2 * length(χ.x0))
    xi_temp = Vector{T}(undef, length(χ.x0))
    @inbounds for i = 1:length(χ.x0)
        xi_temp[:] .= χ.x0 .+ @view(χ.P_chol[:, i])
        𝓨_xi[:, i] = F(xi_temp)
    end
    @inbounds for i = 1:length(χ.x0)
        xi_temp[:] .= χ.x0 .- @view(χ.P_chol[:, i])
        𝓨_xi[:, i + length(χ.x0)] = F(xi_temp)
    end
    TransformedSigmaPoints(𝓨_x0, 𝓨_xi, χ.weight_params)
end

function transform!(𝓨::TransformedSigmaPoints{T}, xi_temp, F!, χ::SigmaPoints{T}) where T
    F!(𝓨.x0, χ.x0)
    @inbounds for i = 1:length(χ.x0)
        xi_temp[:] .= χ.x0 .+ @view(χ.P_chol[:, i])
        F!(@view(𝓨.xi[:, i]), xi_temp)
    end
    @inbounds for i = 1:length(χ.x0)
        xi_temp[:] .= χ.x0 .- @view(χ.P_chol[:, i])
        F!(@view(𝓨.xi[:, i + length(χ.x0)]), xi_temp)
    end
    TransformedSigmaPoints(𝓨.x0, 𝓨.xi, χ.weight_params)
end

function mean(𝓨::TransformedSigmaPoints)
    weight_0, weight_i = calc_mean_weights(𝓨)
    𝓨.x0 .* weight_0 .+ vec(sum(𝓨.xi, dims = 2)) .* weight_i
end

function mean!(y::Vector, 𝓨::TransformedSigmaPoints)
    weight_0, weight_i = calc_mean_weights(𝓨)
    sum!(y, 𝓨.xi)
    y .= y .* weight_i .+ 𝓨.x0 .* weight_0
end

function substract_mean(𝓨::TransformedSigmaPoints, y)
    𝓨 .- y
end

function substract_mean!(unbiased_𝓨::TransformedSigmaPoints, 𝓨::TransformedSigmaPoints, y)
    unbiased_𝓨 .= 𝓨 .- y
    TransformedSigmaPoints(unbiased_𝓨.x0, unbiased_𝓨.xi, 𝓨.weight_params)
end

function cov(unbiased_𝓨::TransformedSigmaPoints, Q::AbstractMatrix)
    weight_0, weight_i = calc_cov_weights(unbiased_𝓨)
    cov(unbiased_𝓨::TransformedSigmaPoints, Augment(Q)) .+ Q
end

function cov!(P, unbiased_𝓨::TransformedSigmaPoints, Q::AbstractMatrix)
    weight_0, weight_i = calc_cov_weights(unbiased_𝓨)
    P .= @~ unbiased_𝓨.x0 * unbiased_𝓨.x0' .* weight_0 .+ unbiased_𝓨.xi * unbiased_𝓨.xi' .* weight_i .+ Q
end

function cov(χ::SigmaPoints, unbiased_𝓨::TransformedSigmaPoints)
    weight_0, weight_i = calc_cov_weights(χ)
    num_states = length(χ.x0)
    χ.P_chol * (@view(unbiased_𝓨.xi[:, 1:num_states]))' .* weight_i .-
        χ.P_chol * (@view(unbiased_𝓨.xi[:, num_states + 1:2 * num_states]))' .* weight_i
end

function cov!(P, χ::SigmaPoints, unbiased_𝓨::TransformedSigmaPoints)
    weight_0, weight_i = calc_cov_weights(χ)
    num_states = length(χ.x0)
    P .= @~ χ.P_chol * (@view(unbiased_𝓨.xi[:, 1:num_states]))' .* weight_i
    P .-= @~ χ.P_chol * (@view(unbiased_𝓨.xi[:, num_states + 1:2 * num_states]))' .* weight_i
    P
end

function mean_and_cov(𝓨::TransformedSigmaPoints, Q)
    y = mean(𝓨)
    Ryy = cov(𝓨 .- y, Q)
    y, Ryy
end

lambda(weight_params::WanMerweWeightingParameters, L) = weight_params.α^2 * (L + weight_params.κ) - L

function calc_mean_weights(weight_params::WanMerweWeightingParameters, num_states)
    λ = lambda(weight_params, num_states)
    weight_0 = λ / (num_states + λ)
    weight_i = 1 / (2 * (num_states + λ))
    weight_0, weight_i
end

function calc_cov_weights(weight_params::WanMerweWeightingParameters, num_states)
    weight_0, weight_i = calc_mean_weights(weight_params, num_states)
    weight_0 + 1 - weight_params.α^2 + weight_params.β, weight_i
end

function calc_cholesky_weight(weight_params::WanMerweWeightingParameters, num_states::Real)
    num_states + lambda(weight_params, num_states)
end

function calc_mean_weights(weight_params::MeanSetWeightingParameters, num_states)
    weight_params.ω₀, (1 - weight_params.ω₀) / (2 * num_states)
end

calc_cov_weights(weight_params::MeanSetWeightingParameters, num_states) =
    calc_mean_weights(weight_params, num_states)

function calc_cholesky_weight(weight_params::MeanSetWeightingParameters, num_states::Real)
    num_states / (1 - weight_params.ω₀)
end

function calc_mean_weights(weight_params::GaussSetWeightingParameters, num_states)
    1 - num_states / weight_params.κ, 1 / (2 * weight_params.κ)
end

calc_cov_weights(weight_params::GaussSetWeightingParameters, num_states) =
    calc_mean_weights(weight_params, num_states)

function calc_cholesky_weight(weight_params::GaussSetWeightingParameters, num_states::Real)
    weight_params.κ
end

function calc_mean_weights(weight_params::ScaledSetWeightingParameters, num_states)
    weight_0 = (weight_params.α^2 * weight_params.κ - num_states) / (weight_params.α^2 * weight_params.κ)
    weight_i = 1 / (2 * weight_params.α^2 * weight_params.κ)
    weight_0, weight_i
end

function calc_cov_weights(weight_params::ScaledSetWeightingParameters, num_states)
    weight_0, weight_i = calc_mean_weights(weight_params, num_states)
    weight_0 + 1 - weight_params.α^2 + weight_params.β, weight_i
end

function calc_cholesky_weight(weight_params::ScaledSetWeightingParameters, num_states::Real)
    weight_params.α^2 * weight_params.κ
end

function calc_mean_weights(χ::AbstractSigmaPoints)
    calc_mean_weights(χ.weight_params, (size(χ, 2) - 1) >> 1)
end

function calc_cov_weights(χ::AbstractSigmaPoints)
    calc_cov_weights(χ.weight_params, (size(χ, 2) - 1) >> 1)
end

function calc_cholesky_weight(weight_params::AbstractWeightingParameters, P)
    calc_cholesky_weight(weight_params, size(P, 2))
end

Base.size(S::SigmaPoints) = (length(S.x0), 2 * length(S.x0) + 1)

Base.getindex(S::SigmaPoints{T}, inds::Vararg{Int,2}) where {T} =
    @inbounds if inds[2] == 1
        S.x0[inds[1]]
    elseif 1 < inds[2] <= length(S.x0) + 1
        S.x0[inds[1]] + S.P_chol[inds[1], inds[2] - 1]
    else
        S.x0[inds[1]] - S.P_chol[inds[1], inds[2] - length(S.x0) - 1]
    end

# make TransformedSigmaPoints broadcastable
Base.size(S::TransformedSigmaPoints) = (length(S.x0), size(S.xi, 2) + 1)

Base.getindex(S::TransformedSigmaPoints{T}, inds::Vararg{Int,2}) where {T} =
    @inbounds if inds[2] == 1
        S.x0[inds[1]]
    else
        S.xi[inds[1], inds[2] - 1]
    end

Base.setindex!(S::TransformedSigmaPoints{T}, val, inds::Vararg{Int,2}) where {T} =
    @inbounds if inds[2] == 1
        S.x0[inds[1]] = val
    else
        S.xi[inds[1], inds[2] - 1] = val
    end

Base.BroadcastStyle(::Type{<:TransformedSigmaPoints}) = Broadcast.ArrayStyle{TransformedSigmaPoints}()

Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{TransformedSigmaPoints}}, ::Type{T}) where T =
    TransformedSigmaPoints(similar(bc.args[1].x0), similar(bc.args[1].xi), bc.args[1].weight_params)

function Base.copyto!(dest::TransformedSigmaPoints, bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{TransformedSigmaPoints}})
    for idx in eachindex(bc)
        @inbounds dest[idx] = bc[idx]
    end
    dest
end
