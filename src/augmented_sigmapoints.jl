struct Augment{B}
    noise::B
end

struct Augmented{A, B}
    P::A
    noise::B
end

Augmented(P, B::Augment) = Augmented(P, B.noise)

Base.size(A::Augmented) = (size(A.P, 1), size(A.P, 2) + size(A.noise, 2))
Base.size(A::Augmented, d::Integer) = size(A)[d]

struct AugmentedSigmaPoints{T, V <: AbstractVector{T}, L <: LowerTriangular{T}, W <: AbstractWeightingParameters} <: AbstractSigmaPoints{T}
    x0::V
    P_chol::L
    noise_chol::L
    weight_params::W
    AugmentedSigmaPoints{T, V, L, W}(x0, P_chol, noise_chol, weight_params) where {T<:Real, V<:AbstractVector{T}, L<:LowerTriangular{T}, W<:AbstractWeightingParameters} =
        size(x0, 1) == size(P_chol, 1) == size(P_chol, 2) ?
        new{T, V, L, W}(x0, P_chol, noise_chol, weight_params) :
        error("The length of the first dimension must be equal to the size of P_chol")
end

Base.size(S::AugmentedSigmaPoints) = (length(S.x0) + size(S.noise_chol, 1), 2 * size(S.P_chol, 2) + 2 * size(S.noise_chol, 2) + 1)

Base.getindex(S::AugmentedSigmaPoints{T}, inds::Vararg{Int,2}) where {T} =
    @inbounds if inds[2] == 1
        inds[1] <= length(S.x0) ? S.x0[inds[1]] : zero(T)
    elseif 1 < inds[2] <= size(S.P_chol, 2) + 1
        inds[1] <= length(S.x0) ? S.x0[inds[1]] + S.P_chol[inds[1], inds[2] - 1] : zero(T)
    elseif size(S.P_chol, 2) + 1 < inds[2] <= size(S.P_chol, 2) + size(S.noise_chol, 2) + 1
        inds[1] <= length(S.x0) ? S.x0[inds[1]] : S.noise_chol[inds[1] - length(S.x0), inds[2] - size(S.P_chol, 2) - 1]
    elseif size(S.P_chol, 2) + size(S.noise_chol, 2) + 1 < inds[2] <= 2 * size(S.P_chol, 2) + size(S.noise_chol, 2) + 1
        inds[1] <= length(S.x0) ? S.x0[inds[1]] - S.P_chol[inds[1], inds[2] - size(S.P_chol, 2) - size(S.noise_chol, 2) - 1] : zero(T)
    else
        inds[1] <= length(S.x0) ? S.x0[inds[1]] : -S.noise_chol[inds[1] - length(S.x0), inds[2] - 2 * size(S.P_chol, 2) - size(S.noise_chol, 2) - 1]
    end

AugmentedSigmaPoints(x0::V, P_chol::L, noise_chol::L, weight_params::W) where {T<:Real, V<:AbstractVector{T}, L<:LowerTriangular{T}, W<:AbstractWeightingParameters} =
    AugmentedSigmaPoints{T, V, L, W}(x0, P_chol, noise_chol, weight_params)
AugmentedSigmaPoints(x0::V, P_chol::L, noise_chol::Cholesky{T}, weight_params::W) where {T<:Real, V<:AbstractVector{T}, L<:LowerTriangular{T}, W<:AbstractWeightingParameters} =
    AugmentedSigmaPoints{T, V, L, W}(x0, P_chol, noise_chol.L, weight_params)
AugmentedSigmaPoints(x0::V, P_chol::Cholesky{T}, noise_chol::L, weight_params::W) where {T<:Real, V<:AbstractVector{T}, L<:LowerTriangular{T}, W<:AbstractWeightingParameters} =
    AugmentedSigmaPoints{T, V, L, W}(x0, P_chol.L, noise_chol, weight_params)
AugmentedSigmaPoints(x0::V, P_chol::Cholesky{T}, noise_chol::Cholesky{T}, weight_params::W) where {T<:Real, V<:AbstractVector{T}, W<:AbstractWeightingParameters} =
    AugmentedSigmaPoints(x0, P_chol.L, noise_chol.L, weight_params)

function calc_sigma_points(
    x::AbstractVector{T},
    P::Augmented{<:AbstractMatrix{T}, <:AbstractMatrix{T}},
    weight_params::W
) where {T, W<:AbstractWeightingParameters}
    weight = calc_cholesky_weight(weight_params, P)
    P_chol = cholesky(Hermitian(P.P .* weight, :L))
    noise_chol = cholesky(Hermitian(P.noise .* weight, :L))
    AugmentedSigmaPoints(x, P_chol.L, noise_chol.L, weight_params)
end

function calc_sigma_points(
    x::AbstractVector{T},
    P::Augmented{<:Cholesky{T}, <:Cholesky{T}},
    weight_params::W
) where {T, W<:AbstractWeightingParameters}
    weight = calc_cholesky_weight(weight_params, P)
    AugmentedSigmaPoints(x, P.P.L * sqrt(weight), P.noise.L * sqrt(weight), weight_params)
end

function calc_sigma_points!(
    P_chol_temp::Augmented{<:AbstractMatrix{T}, <:AbstractMatrix{T}},
    x::AbstractVector{T},
    P::Augmented{<:AbstractMatrix{T}, <:AbstractMatrix{T}},
    weight_params::W
) where {T, W<:AbstractWeightingParameters}
    weight = calc_cholesky_weight(weight_params, P)
    P_chol_temp.P .= P.P .* weight
    P_chol = cholesky!(Hermitian(P_chol_temp.P, :L))
    P_chol_temp.noise .= P.noise .* weight
    P_chol_noise = cholesky!(Hermitian(P_chol_temp.noise, :L))
    AugmentedSigmaPoints(x, P_chol.L, P_chol_noise.L, weight_params)
end

function calc_sigma_points!(
    P_chol_temp::Augmented{<:AbstractMatrix{T}, <:AbstractMatrix{T}},
    x::AbstractVector{T},
    P::Augmented{<:Cholesky{T}, <:Cholesky{T}},
    weight_params::W
) where {T, W<:AbstractWeightingParameters}
    weight = calc_cholesky_weight(weight_params, P)
    P_chol_temp.P .= (P.P.uplo === 'U' ? transpose(P.P.U) : P.P.L) .* sqrt(weight)
    P_chol_temp.noise .= (P.noise.uplo === 'U' ? transpose(P.noise.U) : P.noise.L) .* sqrt(weight)
    AugmentedSigmaPoints(x, LowerTriangular(P_chol_temp.P), LowerTriangular(P_chol_temp.noise), weight_params)
end

function transform(F, χ::AugmentedSigmaPoints{T}) where T
    𝓨_x0 = F(χ.x0)
    num_x = length(χ.x0)
    𝓨_xi = Matrix{T}(undef, length(𝓨_x0), 2 * size(χ.P_chol, 2) + 2 * size(χ.noise_chol, 2))
    xi_temp = copy(χ.x0)
    @inbounds for i = size(χ.P_chol, 2):-1:1
        xi_temp[i:num_x] .= @view(χ.x0[i:num_x]) .+ @view(χ.P_chol.data[i:num_x, i])
        𝓨_xi[:, i] = F(xi_temp)
        xi_temp[i:num_x] .= @view(χ.x0[i:num_x]) .- @view(χ.P_chol.data[i:num_x, i])
        𝓨_xi[:, i + size(χ.P_chol, 2) + size(χ.noise_chol, 2)] = F(xi_temp)
    end
    @inbounds for i = 1:size(χ.noise_chol, 2)
        𝓨_xi[:, i + size(χ.P_chol, 2)] = F(χ.x0, @view(χ.noise_chol[:, i]))
        𝓨_xi[:, i + 2 * size(χ.P_chol, 2) + size(χ.noise_chol, 2)] = F(χ.x0, -@view(χ.noise_chol[:, i]))
    end
    TransformedSigmaPoints(𝓨_x0, 𝓨_xi, χ.weight_params)
end

function transform!(𝓨::TransformedSigmaPoints{T}, xi_temp, F!, χ::AugmentedSigmaPoints{T}) where T
    F!(𝓨.x0, χ.x0)
    num_x = length(χ.x0)
    xi_temp.P .= χ.x0
    @inbounds for i = size(χ.P_chol, 2):-1:1
        xi_temp.P[i:num_x] .= @view(χ.x0[i:num_x]) .+ @view(χ.P_chol.data[i:num_x, i])
        F!(@view(𝓨.xi[:, i]), xi_temp.P)
        xi_temp.P[i:num_x] .= @view(χ.x0[i:num_x]) .- @view(χ.P_chol.data[i:num_x, i])
        F!(@view(𝓨.xi[:, i + size(χ.P_chol, 2) + size(χ.noise_chol, 2)]), xi_temp.P)
    end
    @inbounds for i = 1:size(χ.noise_chol, 2)
        F!(@view(𝓨.xi[:, i + size(χ.P_chol, 2)]), χ.x0, @view(χ.noise_chol[:, i]))
        xi_temp.noise[:] .= -1 .* @view(χ.noise_chol[:, i])
        F!(@view(𝓨.xi[:, i + 2 * size(χ.P_chol, 2) + size(χ.noise_chol, 2)]), χ.x0, xi_temp.noise)
    end
    TransformedSigmaPoints(𝓨.x0, 𝓨.xi, χ.weight_params)
end

function cov(χ::AugmentedSigmaPoints, unbiased_𝓨::TransformedSigmaPoints)
    weight_0, weight_i = calc_cov_weights(χ)
    num_states = length(χ.x0)
    num_noise_states = size(χ.noise_chol, 2)
    (χ.P_chol * (@view(unbiased_𝓨.xi[:, 1:num_states]))' .-
        χ.P_chol * (@view(unbiased_𝓨.xi[:, num_states + num_noise_states + 1:2 * num_states + num_noise_states]))') .* weight_i
end

function cov!(P, χ::AugmentedSigmaPoints, unbiased_𝓨::TransformedSigmaPoints)
    weight_0, weight_i = calc_cov_weights(χ)
    num_states = length(χ.x0)
    num_noise_states = size(χ.noise_chol, 2)
    P .= @~ χ.P_chol * (@view(unbiased_𝓨.xi[:, 1:num_states]))'
    P .-= @~ χ.P_chol * (@view(unbiased_𝓨.xi[:, num_states + num_noise_states + 1:2 * num_states + num_noise_states]))'
    P .*= weight_i
end

function cov(unbiased_𝓨::TransformedSigmaPoints, Q::Augment)
    weight_0, weight_i = calc_cov_weights(unbiased_𝓨)
    unbiased_𝓨.x0 * unbiased_𝓨.x0' .* weight_0 .+ unbiased_𝓨.xi * unbiased_𝓨.xi' .* weight_i
end

function cov!(P, unbiased_𝓨::TransformedSigmaPoints, Q::Augment)
    weight_0, weight_i = calc_cov_weights(unbiased_𝓨)
    P .= @~ unbiased_𝓨.x0 * unbiased_𝓨.x0' .* weight_0 .+ unbiased_𝓨.xi * unbiased_𝓨.xi' .* weight_i
end