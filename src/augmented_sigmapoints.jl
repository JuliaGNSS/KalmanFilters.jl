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

struct AugmentedSigmaPoints{T, W <: AbstractWeightingParameters} <: AbstractSigmaPoints{T}
    x0::Vector{T}
    P_chol::Matrix{T}
    noise_chol::Matrix{T}
    weight_params::W
    AugmentedSigmaPoints{T, W}(x0, P_chol, noise_chol, weight_params) where {T<:Real, W<:AbstractWeightingParameters} =
        size(x0, 1) == size(P_chol, 1) == size(P_chol, 2) && P_chol == LowerTriangular(P_chol) && noise_chol == LowerTriangular(noise_chol) ?
        new{T, W}(x0, P_chol, noise_chol, weight_params) :
        error("The length of the first dimension must be equal to the size of P_chol and P_chol and noise_chol must have a LowerTriangular structure")
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

AugmentedSigmaPoints(x0::Vector{T}, P_chol::LowerTriangular{T}, noise_chol::LowerTriangular{T}, weight_params::W) where {T<:Real, W<:AbstractWeightingParameters} =
    AugmentedSigmaPoints{T, W}(x0, collect(P_chol), collect(noise_chol), weight_params)
AugmentedSigmaPoints(x0::Vector{T}, P_chol::LowerTriangular{T}, noise_chol::Cholesky{T}, weight_params::W) where {T<:Real, W<:AbstractWeightingParameters} =
    AugmentedSigmaPoints{T, W}(x0, collect(P_chol), collect(noise_chol.L), weight_params)
AugmentedSigmaPoints(x0::Vector{T}, P_chol::Cholesky{T}, noise_chol::LowerTriangular{T}, weight_params::W) where {T<:Real, W<:AbstractWeightingParameters} =
    AugmentedSigmaPoints{T, W}(x0, collect(P_chol.L), collect(noise_chol), weight_params)
AugmentedSigmaPoints(x0::Vector{T}, P_chol::Cholesky{T}, noise_chol::Cholesky{T}, weight_params::W) where {T<:Real, W<:AbstractWeightingParameters} =
    AugmentedSigmaPoints{T, W}(x0, collect(P_chol.L), collect(noise_chol.L), weight_params)

function calc_sigma_points(
    x::AbstractVector{T},
    P::Augmented{<:AbstractMatrix{T}, <:AbstractMatrix{T}},
    weight_params::W
) where {T, W<:AbstractWeightingParameters}
    weight = calc_cholesky_weight(weight_params, P)
    P_chol = cholesky(Hermitian(P.P .* weight))
    noise_chol = cholesky(Hermitian(P.noise .* weight))
    AugmentedSigmaPoints{T, W}(x, P_chol.L, noise_chol.L, weight_params)
end

function calc_sigma_points(
    x::AbstractVector{T},
    P::Augmented{<:Cholesky{T}, <:Cholesky{T}},
    weight_params::W
) where {T, W<:AbstractWeightingParameters}
    weight = calc_cholesky_weight(weight_params, P)
    AugmentedSigmaPoints{T, W}(x, P.P.L * sqrt(weight), P.noise.L * sqrt(weight), weight_params)
end

function calc_sigma_points!(
    P_chol_temp::Augmented{<:AbstractMatrix{T}, <:AbstractMatrix{T}},
    x::AbstractVector{T},
    P::Augmented{<:AbstractMatrix{T}, <:AbstractMatrix{T}},
    weight_params::W
) where {T, W<:AbstractWeightingParameters}
    weight = calc_cholesky_weight(weight_params, P)
    P_chol_temp.P .= P.P .* weight
    P_chol = cholesky!(Hermitian(P_chol_temp.P))
    P_chol_temp.P .= P_chol.uplo === 'U' ? transpose(P_chol.U) : P_chol.L
    P_chol_temp.noise .= P.noise .* weight
    P_chol = cholesky!(Hermitian(P_chol_temp.noise))
    P_chol_temp.noise .= P_chol.uplo === 'U' ? transpose(P_chol.U) : P_chol.L
    AugmentedSigmaPoints{T, W}(x, P_chol_temp.P, P_chol_temp.noise, weight_params)
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
    AugmentedSigmaPoints{T, W}(x, P_chol_temp.P, P_chol_temp.noise, weight_params)
end

function transform(F, χ::AugmentedSigmaPoints{T}) where T
    𝓨_x0 = F(χ.x0)
    𝓨_xi = Matrix{T}(undef, length(𝓨_x0), 2 * size(χ.P_chol, 2) + 2 * size(χ.noise_chol, 2))
    xi_temp = Vector{T}(undef, length(χ.x0))
    @inbounds for i = 1:size(χ.P_chol, 2)
        xi_temp[:] .= χ.x0 .+ @view(χ.P_chol[:, i])
        𝓨_xi[:, i] = F(xi_temp)
    end
    @inbounds for i = 1:size(χ.noise_chol, 2)
        𝓨_xi[:, i + size(χ.P_chol, 2)] = F(χ.x0, @view(χ.noise_chol[:, i]))
    end
    @inbounds for i = 1:size(χ.P_chol, 2)
        xi_temp[:] .= χ.x0 .- @view(χ.P_chol[:, i])
        𝓨_xi[:, i + size(χ.P_chol, 2) + size(χ.noise_chol, 2)] = F(xi_temp)
    end
    @inbounds for i = 1:size(χ.noise_chol, 2)
        𝓨_xi[:, i + 2 * size(χ.P_chol, 2) + size(χ.noise_chol, 2)] = F(χ.x0, -@view(χ.noise_chol[:, i]))
    end
    TransformedSigmaPoints(𝓨_x0, 𝓨_xi, χ.weight_params)
end

function transform!(𝓨::TransformedSigmaPoints{T}, xi_temp, F!, χ::AugmentedSigmaPoints{T}) where T
    F!(𝓨.x0, χ.x0)
    @inbounds for i = 1:size(χ.P_chol, 2)
        xi_temp.P[:] .= χ.x0 .+ @view(χ.P_chol[:, i])
        F!(@view(𝓨.xi[:, i]), xi_temp.P)
    end
    @inbounds for i = 1:size(χ.noise_chol, 2)
        F!(@view(𝓨.xi[:, i + size(χ.P_chol, 2)]), χ.x0, @view(χ.noise_chol[:, i]))
    end
    @inbounds for i = 1:size(χ.P_chol, 2)
        xi_temp.P[:] .= χ.x0 .- @view(χ.P_chol[:, i])
        F!(@view(𝓨.xi[:, i + size(χ.P_chol, 2) + size(χ.noise_chol, 2)]), xi_temp.P)
    end
    @inbounds for i = 1:size(χ.noise_chol, 2)
        xi_temp.noise[:] .= -1 .* @view(χ.noise_chol[:, i])
        F!(@view(𝓨.xi[:, i + 2 * size(χ.P_chol, 2) + size(χ.noise_chol, 2)]), χ.x0, xi_temp.noise)
    end
    TransformedSigmaPoints(𝓨.x0, 𝓨.xi, χ.weight_params)
end

function cov(χ::AugmentedSigmaPoints, unbiased_𝓨::TransformedSigmaPoints)
    weight_0, weight_i = calc_cov_weights(χ)
    num_states = length(χ.x0)
    num_noise_states = size(χ.noise_chol, 2)
    χ.P_chol * (@view(unbiased_𝓨.xi[:, 1:num_states]))' .* weight_i .-
        χ.P_chol * (@view(unbiased_𝓨.xi[:, num_states + num_noise_states + 1:2 * num_states + num_noise_states]))' .* weight_i
end

function cov!(P, χ::AugmentedSigmaPoints, unbiased_𝓨::TransformedSigmaPoints)
    weight_0, weight_i = calc_cov_weights(χ)
    num_states = length(χ.x0)
    num_noise_states = size(χ.noise_chol, 2)
    P .= @~ χ.P_chol * (@view(unbiased_𝓨.xi[:, 1:num_states]))' .* weight_i
    P .-= @~ χ.P_chol * (@view(unbiased_𝓨.xi[:, num_states + num_noise_states + 1:2 * num_states + num_noise_states]))' .* weight_i
    P
end

function cov(unbiased_𝓨::TransformedSigmaPoints, Q::Augment)
    weight_0, weight_i = calc_cov_weights(unbiased_𝓨)
    unbiased_𝓨.x0 * unbiased_𝓨.x0' .* weight_0 .+ unbiased_𝓨.xi * unbiased_𝓨.xi' .* weight_i
end

function cov!(P, unbiased_𝓨::TransformedSigmaPoints, Q::Augment)
    weight_0, weight_i = calc_cov_weights(unbiased_𝓨)
    P .= @~ unbiased_𝓨.x0 * unbiased_𝓨.x0' .* weight_0 .+ unbiased_𝓨.xi * unbiased_𝓨.xi' .* weight_i
end