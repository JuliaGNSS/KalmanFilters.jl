abstract type AbstractSigmaPoints{T} <: AbstractMatrix{T} end

struct SigmaPoints{T} <: AbstractSigmaPoints{T}
    x0::Array{T, 1}
    xi_P_plus::Array{T, 2}
    xi_P_minus::Array{T, 2}
    SigmaPoints{T}(x0, xi_P_plus, xi_P_minus) where {T<:Real} =
        size(x0, 1) == size(xi_P_plus, 1) == size(xi_P_minus, 1) ?
        new(x0, xi_P_plus, xi_P_minus) :
        error("The length of the first dimension must be the same for all inputs")
end

struct AugmentedSigmaPoints{T} <: AbstractSigmaPoints{T}
    x0::Array{T, 1}
    xi_P_plus::Array{T, 2}
    xi_noise_plus::Array{T, 2}
    xi_P_minus::Array{T, 2}
    xi_noise_minus::Array{T, 2}
    AugmentedSigmaPoints{T}(x0, xi_P_plus, xi_noise_plus, xi_P_minus, xi_noise_minus) where {T<:Real} =
        size(x0, 1) == size(xi_P_plus, 1) == size(xi_P_minus, 1) == size(xi_noise_plus, 1) == size(xi_noise_minus, 1) ?
        new(x0, xi_P_plus, xi_noise_plus, xi_P_minus, xi_noise_minus) :
        error("The length of the first dimension must be the same for all inputs")
end

struct PseudoSigmaPoints{T} <: AbstractSigmaPoints{T}
    xi_P_plus::LowerTriangular{T}
    xi_P_minus::LowerTriangular{T}
    PseudoSigmaPoints{T}(xi_P_plus, xi_P_minus) where {T<:Real} =
        size(xi_P_plus, 1) == size(xi_P_minus, 1) ?
        new(xi_P_plus, xi_P_minus) :
        error("The length of the first dimension must be the same for all inputs")
end

struct AugmentedPseudoSigmaPoints{T} <: AbstractSigmaPoints{T}
    xi_P_plus::LowerTriangular{T}
    xi_P_minus::LowerTriangular{T}
    AugmentedPseudoSigmaPoints{T}(xi_P_plus, xi_P_minus) where {T<:Real} =
        size(xi_P_plus, 1) == size(xi_P_minus, 1) ?
        new(xi_P_plus, xi_P_minus) :
        error("The length of the first dimension must be the same for all inputs")
end

function PseudoSigmaPoints(weighted_P_chol::LowerTriangular{T}) where T
    PseudoSigmaPoints(weighted_P_chol, -weighted_P_chol)
end

function AugmentedPseudoSigmaPoints(weighted_P_chol)
    AugmentedPseudoSigmaPoints(weighted_P_chol.P, -weighted_P_chol.P)
end

Base.size(S::SigmaPoints) = (size(S.xi_P_plus, 1), size(S.xi_P_plus, 2) + size(S.xi_P_minus, 2) + 1)

Base.getindex(S::SigmaPoints{T}, inds::Vararg{Int,2}) where {T} =
    @inbounds if inds[2] == 1
        S.x0[inds[1]]
    elseif 1 < inds[2] <= size(S.xi_P_plus, 2) + 1
        S.xi_P_plus[inds[1], inds[2] - 1]
    else
        S.xi_P_minus[inds[1], inds[2] - size(S.xi_P_plus, 2) - 1]
    end

Base.setindex!(S::SigmaPoints{T}, val, inds::Vararg{Int,2}) where {T} =
    @inbounds if inds[2] == 1
        S.x0[inds[1]] = val
    elseif 1 < inds[2] <= size(S.xi_P_plus, 2) + 1
        S.xi_P_plus[inds[1], inds[2] - 1] = val
    else
        S.xi_P_minus[inds[1], inds[2] - size(S.xi_P_plus, 2) - 1] = val
    end

SigmaPoints(x0::Vector{T}, xi_P_plus::Matrix{T}, xi_P_minus::Matrix{T}) where {T} = SigmaPoints{T}(x0, xi_P_plus, xi_P_minus)

Base.BroadcastStyle(::Type{<:SigmaPoints}) = Broadcast.ArrayStyle{SigmaPoints}()

Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{SigmaPoints}}, ::Type{T}) where T =
    SigmaPoints(similar(bc.args[1].x0), similar(bc.args[1].xi_P_plus), similar(bc.args[1].xi_P_minus))

function Base.copyto!(dest::SigmaPoints, bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{SigmaPoints}})
    for idx in eachindex(bc)
        @inbounds dest[idx] = bc[idx]
    end
    dest
end

Base.size(S::AugmentedSigmaPoints) = (size(S.xi_P_plus, 1), size(S.xi_P_plus, 2) + size(S.xi_noise_plus, 2) + size(S.xi_P_minus, 2) + size(S.xi_noise_plus, 2) + 1)

Base.getindex(S::AugmentedSigmaPoints{T}, inds::Vararg{Int,2}) where {T} =
    @inbounds if inds[2] == 1
        S.x0[inds[1]]
    elseif 1 < inds[2] <= size(S.xi_P_plus, 2) + 1
        S.xi_P_plus[inds[1], inds[2] - 1]
    elseif size(S.xi_P_plus, 2) + 1 < inds[2] <= size(S.xi_P_plus, 2) + size(S.xi_noise_plus, 2) + 1
        S.xi_noise_plus[inds[1], inds[2] - size(S.xi_P_plus, 2) - 1]
    elseif size(S.xi_P_plus, 2) + size(S.xi_noise_plus, 2) + 1 < inds[2] <= size(S.xi_P_plus, 2) + size(S.xi_noise_plus, 2) + size(S.xi_P_minus, 2) + 1
        S.xi_P_minus[inds[1], inds[2] - size(S.xi_P_plus, 2) - size(S.xi_noise_plus, 2) - 1]
    else
        S.xi_noise_minus[inds[1], inds[2] - size(S.xi_P_plus, 2) - size(S.xi_noise_plus, 2) - size(S.xi_P_minus, 2) - 1]
    end

Base.setindex!(S::AugmentedSigmaPoints{T}, val, inds::Vararg{Int,2}) where {T} =
    @inbounds if inds[2] == 1
        S.x0[inds[1]] = val
    elseif 1 < inds[2] <= size(S.xi_P_plus, 2) + 1
        S.xi_P_plus[inds[1], inds[2] - 1] = val
    elseif size(S.xi_P_plus, 2) + 1 < inds[2] <= size(S.xi_P_plus, 2) + size(S.xi_noise_plus, 2) + 1
        S.xi_noise_plus[inds[1], inds[2] - size(S.xi_P_plus, 2) - 1] = val
    elseif size(S.xi_P_plus, 2) + size(S.xi_noise_plus, 2) + 1 < inds[2] <= size(S.xi_P_plus, 2) + size(S.xi_noise_plus, 2) + size(S.xi_P_minus, 2) + 1
        S.xi_P_minus[inds[1], inds[2] - size(S.xi_P_plus, 2) - size(S.xi_noise_plus, 2) - 1] = val
    else
        S.xi_noise_minus[inds[1], inds[2] - size(S.xi_P_plus, 2) - size(S.xi_noise_plus, 2) - size(S.xi_P_minus, 2) - 1] = val
    end

AugmentedSigmaPoints(x0::Vector{T}, xi_P_plus::Matrix{T}, xi_noise_plus::Matrix{T}, xi_P_minus::Matrix{T}, xi_noise_minus::Matrix{T}) where {T} =
    AugmentedSigmaPoints{T}(x0, xi_P_plus, xi_noise_plus, xi_P_minus, xi_noise_minus)

Base.BroadcastStyle(::Type{<:AugmentedSigmaPoints}) = Broadcast.ArrayStyle{AugmentedSigmaPoints}()

Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{AugmentedSigmaPoints}}, ::Type{T}) where T =
    AugmentedSigmaPoints(similar(bc.args[1].x0), similar(bc.args[1].xi_P_plus), similar(bc.args[1].xi_noise_plus), similar(bc.args[1].xi_P_minus), similar(bc.args[1].xi_noise_minus))

function Base.copyto!(dest::AugmentedSigmaPoints, bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{AugmentedSigmaPoints}})
    for idx in eachindex(bc)
        @inbounds dest[idx] = bc[idx]
    end
    dest
end

PseudoSigmaPoints(xi_P_plus::LowerTriangular{T}, xi_P_minus::LowerTriangular{T}) where {T} = PseudoSigmaPoints{T}(xi_P_plus, xi_P_minus)

Base.size(S::PseudoSigmaPoints) = (size(S.xi_P_plus, 1), size(S.xi_P_plus, 2) + size(S.xi_P_minus, 2) + 1)

Base.getindex(S::PseudoSigmaPoints{T}, inds::Vararg{Int,2}) where {T} =
    @inbounds if inds[2] == 1
        zero(T)
    elseif 1 < inds[2] <= size(S.xi_P_plus, 2) + 1
        S.xi_P_plus[inds[1], inds[2] - 1]
    else
        S.xi_P_minus[inds[1], inds[2] - size(S.xi_P_plus, 2) - 1]
    end

Base.setindex!(S::PseudoSigmaPoints{T}, val, inds::Vararg{Int,2}) where {T} =
    @inbounds if inds[2] == 1
        error("The first column cannot be set")
    elseif 1 < inds[2] <= size(S.xi_P_plus, 2) + 1
        S.xi_P_plus[inds[1], inds[2] - 1] = val
    else
        S.xi_P_minus[inds[1], inds[2] - size(S.xi_P_plus, 2) - 1] = val
    end

Base.size(S::AugmentedPseudoSigmaPoints) = (size(S.xi_P_plus, 1), 2 * size(S.xi_P_plus, 2) + 2 * size(S.xi_P_minus, 2) + 1)

Base.getindex(S::AugmentedPseudoSigmaPoints{T}, inds::Vararg{Int,2}) where {T} =
    @inbounds if inds[2] == 1
        zero(T)
    elseif 1 < inds[2] <= size(S.xi_P_plus, 2) + 1
        S.xi_P_plus[inds[1], inds[2] - 1]
    elseif size(S.xi_P_plus, 2) + 1 < inds[2] <= size(S.xi_P_plus, 2) * 2 + 1
        zero(T)
    elseif size(S.xi_P_plus, 2) * 2 + 1 < inds[2] <= size(S.xi_P_plus, 2) * 2 + size(S.xi_P_minus, 2) + 1
        S.xi_P_minus[inds[1], inds[2] - size(S.xi_P_plus, 2) * 2 - 1]
    else
        zero(T)
    end

Base.setindex!(S::AugmentedPseudoSigmaPoints{T}, val, inds::Vararg{Int,2}) where {T} =
    @inbounds if inds[2] == 1
        error("The first column cannot be set")
    elseif 1 < inds[2] <= size(S.xi_P_plus, 2) + 1
        S.xi_P_plus[inds[1], inds[2] - 1] = val
    elseif size(S.xi_P_plus, 2) + 1 < inds[2] <= size(S.xi_P_plus, 2) * 2 + 1
        error("Noise columns cannot be set")
    elseif size(S.xi_P_plus, 2) * 2 + 1 < inds[2] <= size(S.xi_P_plus, 2) * 2 + size(S.xi_P_minus, 2) + 1
        S.xi_P_minus[inds[1], inds[2] - size(S.xi_P_plus, 2) * 2 - 1] = val
    else
        error("Noise columns cannot be set")
    end

AugmentedPseudoSigmaPoints(xi_P_plus::LowerTriangular{T}, xi_P_minus::LowerTriangular{T}) where {T} =
AugmentedPseudoSigmaPoints{T}(xi_P_plus, xi_P_minus)
